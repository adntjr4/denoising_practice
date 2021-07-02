import math

import torch
from torch._C import has_cuda
import torch.nn as nn
import torch.nn.functional as F

from ..util.util import pixel_shuffle_down_sampling, pixel_shuffle_up_sampling
from . import regist_model, get_model_object
from .DBSN import DBSN


eps = 1e-6

@regist_model
class RBSN(nn.Module):
    def __init__(self, pd=4, pd_pad=0, eval_mu=False, noise_correction=False):
        super().__init__()

        self.in_ch   = 3
        self.avg_size_n_var = 21
        self.avg_size_nc = 9


        self.pd      = pd
        self.pd_pad  = pd_pad
        self.eval_mu = eval_mu
        self.nc      = noise_correction

        self.bsn = DBSN(in_ch=self.in_ch, out_ch=self.in_ch)
        self.mu_var_net = DBSN(in_ch=self.in_ch, out_ch=self.in_ch**2, num_module=3, base_ch=16)
        # self.mu_var_net = CNNest(in_ch=self.in_ch, out_ch=self.in_ch**2)

        self.nlf_net = CNNest(in_ch=self.in_ch, out_ch=self.in_ch)
        self.nlf_est = NLFNet(real=True)

        self.avg_net_n_var = LocalMeanNet(self.in_ch, self.avg_size_n_var)
        if self.nc: self.avg_net_nc = LocalMeanNet(self.in_ch, self.avg_size_nc)

    def forward(self, x):
        b,c,w,h = x.shape

        # noise correction
        est_nlf = self.nlf_est(x)
        if self.nc:
            x = self.clipping_correction(x, est_nlf)

        # PD
        pd_x = pixel_shuffle_down_sampling(x, f=self.pd, pad=self.pd_pad)
        
        # forward blind-spot network
        pd_x_mean = self.bsn(pd_x)
        
        # inverse PD
        x_mean = pixel_shuffle_up_sampling(pd_x_mean, f=self.pd, pad=self.pd_pad)

        # forward mu_var, n_var network
        mu_var = self.mu_var_net(x_mean.detach())
        mu_var = self.make_covar_form(self.make_matrix_form(mu_var))
        mu_var = mu_var.sign() * mu_var.abs().clamp(min=eps).sqrt()

        n_var = self.avg_net_n_var(self.nlf_net(x))
        n_var = self.make_diag_covar_form(n_var)

        return x_mean, mu_var, n_var
    
    def denoise(self, x):
        '''
        inferencing function for denoising.
        because forward operation isn't for denoising.
        (see more details at section 3.3 in D-BSN paper)
        '''
        x_mean, mu_var, n_var = self.forward(x)

        if self.eval_mu:
            return x_mean

        b,c,w,h = x_mean.shape
        x_reshape = x.permute(0,2,3,1).unsqueeze(-1) # b,w,h,c,1
        x_mean = x_mean.permute(0,2,3,1).unsqueeze(-1) # b,w,h,c,1
        mu_var = mu_var.permute(0,3,4,1,2) # b,w,h,c,c
        n_var = n_var.permute(0,3,4,1,2) # b,w,h,c,c
        epsI = eps * torch.eye(c, device=x_mean.device).view(1,1,1,c,c).expand(b,w,h,c,c)

        mu_var_inv = torch.inverse(mu_var + epsI)
        n_var_inv  = torch.inverse(n_var + epsI)

        y_var_inv = torch.inverse(mu_var_inv + n_var_inv + epsI) # b,w,h,c,c
        cross_sum = torch.matmul(n_var_inv, x_reshape) + torch.matmul(mu_var_inv, x_mean)

        results = torch.matmul(y_var_inv, cross_sum).squeeze(-1) # b,w,h,c

        return results.permute(0,3,1,2) # b,c,w,h

    def make_matrix_form(self, m):
        '''
        m      : b,c**2,w,h
        return : b,c,c,w,h
        '''
        b,cc,w,h = m.shape
        assert cc in [1,9], 'number of channel should be square of number of input channel'
        c = int(math.sqrt(cc))
        return m.view(b,c,c,w,h)

    def make_covar_form(self, m):
        '''
        multiply upper triangular part of matrix to make validate covariance matrix.
        m      : b,c,c,w,h
        return : b,c,c,w,h
        '''
        tri_m = torch.triu(m.permute(0,3,4,1,2))
        co_mat = torch.matmul(torch.transpose(tri_m,3,4), tri_m)
        return co_mat.permute(0,3,4,1,2)

    def make_diag_covar_form(self, m):
        '''
        square values of diagonal for validate matrix 
        m      : b,c,w,h
        return : b,c,c,w,h
        '''
        diag = torch.square(m)
        diag = diag.permute(0,2,3,1)       # b,w,h,c
        diag = torch.diag_embed(diag)   # b,w,h,c,c
        return diag.permute(0,3,4,1,2)

    def clipping_correction(self, x, nlf):
        '''
        clipping correction
        '''
        b,_,_,_ = x.shape
        nlf = nlf.view(b,-1).mean(-1)
        mean = self.avg_net_nc(x)
        b,c,w,h = x.shape
        for c_idx in range(c):
            x[:,c_idx] -= (nlf.view(b,1,1)*torch.abs_(torch.randn((b,w,h), device=x.device)) - mean[:,c_idx]) * (x[:,c_idx]<1.0)
        return x

class ResBlock(nn.Module):
    def __init__(self, base_ch, bn=True):
        super().__init__()
        layer = [nn.Conv2d(in_channels=base_ch, out_channels=base_ch, kernel_size=3, padding=1)]
        layer.append(nn.LeakyReLU(inplace=True))
        layer.append(nn.Conv2d(in_channels=base_ch, out_channels=base_ch, kernel_size=3, padding=1))
        if bn: layer.append(nn.BatchNorm2d(base_ch))
        self.body = nn.Sequential(*layer)

    def forward(self, x):
        return x + self.body(x)

class CNNest(nn.Module):
    def __init__(self, in_ch=3, out_ch=9, num_layer=4, base_ch=16):
        super().__init__()
        layer = [nn.Conv2d(in_channels=in_ch, out_channels=base_ch, kernel_size=1)]
        layer.append(nn.LeakyReLU(inplace=True))
        for i in range(num_layer):
            layer.append(ResBlock(base_ch, bn=False))
        layer.append(nn.LeakyReLU(inplace=True))
        layer.append(nn.Conv2d(in_channels=base_ch, out_channels=out_ch, kernel_size=1))
        self.body = nn.Sequential(*layer)
    
    def forward(self, x):
        return self.body(x)

class LocalMeanNet(nn.Conv2d):
    def __init__(self, in_ch, kernel_size):
        super().__init__(in_ch, in_ch, kernel_size=kernel_size, padding=kernel_size//2, padding_mode='circular', bias=False, groups=in_ch)
        self.weight.data.fill_(1/(kernel_size**2))
        for param in self.parameters():
            param.requires_grad = False

class LocalVarianceNet(nn.Module):
    def __init__(self, in_ch, window_size):
        super().__init__()

        self.in_ch = in_ch
        self.ws = window_size

        self.average_net = LocalMeanNet(in_ch, self.ws)

    def forward(self, x):
        squared = torch.square(x)
        return self.average_net(squared) - torch.square(self.average_net(x))        

class NLFNet(nn.Module):
    def __init__(self, window_size=9, real=False):
        super().__init__()

        self.gamma = 0.5
        self.real = real

        self.lvn3 = LocalVarianceNet(3, window_size)
        self.lvn1 = LocalVarianceNet(1, window_size)

    def forward(self, x):
        b,c,w,h = x.shape

        alpha_map = self.lvn3(x).sum(1, keepdim=True)/3
        beta_map = self.lvn1(x.sum(1, keepdim=True)/3)

        weight = self.lvn3(x-x[:, [2,1,0]]).sum(1, keepdim=True)/3

        w_sum = torch.clamp(weight.sum((-1,-2,-3)), eps)

        weight = torch.exp(-self.gamma * w*h * weight / w_sum.view(b,1,1,1))

        # weight = F.softmax(weight.view(1,1,w*h), dim=2).view(1,1,w,h)

        w_sum = torch.clamp(weight.sum((-1,-2,-3)), eps)

        if self.real:
            nlf = 9/4*(weight*(alpha_map-beta_map)).sum((-1,-2,-3)) / w_sum
            nlf = torch.nan_to_num(nlf, nan=eps)
            return torch.sqrt(torch.clamp(nlf, eps))
        else:
            nlf = 3/2*(weight*(alpha_map-beta_map)).sum((-1,-2,-3)) / w_sum
            nlf = torch.nan_to_num(nlf, nan=eps)
            return torch.sqrt(torch.clamp(nlf, eps))
