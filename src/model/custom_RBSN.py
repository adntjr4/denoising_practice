import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..util.util import pixel_shuffle_down_sampling, pixel_shuffle_up_sampling
from . import get_model_object


eps = 1e-6

class RBSN_fusion(nn.Module):
    def __init__(self, in_ch, nlf_net=None, fusion_net='DnCNN_B', pd=4, real=True):
        super().__init__()
        
        self.in_ch      = in_ch
        self.nlf_net    = nlf_net(real=real) if nlf_net is not None else None
        self.pd         = pd
        self.real       = real
        self.fusion_net = get_model_object(fusion_net)(in_ch=in_ch, out_ch=in_ch) if fusion_net is not None else None

        bsn_out_ch = self.in_ch if nlf_net is None else 2*self.in_ch
        self.bsn = DBSN(in_ch=in_ch, out_ch=bsn_out_ch)

    def denoise(self, x):
        _, _, _, _, denoised = self.forward(x)
        return denoised

    def forward(self, x):
        # PD
        pd_x = pixel_shuffle_down_sampling(x, self.pd)

        # blind-spot network forward
        x_mean = self.bsn(pd_x)

        # post-forward processing
        if self.nlf_net is not None:
            # network forward
            x_mean, mu_var = x_mean[:,:self.in_ch], x_mean[:,self.in_ch:]
            n_var = self.nlf_net(pd_x)

            # inverse PD
            x_mean = pixel_shuffle_up_sampling(x_mean, self.pd)
            mu_var = pixel_shuffle_up_sampling(mu_var, self.pd)
            n_var  = pixel_shuffle_up_sampling(n_var, self.pd)

            # make diagonal matrix
            mu_var = self.make_diag_covar_form(mu_var)
            n_var  = self.make_diag_covar_form(n_var)

            # first denoising (using bayes inference)
            pd_denoised = self.bayes_inference(x, x_mean, mu_var, n_var)
        else:
            # inverse PD
            x_mean = pixel_shuffle_up_sampling(x_mean, self.pd)
            mu_var, n_var = None, None
            pd_denoised = x_mean

        # fusion module
        if self.fusion_net is not None:
            denoised = self.fusion_net(pd_denoised)
        else:
            denoised = pd_denoised
                
        return x_mean, mu_var, n_var, pd_denoised, denoised

    def bayes_inference(self, x, x_mean, mu_var, n_var):
        b,c,w,h = x_mean.shape
        x_reshape = x.permute(0,2,3,1).unsqueeze(-1) # b,w,h,c,1
        x_mean_reshape = x_mean.permute(0,2,3,1).unsqueeze(-1) # b,w,h,c,1
        mu_var_reshape = mu_var.permute(0,3,4,1,2) # b,w,h,c,c
        n_var_reshape = n_var.permute(0,3,4,1,2) # b,w,h,c,c
        epsI = eps * torch.eye(c, device=x_mean.device).repeat(b,w,h,1,1)

        mu_var_inv = torch.inverse(mu_var + epsI)
        n_var_inv  = torch.inverse(n_var + epsI)

        y_var_inv = torch.inverse(mu_var_inv + n_var_inv + epsI) # b,w,h,c,c
        cross_sum = torch.matmul(n_var_inv, x_reshape) + torch.matmul(mu_var_inv, x_mean)

        results = torch.matmul(y_var_inv, cross_sum).squeeze(-1) # b,w,h,c

        return results.permute(0,3,1,2) # b,c,w,h

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

class RBSN(nn.Module):
    '''
    Main differences are I divide network for 3 output respectively. (which are x_mean, mu_var. n_var)
    I think mu_var don't need to use blind-spot network.
    '''
    def __init__(self, in_ch=3, nlf_net=None, real=True):
        super().__init__()

        self.in_ch = in_ch

        self.bsn = DBSN(in_ch=in_ch, out_ch=in_ch*2)

        if nlf_net == None:
            self.nlf_net = CNNest(in_ch=in_ch, out_ch=in_ch)
        else:
            self.nlf_net = nlf_net(real=real)

    def forward(self, x):
        b,c,w,h = x.shape

        # forward blind-spot network
        x_mean = self.bsn(x)
        x_mean, mu_var = x_mean[:,:self.in_ch], x_mean[:,self.in_ch:]

        # forward mu-variance network
        mu_var = self.make_diag_covar_form(mu_var)

        # forward noise-level estimation network
        n_var = self.nlf_net(x)
        n_var = self.make_diag_covar_form(n_var)
        n_var = n_var.mean((-1,-2)).repeat(w,h,1,1,1).permute(2,3,4,0,1)

        # n_var = 625 * torch.eye(3, device=x.device).repeat(b,w,h,1,1)
        # n_var = n_var.permute(0,3,4,1,2)

        return x_mean, mu_var, n_var

    def denoise(self, x):
        '''
        inferencing function for denoising.
        because forward operation isn't for denoising.
        (see more details at section 3.3 in D-BSN paper)
        '''
        x_mean, mu_var, n_var = self.forward(x)

        b,c,w,h = x_mean.shape
        x_reshape = x.permute(0,2,3,1).unsqueeze(-1) # b,w,h,c,1
        x_mean = x_mean.permute(0,2,3,1).unsqueeze(-1) # b,w,h,c,1
        mu_var = mu_var.permute(0,3,4,1,2) # b,w,h,c,c
        n_var = n_var.permute(0,3,4,1,2) # b,w,h,c,c
        epsI = eps * torch.eye(c, device=x_mean.device).repeat(b,w,h,1,1)

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


    # def make_covar_form(self, m):
    #     '''
    #     multiply upper triangular part of matrix to make validate covariance matrix.
    #     m      : b,c,w,h
    #     return : b,c,c,w,h
    #     '''
    #     mp = torch.square(m.permute(0,2,3,1))
    #     mp = torch.diag_embed(mp)
    #     return mp.permute(0,3,4,1,2)

class CNNest(nn.Module):
    def __init__(self, in_ch=3, out_ch=9, num_layer=7, base_ch=64):
        super().__init__()

        layer = [nn.Conv2d(in_channels=in_ch, out_channels=base_ch, kernel_size=3, padding=1)]
        for i in range(num_layer-2):
            layer.append(nn.ReLU(inplace=True))
            layer.append(nn.Conv2d(in_channels=base_ch, out_channels=base_ch, kernel_size=3, padding=1))
        layer.append(nn.ReLU(inplace=True))
        layer.append(nn.Conv2d(in_channels=base_ch, out_channels=out_ch, kernel_size=3, padding=1))
        self.body = nn.Sequential(*layer)
    
    def forward(self, x):
        return self.body(x)

class DBSN(nn.Module):
    def __init__(self, num_module=5, in_ch=1, out_ch=1, base_ch=64):
        super().__init__()

        assert base_ch%2 == 0, "base channel should be divided with 2"

        self.head_conv11 = nn.Conv2d(in_ch, base_ch, kernel_size=1)

        self.central_conv33 = CentralMaskedConv2d(base_ch, base_ch, kernel_size=3, stride=1, padding=1)
        self.central_conv55 = CentralMaskedConv2d(base_ch, base_ch, kernel_size=5, stride=1, padding=2)

        self.mdc_branch1 = nn.Sequential(*[MDC(stride=2, in_ch=base_ch) for _ in range(num_module)])
        self.mdc_branch2 = nn.Sequential(*[MDC(stride=3, in_ch=base_ch) for _ in range(num_module)])

        t = []
        t.append(nn.Conv2d(base_ch*2, base_ch//2, kernel_size=1))
        t.append(nn.ReLU(inplace=True))
        for i in range(2):
            t.append(nn.Conv2d(base_ch//2, base_ch//2, kernel_size=1))
            t.append(nn.ReLU(inplace=True))
        t.append(nn.Conv2d(base_ch//2, out_ch, kernel_size=1))
        self.tail = nn.Sequential(*t)

    def forward(self, x):
        x = F.relu(self.head_conv11(x), inplace=True)

        br1 = self.central_conv33(x)
        br2 = self.central_conv55(x)

        br1 = self.mdc_branch1(br1)
        br2 = self.mdc_branch2(br2)

        x = torch.cat([br1, br2], dim=1)

        return self.tail(x)

class MDC(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()

        self.br1_conv11 = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.br2_conv11 = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.br3_conv11 = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.tail_conv11 = nn.Conv2d(3*in_ch, in_ch, kernel_size=1)

        self.br1_conv33_1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride)
        self.br1_conv33_2 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride)
        self.br2_conv33   = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride)

    def forward(self, x):
        residual = x

        branch1 = F.relu(self.br1_conv11(x), inplace=True)
        branch1 = F.relu(self.br1_conv33_1(branch1))
        branch1 = F.relu(self.br1_conv33_2(branch1))

        branch2 = F.relu(self.br2_conv11(x), inplace=True)
        branch2 = F.relu(self.br2_conv33(branch2))

        branch3 = F.relu(self.br3_conv11(x), inplace=True)

        x = torch.cat([branch1, branch2, branch3], dim=1)

        x = F.relu(self.tail_conv11(x))

        return residual + x

class CentralMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH//2, kH//2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

class LocalVarianceNet(nn.Module):
    def __init__(self, in_ch, window_size):
        super().__init__()

        self.in_ch = in_ch
        self.ws = window_size

        self.average_net = nn.Conv2d(in_ch, in_ch, kernel_size=self.ws, padding=self.ws//2, padding_mode='circular', bias=False, groups=in_ch)
        self.average_net.weight.data.fill_(1/(self.ws**2))
        for param in self.average_net.parameters():
            param.requires_grad = False

    def forward(self, x):
        squared = torch.square(x)
        return self.average_net(squared) - torch.square(self.average_net(x))        

class NLFNet(nn.Module):
    def __init__(self, window_size=9, real=False):
        super().__init__()

        self.gamma = 2.0
        self.real = real

        self.lvn3 = LocalVarianceNet(3, window_size)
        self.lvn1 = LocalVarianceNet(1, window_size)

    def forward(self, x):
        b,c,w,h = x.shape

        alpha_map = self.lvn3(x).sum(1, keepdim=True)/3
        beta_map = self.lvn1(x.sum(1, keepdim=True)/3)

        weight = self.lvn3(x-x[:, [2,1,0]]).sum(1, keepdim=True)/3

        w_sum = weight.sum()
        weight = torch.exp(-self.gamma * w*h * weight / w_sum)

        # weight = F.softmax(weight.view(1,1,w*h), dim=2).view(1,1,w,h)

        w_sum = weight.sum()
        if self.real:
            RGB_ratio = [2.0, 1.0, 2.0]
            nlf = 9/4*(weight*(alpha_map-beta_map)).sum() / w_sum
            nlf = torch.stack([ratio * torch.sqrt(nlf) for ratio in RGB_ratio])
            return nlf.repeat(b,w,h,1).permute(0,3,1,2)
        else:
            nlf = 3/2*(weight*(alpha_map-beta_map)).sum() / w_sum
            return torch.sqrt(nlf).repeat(b,w,h,1).permute(0,3,1,2)

class RBSN_nlf(RBSN):
    def __init__(self, real=True):
        super().__init__(in_ch=3, nlf_net=NLFNet, real=real)