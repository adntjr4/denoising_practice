import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.util.util import rot_hflip_img


class shifted_conv(nn.Conv2d):
    def __init__(self, in_ch, out_ch, k_size):
        self.k = math.floor(k_size/2)
        super().__init__(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, padding=self.k)

    def forward(self, x):
        # padding on top
        pad_x = F.pad(x, (0,0,self.k,0))
        # forward
        pad_x = super().forward(pad_x)
        # crop out bottom
        return pad_x[:,:,:-self.k,:]

class shifted_maxpool2d_kernel2(nn.MaxPool2d):
    def __init__(self):
        super().__init__(kernel_size=2)

    def forward(self, x):
        # padding one row on top
        pad_x = F.pad(x, (0,0,1,0))
        # forward
        pad_x = super().forward(pad_x[:,:,:-1,:])
        # crop bottom on row
        return pad_x

class bsn_unet(nn.Module):
    def __init__(self, in_ch, n_depth, base_ch):
        self.n_depth = n_depth
        self.in_ch = in_ch
        self.base_ch = base_ch
        super().__init__()

        self.act = nn.LeakyReLU(0.1, inplace=True)

        self.head_conv = shifted_conv(in_ch=in_ch, out_ch=base_ch, k_size=3)
        self.down_convs = nn.ModuleList([shifted_conv(in_ch=base_ch, out_ch=base_ch, k_size=3) for i in range(self.n_depth)])
        self.maxpools = nn.ModuleList([shifted_maxpool2d_kernel2() for i in range(self.n_depth)])
        
        self.middle = shifted_conv(in_ch=base_ch, out_ch=base_ch, k_size=3)

        self.de_convs = nn.ModuleList([nn.ConvTranspose2d(in_channels=base_ch, out_channels=base_ch, kernel_size=2, stride=2)])
        self.de_convs.extend([nn.ConvTranspose2d(in_channels=base_ch*2, out_channels=base_ch*2, kernel_size=2, stride=2) for i in range(self.n_depth)])

        self.up_convs = nn.ModuleList([nn.Sequential(*[shifted_conv(base_ch*2, base_ch*2, k_size=3), self.act, shifted_conv(base_ch*2, base_ch*2, k_size=3), self.act])])
        self.up_convs.extend([nn.Sequential(*[shifted_conv(base_ch*3, base_ch*2, k_size=3), self.act, shifted_conv(base_ch*2, base_ch*2, k_size=3), self.act]) for i in range(self.n_depth-2)])
        self.up_convs.extend([nn.Sequential(*[shifted_conv(base_ch*2+in_ch, base_ch*2, k_size=3), self.act, shifted_conv(base_ch*2, base_ch*2, k_size=3), self.act])])

    def forward(self, x):
        skips = [x]
        x = self.act(self.head_conv(x))

        # down
        for l in range(self.n_depth):
            x = self.act(self.down_convs[l](x))
            x = self.maxpools[l](x)
            skips.append(x)
            
        # middle
        x = self.act(self.middle(x))
        skips = list(reversed(skips[:self.n_depth]))

        # up
        for l in range(self.n_depth):
            x = self.de_convs[l](x)
            x = torch.cat([x, skips[l]], dim=1)
            x = self.up_convs[l](x)

        return x

class Laine19(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, n_depth=5, base_ch=48):
        self.n_depth = n_depth
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.base_ch = base_ch
        super().__init__()

        self.unet = bsn_unet(in_ch=in_ch, n_depth=n_depth, base_ch=base_ch)
        self.tail = nn.Sequential(*[nn.Conv2d(in_channels=base_ch*8, out_channels=base_ch*8, kernel_size=1), nn.LeakyReLU(0.1, inplace=True),
                                    nn.Conv2d(in_channels=base_ch*8, out_channels=base_ch*2, kernel_size=1),  nn.LeakyReLU(0.1, inplace=True),
                                    nn.Conv2d(in_channels=base_ch*2, out_channels=out_ch,    kernel_size=1)])

    def forward(self, x):
        # handle image size for power of 2
        b,c,h,w = x.shape
        multiple = 2**self.n_depth
        pad_h, pad_w = multiple-1-(h+multiple-1)%multiple, multiple-1-(w+multiple-1)%multiple
        x = F.pad(x, (0,pad_h,0,pad_w))

        # rotate
        x = [x, rot_hflip_img(x, 1), rot_hflip_img(x, 2), rot_hflip_img(x, 3)]

        # Unet
        for idx, single_img in enumerate(x):
            x[idx] = self.unet(single_img)

        # rotate, concat
        for idx, single_img in enumerate(x):
            x[idx] = rot_hflip_img(F.pad(single_img, (0,0,1,0))[:,:,:-1,:], -idx)
        x = torch.cat(x, dim=1)

        # remvoe padded area
        x = x[:,:,:h,:w]

        # tail (1x1 convs)
        x = self.tail(x)

        return x

class Laine19_Likelihood(nn.Module):
    '''
    Module for using bayesian inference step from Laine et al.

    For blind-spot network above Laine19 network is used.
    Estimation network return noise level as scalar (sigma)
    '''
    def __init__(self, in_ch=1, est_net=None):
        super().__init__()
        self.in_ch = in_ch

        self.bsn = Laine19(in_ch=in_ch, out_ch=in_ch*(in_ch+1))
        if est_net is None:
            self.estn = Laine19(in_ch=in_ch, out_ch=1)
        else:
            self.estn = est_net(in_ch=in_ch, out_ch=1)

    def forward(self, x):
        '''
        Args:
            x : input of network
        Returns:
            x_mean (Tensor)  : mean of clean signal        [b,c,h,w]
            x_var (Tensor)   : covariance of clean signal  [b,c,c,h,w]
            n_sigma (Tensor) : standard deviation of noise [b]
        '''
        # get outputs (mean of clean, variance of clean, std of noise)
        b,c,w,h = x.shape
        bsn_out = self.bsn(x)
        x_mean = bsn_out[:,:self.in_ch,:,:]
        x_var = self.make_covar_form(self.make_matrix_form(bsn_out[:,self.in_ch:,:,:]))
        n_sigma = self.estn(x).view(b,-1).mean(dim=1)

        return x_mean, x_var, n_sigma

    def denoise(self, x):
        '''
        inferencing function for denoising.
        because forward operation isn't for denoising.
        '''
        x_mean, x_var, n_sigma = self.forward(x)

        # reshape input, mean of clean, covariance of clean
        x_reshape = x.permute(0,2,3,1).unsqueeze(-1)   # b,w,h,c,1
        x_mean = x_mean.permute(0,2,3,1).unsqueeze(-1) # b,w,h,c,1
        x_var = x_var.permute(0,3,4,1,2)               # b,w,h,c,c

        # reshape std of noise
        b,w,h,c,_ = x_mean.shape
        n_var = torch.eye(c).view(-1)
        if x_mean.is_cuda: n_var = n_var.cuda()
        n_var = torch.pow(n_sigma, 2) * n_var.repeat(b,w,h,1).permute(1,2,3,0) # w,h,c**2,b
        n_var = n_var.permute(3,0,1,2) # b,w,h,c**2
        n_var = n_var.view(b,w,h,c,c)  # b,w,h,c,c

        # denoising
        y_var_inv = torch.inverse(x_var + n_var) # b,w,h,c,c
        cross_sum = torch.matmul(x_var, x_reshape) + torch.matmul(n_var, x_mean)
        denoised = torch.matmul(y_var_inv, cross_sum).squeeze(-1) # b,w,h,c

        return denoised.permute(0,3,1,2) # b,c,w,h

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

if __name__ == '__main__':
    net = Laine19()
    # net = shifted_conv(3, 3, 3)
    # print(net)
    batch = torch.randn(16,3,48,48)
    print(net(batch).shape)
