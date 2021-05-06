import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DBSN_Likelihood(nn.Module):
    def __init__(self, in_ch=1, nlf_scalar=True, est_net=None):
        super().__init__()

        self.in_ch = in_ch
        self.nlf_scalar = True

        self.bsn = DBSN(in_ch=in_ch, out_ch=(in_ch+1)*in_ch)
        num_ch_nlf = 1 if self.nlf_scalar else in_ch*in_ch
        if est_net is None:
            self.estn = CNNest(in_ch=in_ch, out_ch=num_ch_nlf)
        else:
            self.estn = est_net(in_ch=in_ch, out_ch=num_ch_nlf)

    def forward(self, x):
        # forward BSN
        bsn_out = self.bsn(x)
        x_mean, mu_var = bsn_out[:,:self.in_ch,:,:], self.make_matrix_form(bsn_out[:,self.in_ch:,:,:])

        # forward noise level estimation network.
        n_sigma = self.estn(x)
        n_sigma = self.make_matrix_form(n_sigma)
        # averaging
        b,c,c,w,h = n_sigma.shape
        n_sigma = n_sigma.view(b,-1).mean(dim=1, keepdim=True).expand(b, c*c*w*h).view(b,c,c,w,h)
        
        # n_sigma = torch.full_like(n_sigma, 25.)

        return x_mean, self.make_covar_form(mu_var), n_sigma   

    def denoise(self, x):
        '''
        inferencing function for denoising.
        because forward operation isn't for denoising.
        (see more details at section 3.3 in D-BSN paper)
        '''
        x_mean, mu_var, n_sigma = self.forward(x)

        x_reshape = x.permute(0,2,3,1).unsqueeze(-1) # b,w,h,c,1
        x_mean = x_mean.permute(0,2,3,1).unsqueeze(-1) # b,w,h,c,1
        mu_var = mu_var.permute(0,3,4,1,2) # b,w,h,c,c
        n_sigma = n_sigma.permute(0,3,4,1,2) # b,w,h,c,c or b,w,h,1,1

        if self.nlf_scalar:
            b, w, h, c, _ = x_mean.shape
            eye = torch.eye(c).view(-1)
            if x_mean.is_cuda: eye = eye.cuda()
            n_sigma = torch.pow(n_sigma, 2).squeeze(-1) * eye.repeat(b,w,h,1) #b,w,h,c**2
            n_sigma = n_sigma.view(b,w,h,c,c)

        y_var_inv = torch.inverse(mu_var + n_sigma) # b,w,h,c,c
        cross_sum = torch.matmul(mu_var, x_reshape) + torch.matmul(n_sigma, x_mean)

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
        x = self.head_conv11(x)

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

class CNNest(nn.Module):
    def __init__(self, in_ch=3, out_ch=9, num_layer=5, base_ch=16):
        super().__init__()

        layer = [nn.Conv2d(in_channels=in_ch, out_channels=base_ch, kernel_size=1)]
        for i in range(num_layer-2):
            layer.append(nn.ReLU(inplace=True))
            layer.append(nn.Conv2d(in_channels=base_ch, out_channels=base_ch, kernel_size=1))
        layer.append(nn.ReLU(inplace=True))
        layer.append(nn.Conv2d(in_channels=base_ch, out_channels=out_ch, kernel_size=1))
        self.body = nn.Sequential(*layer)
    
    def forward(self, x):
        return self.body(x)

class CNNest3(nn.Module):
    def __init__(self, in_ch=3, out_ch=9, num_layer=5, base_ch=16):
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

class DBSN_Likelihood3(DBSN_Likelihood):
    def __init__(self, in_ch=3):
        super().__init__(in_ch=in_ch, est_net=DBSN)

if __name__ == "__main__":
    t = torch.randn(16,3,64,64)

    model = DBSN_Likelihood()

    mu, s, n_s = model(t)

    print(mu.shape)
    print(s.shape)
    print(n_s.shape)
    