import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as torch_distb

eps = 1e-6

class CLtoN_D(nn.Module):
    def __init__(self, n_ch_in=3, n_ch=64, n_block=6):
        super().__init__()
        
        bn = False

        self.n_ch_in = n_ch_in
        self.n_ch = n_ch
        self.n_block = n_block

        self.head = nn.Sequential(nn.Conv2d(self.n_ch_in, self.n_ch, kernel_size=3, padding=1, bias=True),
                                  nn.PReLU())

        layers = [ResBlock(n_ch=self.n_ch, kernel_size=3, act='PReLU', bias=True, bn=bn) for _ in range(self.n_block)]
        self.body = nn.Sequential(*layers)

        self.tail = nn.Conv2d(self.n_ch, 1, kernel_size=3, padding=1, bias=True)
    
    def forward(self, img_Gout):
        x = self.head(img_Gout)
        x = self.body(x)
        x = self.tail(x)
        
        return x

class CLtoN_D_one_out(nn.Module):
    def __init__(self, n_ch_in=3, n_ch=64, n_block=6):
        super().__init__()

        self.n_ch_in = n_ch_in
        self.n_ch = n_ch
        self.n_block = n_block

        self.head = nn.Sequential(nn.Conv2d(self.n_ch_in, self.n_ch, kernel_size=3, padding=1, bias=True),
                                  nn.PReLU())

        layers = [ResBlock(n_ch=self.n_ch, kernel_size=3, act='PReLU', bias=True) for _ in range(self.n_block)]
        self.body = nn.Sequential(*layers)

        tail = []
        tail.append(nn.Conv2d(self.n_ch, self.n_ch, kernel_size=3, padding=1, stride=2, bias=True))
        tail.append(nn.Conv2d(self.n_ch, self.n_ch, kernel_size=3, padding=1, stride=2, bias=True))
        tail.append(nn.Conv2d(self.n_ch, self.n_ch, kernel_size=3, padding=1, stride=2, bias=True))
        tail.append(nn.Conv2d(self.n_ch, self.n_ch, kernel_size=3, padding=1, stride=2, bias=True))
        tail.append(nn.Conv2d(self.n_ch, self.n_ch, kernel_size=3, padding=1, stride=2, bias=True))
        tail.append(nn.Conv2d(self.n_ch, 1, kernel_size=3, bias=True))
        self.tail = nn.Sequential(*tail)
    
    def forward(self, img_Gout):
        x = self.head(img_Gout)
        x = self.body(x)
        x = self.tail(x)
        
        return x

class LtoN_D_one_out(CLtoN_D_one_out):
    def __init__(self):
        super().__init__(n_ch_in=1)

class CLtoN_G(nn.Module):
    def __init__(self, n_ch_in=3, n_ch_out=3, n_ch=64, n_rand=32, n_ext_block=5, n_indep_block=3, n_dep_block=2,
                    pipe_indep=False, pipe_dep=True, pipe_conv1=True, pipe_conv3=False):
        super().__init__()

        bn = False

        self.n_ch_in = n_ch_in
        self.n_ch_out = n_ch_out
        self.n_ch = n_ch
        self.n_rand = n_rand

        self.n_ext_block = n_ext_block
        self.n_indep_block = n_indep_block
        self.n_dep_block = n_dep_block

        self.pipe_indep = pipe_indep
        self.pipe_dep   = pipe_dep
        self.pipe_conv1 = pipe_conv1
        self.pipe_conv3 = pipe_conv3

        # feat extractor
        if self.pipe_dep:
            self.ext_head = nn.Sequential(nn.Conv2d(n_ch_in, n_ch, 3, padding=1, bias=True, padding_mode='reflect'),
                                          nn.PReLU(),
                                          nn.Conv2d(n_ch, 2*n_ch, 3, padding=1, bias=True))
            self.ext_merge = nn.Sequential(nn.Conv2d(2*n_ch+n_rand, 2*n_ch, 3, padding=1, bias=True),
                                           nn.PReLU())
            self.ext = nn.Sequential(*[ResBlock(n_ch=2*n_ch, kernel_size=3, act='PReLU', bias=True, bn=bn) for i in range(self.n_ext_block)])

        # pipe-indep
        if self.pipe_indep:
            if self.pipe_conv1:
                self.pipe_indep_1 = nn.Sequential(*[ResBlock(n_ch, kernel_size=1, act='PReLU', bias=True, bn=bn) for _ in range(n_indep_block)])
            if self.pipe_conv3:
                self.pipe_indep_3 = nn.Sequential(*[ResBlock(n_ch, kernel_size=3, act='PReLU', bias=True, bn=bn) for _ in range(n_indep_block)])

        # pipe-dep
        if self.pipe_dep:
            if self.pipe_conv1:
                self.pipe_dep_1 = nn.Sequential(*[ResBlock(n_ch, kernel_size=1, act='PReLU', bias=True, bn=bn) for i in range(n_dep_block)])
            if self.pipe_conv3:
                self.pipe_dep_3 = nn.Sequential(*[ResBlock(n_ch, kernel_size=3, act='PReLU', bias=True, bn=bn) for i in range(n_dep_block)])

        # T tail
        self.T_tail = nn.Conv2d(n_ch, n_ch_out, kernel_size=1, padding=0, bias=True)

    def forward(self, img_CL, rand_vec=None):
        (N, C, H, W) = img_CL.size()

        noise = []

        if self.pipe_dep:
            # random vector (is None sample it)
            if rand_vec is None:
                rand_vec = torch.randn((N, self.n_rand), device=img_CL.device)

            # random vector repeat (to same device)
            rand_vec_map = rand_vec.unsqueeze(-1).unsqueeze(-1).repeat(1,1,H,W)

            # feat extractor
            feat_CL = self.ext_head(img_CL)
            list_cat = [feat_CL, rand_vec_map]
            feat_CL = self.ext_merge(torch.cat(list_cat, 1))
            feat_CL = self.ext(feat_CL)

            # make initial dep noise feature
            feat_noise_dep = torch_distb.Normal(loc=feat_CL[:,:self.n_ch,:,:], scale=torch.clip(feat_CL[:,self.n_ch:,:,:], eps)).rsample()
            feat_noise_dep.to(img_CL.device)

            # pipe-dep
            noise_dep = []
            if self.pipe_conv1:
                noise_dep.append(self.pipe_dep_1(feat_noise_dep))
            if self.pipe_conv3:
                noise_dep.append(self.pipe_dep_3(feat_noise_dep))
            noise_dep = sum(noise_dep)

            noise.append(noise_dep)

        if self.pipe_indep:
            # make initial indep noise feature
            # feat_noise_indep = torch.rand_like(feat_noise_dep, requires_grad=True)
            feat_noise_indep = torch.randn((N,self.n_ch,H,W), device=img_CL.device)

            # pipe-indep
            noise_indep = []
            if self.pipe_conv1:
                noise_indep.append(feat_noise_indep)
            if self.pipe_conv3:
                noise_indep.append(feat_noise_indep)
            noise_indep = sum(noise_indep)

            noise.append(noise_indep)

        # pipe-merge
        noise = self.T_tail(sum(noise))

        return noise


class LtoN_D(CLtoN_D):
    def __init__(self):
        super().__init__(n_ch_in=1)

class LtoN_G(CLtoN_G):
    def __init__(self):
        super().__init__(n_ch_in=1, n_ch_out=1)

class CLtoN_G_indep_1(CLtoN_G):
    def __init__(self):
        super().__init__(pipe_dep=False, pipe_conv3=False)

class CLtoN_G_indep_3(CLtoN_G):
    def __init__(self):
        super().__init__(pipe_dep=False, pipe_conv1=False)

class CLtoN_G_indep_13(CLtoN_G):
    def __init__(self):
        super().__init__(pipe_dep=False)

class CLtoN_G_dep_1(CLtoN_G):
    def __init__(self):
        super().__init__(pipe_indep=False, pipe_conv3=False)

class CLtoN_G_dep_3(CLtoN_G):
    def __init__(self):
        super().__init__(pipe_indep=False, pipe_conv1=False)

class CLtoN_G_dep_13(CLtoN_G):
    def __init__(self):
        super().__init__(pipe_indep=False)

class CLtoN_G_indep_dep_1(CLtoN_G):
    def __init__(self):
        super().__init__(pipe_conv3=False)

class CLtoN_G_indep_dep_3(CLtoN_G):
    def __init__(self):
        super().__init__(pipe_conv1=False)


class ResBlock(nn.Module):
    def __init__(self, n_ch, kernel_size, act, bias, bn=False):
        super().__init__()

        layer = []
        layer.append(nn.Conv2d(n_ch, n_ch, kernel_size=kernel_size, padding=kernel_size//2, bias=bias))
        if bn:
            layer.append(nn.BatchNorm2d(n_ch))
        if act == 'ReLU':
            layer.append(nn.ReLU(inplace=True))
        elif act == 'PReLU':
            layer.append(nn.PReLU())
        elif act is not None:
            raise RuntimeError('undefined activation function %s'%act)
        layer.append(nn.Conv2d(n_ch, n_ch, kernel_size=kernel_size, padding=kernel_size//2, bias=bias))
        if bn:
            layer.append(nn.BatchNorm2d(n_ch))

        self.body = nn.Sequential(*layer)

    def forward(self, x):
        return x + self.body(x)


if __name__ == '__main__':
    dnet = CLtoN_D(n_ch_in=3)
    gent = CLtoN_G_indep_dep_3()

    noisy_image_batch = torch.randn(10,3,20,20)

    r = dnet(noisy_image_batch)

    print(r.shape)

    r = gent(noisy_image_batch)

    print(r.shape)

