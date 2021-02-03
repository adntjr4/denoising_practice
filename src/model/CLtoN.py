import torch
import torch.nn as nn
import torch.distributions as torch_distb


class ResBlock(nn.Module):
    def __init__(self, n_ch, kernel_size, act, bias):
        super().__init__()

        layer = []
        layer.append(nn.Conv2d(n_ch, n_ch, kernel_size=kernel_size, padding=kernel_size//2, bias=bias))
        if act == 'ReLU':
            layer.append(nn.ReLU(inplace=True))
        elif act == 'PReLU':
            layer.append(nn.PReLU())
        elif act is not None:
            raise RuntimeError('undefined activation function %s'%act)
        layer.append(nn.Conv2d(n_ch, n_ch, kernel_size=kernel_size, padding=kernel_size//2, bias=bias))

        self.body = nn.Sequential(*layer)

    def forward(self, x):
        return x + self.body(x)


class CLtoN_D(nn.Module):
    def __init__(self, n_ch_in=3, n_ch=64, n_block=6):
        super().__init__()

        self.n_ch_in = n_ch_in
        self.n_ch = n_ch
        self.n_block = n_block

        self.head = nn.Sequential(nn.Conv2d(self.n_ch_in, self.n_ch, kernel_size=3, padding=1, bias=True),
                                  nn.PReLU())

        layers = [ResBlock(n_ch=self.n_ch, kernel_size=3, act='PReLU', bias=True) for _ in range(self.n_block)]
        self.body = nn.Sequential(*layers)

        self.tail = nn.Conv2d(self.n_ch, 1, kernel_size=3, padding=1, bias=True)
    
    def forward(self, img_Gout):
        x = self.head(img_Gout)
        x = self.body(x)
        x = self.tail(x)

        # if GAN_type == 'DCGAN':
        #     y = torch.sigmoid(y)
        
        return x

class CLtoN_G(nn.Module):
    def __init__(self, n_ch_in=3, n_ch_out=3, n_ch=64, n_rand=32, n_ext_block=5, n_indep_block=3, n_dep_block=2):
        super().__init__()

        self.n_ch_in = n_ch_in
        self.n_ch_out = n_ch_out
        self.n_ch = n_ch
        self.n_rand = n_rand

        self.n_ext_block = n_ext_block
        self.n_indep_block = n_indep_block
        self.n_dep_block = n_dep_block

        # feat extractor
        feat_ext_layer = []
        feat_ext_layer.append(nn.Conv2d(n_ch_in+n_rand, self.n_ch*2, kernel_size=3, padding=1, bias=True))
        feat_ext_layer.append(nn.PReLU())
        feat_ext_layer += [ResBlock(n_ch=self.n_ch*2, kernel_size=3, act='PReLU', bias=True) for _ in range(self.n_ext_block)]
        self.feat_ext = nn.Sequential(*feat_ext_layer)

        # pipe-indep
        self.pipe_indep_1 = nn.Sequential(*[ResBlock(n_ch, kernel_size=1, act='PReLU', bias=True) for _ in range(n_indep_block)])
        self.pipe_indep_3 = nn.Sequential(*[ResBlock(n_ch, kernel_size=3, act='PReLU', bias=True) for _ in range(n_indep_block)])

        # pipe-dep
        self.pipe_dep_1 = nn.Sequential(*[ResBlock(n_ch, kernel_size=1, act='PReLU', bias=True) for i in range(n_dep_block)])
        self.pipe_dep_3 = nn.Sequential(*[ResBlock(n_ch, kernel_size=3, act='PReLU', bias=True) for i in range(n_dep_block)])

        # T tail
        self.T_tail = nn.Conv2d(n_ch, n_ch_out, kernel_size=1, padding=0, bias=True)

    def forward(self, img_CL, rand_vec=None):
        (N, C, H, W) = img_CL.size()
        model_on_cuda = next(self.parameters()).is_cuda

        # random vector (is None sample it)
        if rand_vec is None:
            rand_vec = torch.randn((N, self.n_rand))

        # random vector repeat (to same device)
        rand_vec_map = rand_vec.unsqueeze(-1).unsqueeze(-1).repeat(1,1,H,W)
        if model_on_cuda:
            rand_vec_map = rand_vec_map.cuda()

        # feat extractor
        list_cat = [img_CL, rand_vec_map]
        feat_CL = self.feat_ext(torch.cat(list_cat, 1))

        # make initial dep noise feature
        feat_noise_dep = torch_distb.Normal(loc=feat_CL[:,:self.n_ch,:,:], scale=feat_CL[:,self.n_ch:,:,:]).rsample()
        if model_on_cuda:
            feat_noise_dep = feat_noise_dep.cuda()

        # pipe-dep
        feat_noise_dep = self.pipe_dep_1(feat_noise_dep) + \
                         self.pipe_dep_3(feat_noise_dep)

        # make initial indep noise feature
        # feat_noise_indep = torch.rand_like(feat_noise_dep, requires_grad=True)
        feat_noise_indep = torch.randn_like(feat_noise_dep)

        # pipe-indep
        feat_noise_indep = self.pipe_indep_1(feat_noise_indep) + \
                           self.pipe_indep_3(feat_noise_indep)

        # pipe-merge
        noise = self.T_tail(feat_noise_indep + feat_noise_dep)

        return noise


if __name__ == '__main__':
    dnet = CLtoN_D(n_ch_in=3)
    gent = CLtoN_G(n_ch_in=3, n_ch_out=3)

    noisy_image_batch = torch.randn(10,3,20,20)

    r = dnet(noisy_image_batch)

    print(r.shape)

    r = gent(noisy_image_batch)

    print(r.shape)

