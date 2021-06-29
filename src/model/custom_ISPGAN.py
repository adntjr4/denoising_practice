import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as torch_distb

from . import regist_model, get_model_object

eps = 1e-6

@regist_model
class ISPGAN_Generator(nn.Module):
    def __init__(self, in_ch, out_ch, base_ch=64, r_rand=32, n_pipe1=3, n_pipe2=3, n_pipe3=3):
        super().__init__()

        self.in_ch   = in_ch
        self.out_ch  = out_ch
        self.base_ch = base_ch
        self.r_rand  = r_rand

        self.init_conv = nn.Conv2d(in_ch, 2*base_ch, kernel_size=1)

        bias = True
        self.pipe1 = nn.Sequential(*[ConditionalResBlock(base_ch, r_rand, kernel_size=1, act='ReLU', bias=bias) for i in range(n_pipe1)])
        self.pipe2 = nn.Sequential(*[ConditionalResBlock(base_ch, r_rand, kernel_size=1, act='ReLU', bias=bias) for i in range(n_pipe2)])
        self.pipe3 = nn.Sequential(*[ConditionalResBlock(base_ch, r_rand, kernel_size=3, act='ReLU', bias=bias) for i in range(n_pipe3)])
        self.tail  = nn.Conv2d(base_ch, out_ch,  kernel_size=3, padding=1, bias=bias)

    def forward(self, x, r=None):
        device = x.device
        b,c,h,w = x.shape

        if r is None:
            r = torch.randn((b, self.r_rand), device=device)
        #r_map = r.unsqueeze(-1).unsqueeze(-1).repeat(1,1,h,w)

        x = self.init_conv(x)

        x = torch_distb.Normal(loc=x[:,:self.base_ch,:,:], scale=torch.clip(x[:,self.base_ch:,:,:], eps)).rsample()
        x.to(device)

        x,_ = self.pipe1((x, r))
        x = F.avg_pool2d(x, 2)
        x,_ = self.pipe2((x, r))
        x = F.interpolate(x, size=(h,w), mode='bilinear', align_corners=False)
        x,_ = self.pipe3((x, r))
        return self.tail(x)

class ConditionalResBlock(nn.Module):
    def __init__(self, n_ch, n_rand, kernel_size, act, bias):
        super().__init__()

        self.n_ch = n_ch
        self.conv1 = nn.Conv2d(n_ch, n_ch, kernel_size=kernel_size, padding=kernel_size//2, bias=bias)
        self.conv2 = nn.Conv2d(n_ch, n_ch, kernel_size=kernel_size, padding=kernel_size//2, bias=bias)
        self.mlp1  = MLP(n_rand, 2*n_ch, 3)
        self.mlp2  = MLP(n_rand, 2*n_ch, 3)

        if act == 'ReLU':
            self.act = nn.ReLU(inplace=True)
        elif act == 'LReLU':
            self.act = nn.LeakyReLU(inplace=True)
        else:
            raise RuntimeError('undefined activation function %s'%act)

    def forward(self, xr):
        x, r = xr
        res = x

        p1, p2  = self.mlp1(r), self.mlp2(r)
        r1,b1 = p1[:,:self.n_ch], p1[:,self.n_ch:]
        r2,b2 = p2[:,:self.n_ch], p2[:,self.n_ch:]

        x = self.conv1(x)
        x = F.instance_norm(x)
        x = self.act(x*r1[..., None, None]+b1[..., None, None])

        x = self.conv2(x)
        x = F.instance_norm(x)
        x = x*r2[..., None, None]+b2[..., None, None]

        return res + x, r

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
        elif act == 'LReLU':
            layer.append(nn.LeakyReLU(inplace=True))
        elif act is not None:
            raise RuntimeError('undefined activation function %s'%act)
        layer.append(nn.Conv2d(n_ch, n_ch, kernel_size=kernel_size, padding=kernel_size//2, bias=bias))
        if bn:
            layer.append(nn.BatchNorm2d(n_ch))

        self.body = nn.Sequential(*layer)

    def forward(self, x):
        return x + self.body(x)

class MLP(nn.Module):
    def __init__(self, in_ch, out_ch, n_layer):
        super().__init__()

        layers = [nn.Linear(in_ch, out_ch)]
        for i in range(n_layer-1):
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(out_ch, out_ch))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)

if __name__ == '__main__':
    model = ISPGAN_Generator(3, 3, 64, 32)

    t = torch.randn((16,3,64,64))

    print(model(t).shape)
