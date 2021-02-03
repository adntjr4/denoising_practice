import torch
import torch.nn as nn


class HeadMaskedConv2d(nn.Conv2d):
    def __init__(self, in_ch, out_ch, bias):
        super().__init__(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=bias)

        self.register_buffer('mask', self.weight.data.clone())
        self.mask.fill_(0)
        self.mask[:, :, 0, 1] = 1
        self.mask[:, :, 2, 1] = 1
        self.mask[:, :, 1, 0] = 1
        self.mask[:, :, 1, 2] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

class CollectMaskedConv2d(nn.Conv2d):
    def __init__(self, in_ch, out_ch, bias):
        super().__init__(in_ch, out_ch, kernel_size=5, stride=1, padding=2, bias=bias)

        self.register_buffer('mask', self.weight.data.clone())
        self.mask.fill_(1)
        self.mask[:, :, 1, 2] = 0
        self.mask[:, :, 3, 2] = 0
        self.mask[:, :, 2, 1] = 0
        self.mask[:, :, 2, 3] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

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

class FBSNet(nn.Module):
    def __init__(self, num_block=5, in_ch=1, base_ch=64):
        super().__init__()

        bias = True

        initial_conv11 = nn.Conv2d(in_ch, base_ch, kernel_size=1)

        head_conv = HeadMaskedConv2d(base_ch, base_ch, bias=bias)
        head_conv11 = [ResBlock(base_ch, kernel_size=1, act='ReLU', bias=bias) for _ in range(num_block)]

        collect_conv = CollectMaskedConv2d(base_ch, base_ch, bias=bias)
        collect_conv11 = [ResBlock(base_ch, kernel_size=1, act='ReLU', bias=bias) for _ in range(num_block)]

        last_conv11 = nn.Conv2d(base_ch, in_ch, kernel_size=1)

        self.model = nn.Sequential(initial_conv11, head_conv, *head_conv11, collect_conv, *collect_conv11, last_conv11)

    def forward(self, x):
        return self.model(x)

# ========================

class HeadMaskedConv2d_R(nn.Conv2d):
    def __init__(self, in_ch, out_ch, bias):
        super().__init__(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=bias)

        self.register_buffer('mask', self.weight.data.clone())
        self.mask.fill_(1)
        self.mask[:, :, 1, 1] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

class CollectMaskedConv2d_R(nn.Conv2d):
    def __init__(self, in_ch, out_ch, bias):
        super().__init__(in_ch, out_ch, kernel_size=5, stride=1, padding=2, bias=bias)

        self.register_buffer('mask', self.weight.data.clone())
        self.mask.fill_(1)
        self.mask[:, :, 1, 1] = 0
        self.mask[:, :, 1, 2] = 0
        self.mask[:, :, 1, 3] = 0
        self.mask[:, :, 2, 1] = 0
        self.mask[:, :, 2, 3] = 0
        self.mask[:, :, 3, 1] = 0
        self.mask[:, :, 3, 2] = 0
        self.mask[:, :, 3, 3] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

class FBSNet_R(nn.Module):
    def __init__(self, num_block=5, in_ch=1, base_ch=64):
        super().__init__()

        bias = True

        initial_conv11 = nn.Conv2d(in_ch, base_ch, kernel_size=1)

        head_conv = HeadMaskedConv2d_R(base_ch, base_ch, bias=bias)
        head_conv11 = [ResBlock(base_ch, kernel_size=1, act='ReLU', bias=bias) for _ in range(num_block)]

        collect_conv = CollectMaskedConv2d_R(base_ch, base_ch, bias=bias)
        collect_conv11 = [ResBlock(base_ch, kernel_size=1, act='ReLU', bias=bias) for _ in range(num_block)]

        last_conv11 = nn.Conv2d(base_ch, in_ch, kernel_size=1)

        self.model = nn.Sequential(initial_conv11, head_conv, *head_conv11, collect_conv, *collect_conv11, last_conv11)

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    fbsnet = FBSNet()
    i = torch.randn(16,1,64,64)
    o = fbsnet(i)
    print(o.shape)
    print(sum(p.numel() for p in fbsnet.parameters()))
