import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_ch, kernel_size, act, bias, bn=False):
        super().__init__()

        layer = []
        layer.append(nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=kernel_size//2, bias=bias))
        if bn:
            layer.append(nn.BatchNorm2d(in_ch))
        if act == 'ReLU':
            layer.append(nn.ReLU(inplace=True))
        elif act == 'PReLU':
            layer.append(nn.PReLU())
        elif act == 'LReLU':
            layer.append(nn.LeakyReLU(inplace=True))
        elif act is not None:
            raise RuntimeError('undefined activation function %s'%act)
        layer.append(nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=kernel_size//2, bias=bias))
        if bn:
            layer.append(nn.BatchNorm2d(in_ch))

        self.body = nn.Sequential(*layer)

    def forward(self, x):
        return x + self.body(x)

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
