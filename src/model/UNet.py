import torch
import torch.nn as nn
import torch.nn.functional as F

'''
UNet baseline model for reproducing noise2void.
base block structures are coded by refering "CSBDeep" 
: https://github.com/CSBDeep/CSBDeep/blob/master/csbdeep/internals/blocks.py
'''


class Conv_Block(nn.Module):
    def __init__(self, n_in_ch:int, n_out_ch:int, bn=True, act='ReLU', bias=True):
        super().__init__()

        layers = []
        layers.append(nn.Conv2d(n_in_ch, n_out_ch, kernel_size=3, padding=1, bias=bias))
        if bn: layers.append(nn.BatchNorm2d(n_out_ch, eps=1e-04, momentum=0.9, affine=True))
        if act == 'ReLU':
            layers.append(nn.ReLU(inplace=True))
        elif act is not None:
            raise RuntimeError('Wrong activation function %s'%act)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Conv_Block_Serial(nn.Module):
    def __init__(self, n_blocks:int, n_in_ch:int, n_out_ch:int, bn=True, act='ReLU', bias=True):
        super().__init__()
        blocks = [Conv_Block(n_in_ch, n_out_ch, bn, act, bias)]
        blocks = blocks + [Conv_Block(n_out_ch, n_out_ch, bn, act, bias) for i in range(n_blocks-1)]
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class UNet_Block(nn.Module):
    def __init__(self, n_depth=2, n_in_ch=3, n_base_ch=16, n_conv_per_depth=16, n_pooling=2, bn=True, act='ReLU', bias=True):
        super().__init__()
        self.bias = True
        self.n_depth = n_depth
        self.n_pooling = n_pooling

        # down
        down_list = []
        down_list.append(Conv_Block_Serial(n_conv_per_depth, n_in_ch=n_in_ch, n_out_ch=n_base_ch, bn=bn, act=act, bias=bias))
        for n in range(1,n_depth):
            down_list.append(Conv_Block_Serial(n_conv_per_depth, n_in_ch=n_base_ch * n_pooling**(n-1), n_out_ch=n_base_ch * n_pooling**n, bn=bn, act=act, bias=bias))
        self.down_modules = nn.ModuleList(down_list)

        # middle
        self.middle = Conv_Block_Serial(n_conv_per_depth, n_in_ch=n_base_ch * n_pooling**(n_depth-1), n_out_ch=n_base_ch * n_pooling**n_depth, bn=bn, act=act, bias=bias)

        # up
        upconv_list = []
        up_list = []
        for n in reversed(range(n_depth)):
            upconv_list.append(nn.ConvTranspose2d(n_base_ch * n_pooling**(n+1), n_base_ch * n_pooling**n, kernel_size=n_pooling, stride=n_pooling))
            up_list.append(Conv_Block_Serial(n_conv_per_depth, n_in_ch=n_base_ch * n_pooling**(n+1), n_out_ch=n_base_ch * n_pooling**n, bn=bn, act=act, bias=bias))
        self.upconv_modules = nn.ModuleList(upconv_list)
        self.up_modules = nn.ModuleList(up_list)

    def forward(self, x):
        # down
        skip_x = []
        for idx in range(self.n_depth):
            x = self.down_modules[idx](x)
            skip_x.append(x)
            x = F.max_pool2d(x, self.n_pooling)
        
        # middle
        x = self.middle(x)

        # up
        for idx in range(self.n_depth):
            x = torch.cat([self.upconv_modules[idx](x), skip_x[self.n_depth-idx-1]], dim=1)
            x = self.up_modules[idx](x)

        return x

class N2V_UNet(nn.Module):
    def __init__(self, n_depth=2, n_ch_in=3, n_base_ch=16, n_conv_per_depth=16, n_pooling=2, bn=True):
        super().__init__()
        modules = [UNet_Block(n_depth, n_ch_in, n_base_ch, n_conv_per_depth, n_pooling, bn), nn.Conv2d(n_base_ch, n_ch_in, kernel_size=3, padding=1)]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return x + self.model(x)


if __name__ == "__main__":
    dn = N2V_UNet()
    i = torch.randn(10,3,28,28)
    o = dn(i)
    print(o.shape)