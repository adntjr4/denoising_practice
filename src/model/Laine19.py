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

        self.up_convs = nn.ModuleList([nn.Sequential(*[shifted_conv(base_ch*2, base_ch*2, k_size=3), self.act, shifted_conv(base_ch*2, base_ch*2, k_size=3)])])
        self.up_convs.extend([nn.Sequential(*[shifted_conv(base_ch*3, base_ch*2, k_size=3), self.act, shifted_conv(base_ch*2, base_ch*2, k_size=3)]) for i in range(self.n_depth-2)])
        self.up_convs.extend([nn.Sequential(*[shifted_conv(base_ch*2+in_ch, base_ch*2, k_size=3), self.act, shifted_conv(base_ch*2, base_ch*2, k_size=3)])])

    def forward(self, x):
        skips = [x]
        x = self.act(self.head_conv(x))

        # down
        for l in range(self.n_depth):
            x = self.down_convs[l](x)
            x = self.maxpools[l](x)
            skips.append(x)
            
        # middle
        x = self.middle(x)
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
            x[idx] = rot_hflip_img(single_img, -idx)
        x = torch.cat(x, dim=1)

        # remvoe padded area
        x = x[:,:,:h,:w]

        # tail (1x1 convs)
        x = self.tail(x)

        return x


if __name__ == '__main__':
    net = Laine19()
    # net = shifted_conv(3, 3, 3)
    # print(net)
    batch = torch.randn(16,3,48,48)
    print(net(batch).shape)
