import torch
import torch.nn as nn


class FBSNet(nn.Module):
    def __init__(self, n_in_ch=1, n_out_ch=1, n_ch=64, n_layer=20):
        super().__init__()
        assert n_ch%2 == 0

        self.n_layer = n_layer
        bias = True

        self.init_conv11 = nn.Conv2d(in_channels=n_in_ch, out_channels=n_ch, kernel_size=1, stride=1, padding=0)

        self.body_conv33_0 = nn.ModuleList
        self.body_conv33_1 = nn.ModuleList([Block(n_ch, n_ch, kernel_size=3, bn=True, act='ReLU', bias=bias) for _ in range(n_layer)])

        self.blind_spot_conv33 = nn.ModuleList([CentralMaskedConv2d(in_channels=n_ch, out_channels=n_ch, kernel_size=3, stride=1, padding=l+1, dilation=l+1) for l in range(n_layer+1)])

        concat_ch = (n_layer+1)*n_ch
        self.tail_conv11_0 = Block(concat_ch//1, concat_ch//2, kernel_size=1, bn=True, act='ReLU', bias=bias)
        self.tail_conv11_1 = Block(concat_ch//2, concat_ch//4, kernel_size=1, bn=True, act='ReLU', bias=bias)
        self.tail_conv11_2 = Block(concat_ch//4, concat_ch//8, kernel_size=1, bn=True, act='ReLU', bias=bias)
        self.tail_conv11_3 = Block(concat_ch//8, n_out_ch    , kernel_size=1, bn=False, act= None , bias=bias)

    def forward(self, x):
        x = self.init_conv11(x)

        concat_tensor = []
        for layer_idx in range(self.n_layer):
            concat_tensor.append(self.blind_spot_conv33[layer_idx](x))
            x = self.body_conv33[layer_idx](x)
        concat_tensor.append(self.blind_spot_conv33[self.n_layer](x))

        x = torch.cat(concat_tensor, dim=1)

        x = self.tail_conv11_0(x)
        x = self.tail_conv11_1(x)
        x = self.tail_conv11_2(x)
        x = self.tail_conv11_3(x)
        
        return x