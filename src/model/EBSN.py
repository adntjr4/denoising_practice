import torch
import torch.nn as nn

from .DnCNN import DnCNN, Block


class EBSN(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=64, n_layer=20):
        super().__init__()

        self.n_layer = n_layer
        bias = True

        self.init_conv11 = nn.Conv2d(in_channels=in_ch, out_channels=base_ch, kernel_size=1, stride=1, padding=0)

        self.body_conv33 = nn.ModuleList([Block(base_ch, base_ch, kernel_size=3, bn=True, act='ReLU', bias=bias) for _ in range(n_layer)])

        self.blind_spot_conv33 = nn.ModuleList([CentralMaskedConv2d(in_channels=base_ch, out_channels=base_ch, kernel_size=3, stride=1, padding=l+1, dilation=l+1) for l in range(n_layer+1)])

        concat_ch = (n_layer+1)*base_ch
        self.tail_conv11_0 = Block(concat_ch//1, concat_ch//2, kernel_size=1, bn=True, act='ReLU', bias=bias)
        self.tail_conv11_1 = Block(concat_ch//2, concat_ch//4, kernel_size=1, bn=True, act='ReLU', bias=bias)
        self.tail_conv11_2 = Block(concat_ch//4, concat_ch//8, kernel_size=1, bn=True, act='ReLU', bias=bias)
        self.tail_conv11_3 = Block(concat_ch//8, out_ch    , kernel_size=1, bn=False, act= None , bias=bias)

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

if __name__ == "__main__":
    model = EBSN()

    t = torch.randn((16,1,64,64))

    print(model(t).shape)