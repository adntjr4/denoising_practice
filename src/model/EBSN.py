import torch
import torch.nn as nn
import torch.nn.functional as F

from .DnCNN import DnCNN, Block


class EBSN(nn.Module):
    def __init__(self, n_in_ch=1, n_out_ch=1, n_ch=64, n_layer=20):
        super().__init__()

        self.n_layer = n_layer
        assert n_layer%2 == 0
        bias = True

        self.init_conv11 = nn.Conv2d(in_channels=n_in_ch, out_channels=n_ch, kernel_size=1, stride=1, padding=0)

        self.body_conv33 = nn.ModuleList([nn.Conv2d(n_ch, n_ch, kernel_size=3, stride=1, padding=1, bias=bias) for _ in range(n_layer)])

        self.blind_spot_conv33 = nn.ModuleList([CentralMaskedConv2d(in_channels=n_ch, out_channels=n_ch, kernel_size=3, stride=1, padding=l+1, dilation=l+1) for l in range(n_layer+1)])

        concat_ch = (n_layer+1)*n_ch
        self.tail_conv11_0 = Block(concat_ch//1, concat_ch//2, kernel_size=1, bn=True, act='ReLU', bias=bias)
        self.tail_conv11_1 = Block(concat_ch//2, concat_ch//4, kernel_size=1, bn=True, act='ReLU', bias=bias)
        self.tail_conv11_2 = Block(concat_ch//4, concat_ch//8, kernel_size=1, bn=True, act='ReLU', bias=bias)
        self.tail_conv11_3 = Block(concat_ch//8, n_out_ch    , kernel_size=1, bn=False, act= None , bias=bias)

    def forward(self, x):
        x = self.init_conv11(x)

        concat_tensor = []
        for layer_idx in range(self.n_layer//2):
            residual = x
            concat_tensor.append(self.blind_spot_conv33[2*layer_idx+0](x))
            x = self.body_conv33[layer_idx](x)
            concat_tensor.append(self.blind_spot_conv33[2*layer_idx+1](x))
            x = F.relu(x)
            x = self.body_conv33[layer_idx](x)
            x = residual + x
        concat_tensor.append(self.blind_spot_conv33[self.n_layer](x))

        x = torch.cat(concat_tensor, dim=1)

        x = self.tail_conv11_0(x)
        x = self.tail_conv11_1(x)
        x = self.tail_conv11_2(x)
        x = self.tail_conv11_3(x)
        
        return x

class EBSN_Wide(nn.Module):
    def __init__(self, n_in_ch=1, n_out_ch=1, n_ch=64, n_layer=20):
        super().__init__()

        self.n_layer = n_layer
        assert n_layer%2 == 0
        bias = True

        self.init_conv11 = nn.Conv2d(in_channels=n_in_ch, out_channels=n_ch, kernel_size=1, stride=1, padding=0)

        self.body_conv33 = nn.ModuleList([nn.Conv2d(n_ch, n_ch, kernel_size=3, stride=1, padding=1, bias=bias) for _ in range(n_layer)])

        self.blind_spot_conv33 = nn.ModuleList([CentralMaskedConv2d(in_channels=n_ch, out_channels=n_ch, kernel_size=3, stride=1, padding=l+2, dilation=l+2) for l in range(n_layer+1)])

        concat_ch = (n_layer+1)*n_ch
        self.tail_conv11_0 = Block(concat_ch//1, concat_ch//2, kernel_size=1, bn=True, act='ReLU', bias=bias)
        self.tail_conv11_1 = Block(concat_ch//2, concat_ch//4, kernel_size=1, bn=True, act='ReLU', bias=bias)
        self.tail_conv11_2 = Block(concat_ch//4, concat_ch//8, kernel_size=1, bn=True, act='ReLU', bias=bias)
        self.tail_conv11_3 = Block(concat_ch//8, n_out_ch    , kernel_size=1, bn=False, act= None , bias=bias)

    def forward(self, x):
        x = self.init_conv11(x)

        concat_tensor = []
        for layer_idx in range(self.n_layer//2):
            residual = x
            concat_tensor.append(self.blind_spot_conv33[2*layer_idx+0](x))
            x = self.body_conv33[layer_idx](x)
            concat_tensor.append(self.blind_spot_conv33[2*layer_idx+1](x))
            x = F.relu(x)
            x = self.body_conv33[layer_idx](x)
            x = residual + x
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

class RoundMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(0)
        self.mask[:, :, 0   , :   ] = 1
        self.mask[:, :, kH-1, :   ] = 1
        self.mask[:, :, :   , 0   ] = 1
        self.mask[:, :, :   , kW-1] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

class C_EBSN(EBSN):
    def __init__(self):
        super().__init__(n_in_ch=3, n_out_ch=3)

class C_EBSN_Wide(EBSN_Wide):
    def __init__(self):
        super().__init__(n_in_ch=3, n_out_ch=3)

if __name__ == "__main__":
    model = EBSN()

    t = torch.randn((16,1,64,64))

    print(model(t).shape)
    print(sum(p.numel() for p in model.parameters()))


