import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearMaskedConv2d(nn.Conv2d):
    def __init__(self, hor, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(0)
        if hor:
            self.mask[:, :, :, kH//2] = 1
        else:
            self.mask[:, :, kH//2, :] = 1
        self.mask[:, :, kH//2, kH//2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

class EdgeConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, bias=True):
        super().__init__()

        assert (kernel_size+1)%2 == 0, 'kernel size should be odd'

        self.horizontal_linear = nn.Conv2d(in_ch, out_ch, kernel_size=(1, kernel_size), stride=1, padding=(0, kernel_size//2), bias=bias)
        self.vertical_linear = nn.Conv2d(in_ch, out_ch, kernel_size=(kernel_size, 1), stride=1, padding=(kernel_size//2, 0), bias=bias)
        
        self.horizontal_dilated_conv = LinearMaskedConv2d(True, out_ch, out_ch, kernel_size=3, stride=1, padding=kernel_size//2, dilation=kernel_size//2, bias=bias)
        self.vertical_dilated_conv = LinearMaskedConv2d(False, out_ch, out_ch, kernel_size=3, stride=1, padding=kernel_size//2, dilation=kernel_size//2, bias=bias)

    def forward(self, x):
        hor = self.horizontal_linear(x)
        ver = self.vertical_linear(x)

        hor = self.horizontal_dilated_conv(hor)
        ver = self.vertical_dilated_conv(ver)

        return hor + ver

class RBSN(nn.Module):
    def __init__(self, n_in_ch=1, n_out_ch=1, base_ch=64, n_layer=14, near_layer=5, RF_branch=True, near_branch=True):
        super().__init__()

        assert n_layer%2 == 0
        bias = True
        self.n_layer = n_layer

        if RF_branch:
            self.init_conv11 = nn.Conv2d(n_in_ch, base_ch, kernel_size=1, stride=1, padding=0)
            self.backbone = nn.ModuleList([nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=1, padding=1, bias=bias) for _ in range(n_layer)])
            self.edge_convs = nn.ModuleList([EdgeConv2d(base_ch, base_ch, kernel_size=4*l+9) for l in range(n_layer)])
            self.tail_convs = [nn.Conv2d(n_layer*base_ch, base_ch, kernel_size=1, stride=1, padding=0)]
            self.tail_convs = self.tail_convs + [nn.Conv2d(base_ch, base_ch, kernel_size=1, stride=1, padding=0) for _ in range(3)]
            self.tail_convs = self.tail_convs + [nn.Conv2d(base_ch, n_out_ch, kernel_size=1, stride=1, padding=0)]
            self.tail_convs = nn.Sequential(*self.tail_convs)
    
    def forward(self, x):
        org = x

        # full receptive field layers except 3x3 pixels
        results = []
        x = self.init_conv11(x)
        for n_block in range(self.n_layer//2):
            res = x
            x = self.backbone[2*n_block](x)

            results.append(self.edge_convs[2*n_block](x))

            x = F.relu(x)
            x = res + self.backbone[2*n_block+1](x)

            results.append(self.edge_convs[2*n_block+1](x))

        RF_result = torch.cat(results, dim=1)
        RF_result = self.tail_convs(RF_result)

        return RF_result




if __name__ == '__main__':
    # # verification of EdgeConv2d
    # lc = LinearMaskedConv2d(True, in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
    # hl = nn.Conv2d(1, 1, kernel_size=(1, 5), stride=1, padding=(0,2), bias=False)
    # ec = EdgeConv2d(1, 1, kernel_size=5, bias=False)
    # i = torch.zeros((1,1,5,5))
    # i[:,:,1,0] = 1
    # print(i)
    # print(ec(i))

    model = RBSN()
    images = torch.randn((16,1,64,64))
    print(model(images).shape)


