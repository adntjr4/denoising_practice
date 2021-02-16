import torch
import torch.nn as nn
import torch.nn.functional as F


class RBSN(nn.Module):
    '''
    RBSN 
    mark 1: which have RF w/o 3x3. failed to converge.
    '''
    def __init__(self, n_in_ch=1, n_out_ch=1, base_ch=64, n_layer=8, tail_layer=5):
        super().__init__()

        assert n_layer%2 == 0
        bias = True
        self.n_layer = n_layer

        self.init_conv11 = nn.Conv2d(n_in_ch, base_ch, kernel_size=1, stride=1, padding=0)

        self.backbone = nn.ModuleList([nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=1, padding=1, bias=bias) for _ in range(n_layer)])
        self.edge_convs = nn.ModuleList([EdgeConv2d(base_ch, base_ch, kernel_size=4*l+9) for l in range(n_layer)])
        self.masked_conv = CenterMaskedConv2d(in_channels=base_ch, out_channels=base_ch, kernel_size=3, stride=1, padding=1)

        self.tail_convs = [nn.Conv2d((n_layer+1)*base_ch, base_ch, kernel_size=1, stride=1, padding=0)]
        self.tail_convs = self.tail_convs + [nn.Conv2d(base_ch, base_ch, kernel_size=1, stride=1, padding=0) for _ in range(tail_layer-2)]
        self.tail_convs = self.tail_convs + [nn.Conv2d(base_ch, n_out_ch, kernel_size=1, stride=1, padding=0)]
        self.tail_convs = nn.Sequential(*self.tail_convs)

    def forward(self, x):
        x = self.init_conv11(x)
        org = x

        # full receptive field layers except 3x3 pixels
        results = []
        for n_block in range(self.n_layer//2):
            res = x
            x = self.backbone[2*n_block](x)

            results.append(self.edge_convs[2*n_block](x))

            x = F.relu(x)
            x = res + self.backbone[2*n_block+1](x)

            results.append(self.edge_convs[2*n_block+1](x))

        # near receptive field layer
        x = self.masked_conv(org)
        x = F.relu(x)

        results.append(x)
        result = torch.cat(results, dim=1)
        result = self.tail_convs(result)

        return result

class Conditioned_RBSN(nn.Module):
    def __init__(self, n_in_ch=1, n_out_ch=1, base_ch=64, n_block=4):
        super().__init__()
        self.n_block = n_block

        self.init_conv11 = nn.Conv2d(n_in_ch, base_ch, kernel_size=1, stride=1, padding=0)
        self.blocks = nn.ModuleList([ConditionedResBlock(base_ch, base_ch, n_in_ch) for _ in range(n_block)])
        self.edge_convs = nn.ModuleList([EdgeConv2d(base_ch, base_ch, kernel_size=8*l+13) for l in range(n_block)])

        self.tail_convs = [nn.Conv2d(n_block*base_ch, base_ch, kernel_size=1, stride=1, padding=0)]
        self.tail_convs = self.tail_convs + [nn.Conv2d(base_ch, base_ch, kernel_size=1, stride=1, padding=0) for _ in range(2)]
        self.tail_convs = self.tail_convs + [nn.Conv2d(base_ch, n_out_ch, kernel_size=1, stride=1, padding=0)]
        self.tail_convs = nn.Sequential(*self.tail_convs)

    def forward(self, x):
        org_img = x
        x = self.init_conv11(x)

        results = []
        for block_idx in range(self.n_block):
            x = self.blocks[block_idx](x, org_img)
            results.append(self.edge_convs[block_idx](x))
        
        results = torch.cat(results, dim=1)

        return self.tail_convs(results)



class ConditionedResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, img_ch):
        super().__init__()

        self.in_ch = in_ch

        # conditional branch
        self.cond_conv = CenterMaskedConv2d(in_channels=img_ch, out_channels=2*in_ch, kernel_size=3, padding=1)
        self.gamma_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1)
        self.beta_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1)

        # main branch
        self.conv1 = nn.Conv2d(in_channels=in_ch,  out_channels=out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1)

    def forward(self, x, img):
        # conditional branch
        c = F.relu(self.cond_conv(img), inplace=True)
        gamma, beta = torch.split(c, self.in_ch, dim=1)
        gamma = self.gamma_conv(gamma)
        beta = self.beta_conv(beta)

        # main branch
        res = x
        x = x*gamma + beta
        x = self.conv1(x)
        x = self.conv2(F.relu(x))

        return res + x


class CenterMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH//2, kH//2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

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



if __name__ == '__main__':
    # # verification of EdgeConv2d
    # lc = LinearMaskedConv2d(True, in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
    # hl = nn.Conv2d(1, 1, kernel_size=(1, 5), stride=1, padding=(0,2), bias=False)
    # ec = EdgeConv2d(1, 1, kernel_size=5, bias=False)
    # i = torch.zeros((1,1,5,5))
    # i[:,:,1,0] = 1
    # print(i)
    # print(ec(i))

    model = Conditioned_RBSN()
    images = torch.randn((16,1,64,64))
    print(model(images).shape)


