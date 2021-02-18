import torch
import torch.nn as nn
import torch.nn.functional as F



class RBSN_sep(nn.Module):
    '''
    RBSN 
    mark 1: which have RF w/o 3x3. failed to converge.
    mark 2: include center masked 3x3. cannot denoise image.
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

        self.tail_convs = [nn.Conv2d((n_layer+1)*base_ch, base_ch, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True)]
        for _ in range(tail_layer-2):
            self.tail_convs.append(nn.Conv2d(base_ch, base_ch, kernel_size=1, stride=1, padding=0))
            self.tail_convs.append(nn.ReLU(inplace=True))
        self.tail_convs.append(nn.Conv2d(base_ch, n_out_ch, kernel_size=1, stride=1, padding=0))
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

class C_RBSN_sep(RBSN_sep):
    def __init__(self):
        super().__init__(n_in_ch=3, n_out_ch=3)

class RBSN_cond(nn.Module):
    def __init__(self, n_in_ch=1, n_out_ch=1, base_ch=64, n_block=4):
        super().__init__()
        self.n_block = n_block

        self.init_conv11 = nn.Conv2d(n_in_ch, base_ch, kernel_size=1, stride=1, padding=0)
        self.blocks = nn.ModuleList([ConditionedResBlock(base_ch, base_ch, n_in_ch) for _ in range(n_block)])
        self.edge_convs = nn.ModuleList([EdgeConv2d(base_ch, base_ch, kernel_size=8*l+13) for l in range(n_block)])
        #self.edge_convs = nn.ModuleList([CenterMaskedConv2d(base_ch, base_ch, kernel_size=3, stride=1, padding=4*l+6, dilation=4*l+6) for l in range(n_block)])

        self.tail_convs = [nn.Conv2d(n_block*base_ch, base_ch, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True)]
        for _ in range(2):
            self.tail_convs.append(nn.Conv2d(base_ch, base_ch, kernel_size=1, stride=1, padding=0))
            self.tail_convs.append(nn.ReLU(inplace=True))
        self.tail_convs.append(nn.Conv2d(base_ch, n_out_ch, kernel_size=1, stride=1, padding=0))
        self.tail_convs = nn.Sequential(*self.tail_convs)

    def forward(self, x):
        org_img = x
        x = self.init_conv11(x)

        results = []
        for block_idx in range(self.n_block):
            x = self.blocks[block_idx](x, org_img)
            results.append(F.relu(self.edge_convs[block_idx](x), inplace=True))
        
        results = torch.cat(results, dim=1)

        return self.tail_convs(results)

class RBSN_FS(nn.Module):
    '''
    RBSN filter sharing.
    '''
    def __init__(self, in_ch=3, out_ch=3, base_ch=64, n_block=4):
        super().__init__()
        self.n_block = n_block

        self.init_conv = nn.Conv2d(in_ch, base_ch, kernel_size=1)

        self.shared_center_masked_conv = CenterMaskedConv2d(base_ch, base_ch, kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([RBSN_FS_Block(base_ch=base_ch, block_idx=idx, center_hole=1) for idx in range(n_block)])

        cat_ch = 2*n_block*base_ch
        self.gather_tail = nn.Conv2d(cat_ch, base_ch, kernel_size=1)
        self.tail = []
        for _ in range(3):
            self.tail += [nn.ReLU(inplace=True), nn.Conv2d(base_ch, base_ch, kernel_size=1)]
        self.tail += [nn.ReLU(inplace=True), nn.Conv2d(base_ch, out_ch, kernel_size=1)]
        self.tail = nn.Sequential(*self.tail)

    def forward(self, x):
        x = self.init_conv(x)

        feat = self.shared_center_masked_conv(x)

        results = []
        for block_idx in range(self.n_block):
            x, r1, r2 = self.blocks[block_idx](x, feat)
            results += [r1, r2]
        
        x = self.gather_tail(torch.cat(results, dim=1))
        x = self.tail(x)

        return x

# region - EBSN_Edge
# class EBSN_Edge(nn.Module):
#     def __init__(self, n_in_ch=1, n_out_ch=1, n_ch=64, n_layer=20):
#         super().__init__()

#         self.n_layer = n_layer
#         assert n_layer%2 == 0
#         bias = True

#         self.init_conv11 = nn.Conv2d(in_channels=n_in_ch, out_channels=n_ch, kernel_size=1, stride=1, padding=0)

#         self.body_conv33 = nn.ModuleList([nn.Conv2d(n_ch, n_ch, kernel_size=3, stride=1, padding=1, bias=bias) for _ in range(n_layer)])

#         self.blind_spot_conv33 = nn.ModuleList([EdgeConv2d(n_ch, n_ch, kernel_size=2*l+3) for l in range(n_layer+1)])

#         concat_ch = (n_layer+1)*n_ch
#         self.tail_conv11_0 = Block(concat_ch//1, concat_ch//2, kernel_size=1, bn=True, act='ReLU', bias=bias)
#         self.tail_conv11_1 = Block(concat_ch//2, concat_ch//4, kernel_size=1, bn=True, act='ReLU', bias=bias)
#         self.tail_conv11_2 = Block(concat_ch//4, concat_ch//8, kernel_size=1, bn=True, act='ReLU', bias=bias)
#         self.tail_conv11_3 = Block(concat_ch//8, n_out_ch    , kernel_size=1, bn=False, act= None , bias=bias)

#     def forward(self, x):
#         x = self.init_conv11(x)

#         concat_tensor = []
#         for layer_idx in range(self.n_layer//2):
#             residual = x
#             concat_tensor.append(self.blind_spot_conv33[2*layer_idx+0](x))
#             x = self.body_conv33[layer_idx](x)
#             concat_tensor.append(self.blind_spot_conv33[2*layer_idx+1](x))
#             x = F.relu(x)
#             x = self.body_conv33[layer_idx](x)
#             x = residual + x
#         concat_tensor.append(self.blind_spot_conv33[self.n_layer](x))

#         x = torch.cat(concat_tensor, dim=1)

#         x = self.tail_conv11_0(x)
#         x = self.tail_conv11_1(x)
#         x = self.tail_conv11_2(x)
#         x = self.tail_conv11_3(x)
        
#         return x
# endregion

class RBSN_FS_Block(nn.Module):
    def __init__(self, base_ch, block_idx, center_hole=1):
        super().__init__()
        
        self.base_ch = base_ch

        # for conditional branch
        self.feat_conv11 = nn.Conv2d(base_ch, 2*base_ch, kernel_size=1)
        self.feat_r_conv11 = nn.Conv2d(base_ch, base_ch, kernel_size=1)
        self.feat_b_conv11 = nn.Conv2d(base_ch, base_ch, kernel_size=1)

        # for main branch
        self.conv33_1 = nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1)
        self.conv33_2 = nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1)

        # for extract result
        ch = center_hole
        ex_kernel_size = [4*block_idx + 2*ch + 3, 4*block_idx + 2*ch + 5]

        self.edge_conv_1 = EdgeConv2d(base_ch, base_ch, kernel_size=ex_kernel_size[0])
        self.edge_conv_2 = EdgeConv2d(base_ch, base_ch, kernel_size=ex_kernel_size[1])

    def forward(self, x, feat):
        # conditional branch
        f = F.relu(self.feat_conv11(feat.detach()), inplace=True)
        r, b = torch.split(f, self.base_ch, dim=1)
        r = self.feat_r_conv11(r)
        b = self.feat_b_conv11(b)

        # main branch
        keep_x = x
        x = x*r + b

        result1 = self.edge_conv_1(x)

        x = F.relu(self.conv33_1(x), inplace=True)
        
        result2 = self.edge_conv_2(x)
        
        x = self.conv33_2(x)
        x = keep_x + x

        return x, result1, result2

class RBSNBlock(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()
        self.conv33 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride)

    def forward(self, x):
        residual = x
        x = F.relu(self.conv33(x), inplace=True)
        return residual + x

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
        self.kernel_size = kernel_size

        self.horizontal_up_linear  = nn.Conv2d(in_ch, out_ch, kernel_size=(1, kernel_size), stride=1, padding=kernel_size//2, bias=bias)
        self.horizontal_dw_linear  = nn.Conv2d(in_ch, out_ch, kernel_size=(1, kernel_size), stride=1, padding=kernel_size//2, bias=bias)
        self.vertical_left_linear  = nn.Conv2d(in_ch, out_ch, kernel_size=(kernel_size, 1), stride=1, padding=kernel_size//2, bias=bias)
        self.vertical_right_linear = nn.Conv2d(in_ch, out_ch, kernel_size=(kernel_size, 1), stride=1, padding=kernel_size//2, bias=bias)

    def forward(self, x):
        hor_up = self.horizontal_up_linear(x)
        hor_dw = self.horizontal_dw_linear(x)

        ver_left  = self.vertical_left_linear(x)
        ver_right = self.vertical_right_linear(x)

        hor_up = hor_up[:,:,:-2*(self.kernel_size//2),:]
        hor_dw = hor_dw[:,:,2*(self.kernel_size//2):,:]

        ver_left  = ver_left[:,:,:,:-2*(self.kernel_size//2)]
        ver_right = ver_right[:,:,:,2*(self.kernel_size//2):]

        return hor_up + hor_dw + ver_left + ver_right



if __name__ == '__main__':
    # # verification of EdgeConv2d
    # lc = LinearMaskedConv2d(True, in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
    # hl = nn.Conv2d(1, 1, kernel_size=(1, 5), stride=1, padding=(0,2), bias=False)
    # ec = EdgeConv2d(1, 1, kernel_size=5, bias=False)
    # i = torch.zeros((1,1,5,5))
    # i[:,:,1,0] = 1
    # print(i)
    # print(ec(i))

    model = RBSN_FS()
    images = torch.randn((16,3,64,64))
    print(model(images).shape)


