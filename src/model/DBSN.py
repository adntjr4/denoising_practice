import torch
import torch.nn as nn
import torch.nn.functional as F

class DBSN(nn.Module):
    def __init__(self, num_module=5, in_ch=1, out_ch=1, base_ch=64):
        super().__init__()

        assert base_ch%2 == 0, "base channel should be divided with 2"

        self.head_conv11 = nn.Conv2d(in_ch, base_ch, kernel_size=1)

        self.central_conv33 = CentralMaskedConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.central_conv55 = CentralMaskedConv2d(64, 64, kernel_size=5, stride=1, padding=2)

        self.mdc_branch1 = nn.Sequential(*[MDC(stride=2, in_ch=base_ch) for _ in range(num_module)])
        self.mdc_branch2 = nn.Sequential(*[MDC(stride=3, in_ch=base_ch) for _ in range(num_module)])

        t = []
        t.append(nn.Conv2d(base_ch*2, base_ch//2, kernel_size=1))
        t.append(nn.ReLU(inplace=True))
        for i in range(2):
            t.append(nn.Conv2d(base_ch//2, base_ch//2, kernel_size=1))
            t.append(nn.ReLU(inplace=True))
        t.append(nn.Conv2d(base_ch//2, out_ch, kernel_size=1))
        self.tail = nn.Sequential(*t)

    def forward(self, x):
        x = self.head_conv11(x)

        br1 = self.central_conv33(x)
        br2 = self.central_conv55(x)

        br1 = self.mdc_branch1(br1)
        br2 = self.mdc_branch2(br2)

        x = torch.cat([br1, br2], dim=1)

        return self.tail(x)

class MDC(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()

        self.br1_conv11 = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.br2_conv11 = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.br3_conv11 = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.tail_conv11 = nn.Conv2d(3*in_ch, in_ch, kernel_size=1)

        self.br1_conv33_1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride)
        self.br1_conv33_2 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride)
        self.br2_conv33   = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride)

    def forward(self, x):
        residual = x

        branch1 = F.relu(self.br1_conv11(x), inplace=True)
        branch1 = F.relu(self.br1_conv33_1(branch1))
        branch1 = F.relu(self.br1_conv33_2(branch1))

        branch2 = F.relu(self.br2_conv11(x), inplace=True)
        branch2 = F.relu(self.br2_conv33(branch2))

        branch3 = F.relu(self.br3_conv11(x), inplace=True)

        x = torch.cat([branch1, branch2, branch3], dim=1)

        x = F.relu(self.tail_conv11(x))

        return residual + x


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

class C_DBSN(DBSN):
    def __init__(self):
        super().__init__(in_ch=3, out_ch=3)



class DBSN_Likelihood(DBSN):
    def __init__(self):
        super().__init__(in_ch=1, out_ch=2)

class MDC_Cond(nn.Module):
    def __init__(self, stride, in_ch, c_ch, condition_ch):
        super().__init__()

        self.condition_ch = condition_ch

        self.br1_conv11 = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.br2_conv11 = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.br3_conv11 = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.tail_conv11 = nn.Conv2d(3*in_ch, in_ch, kernel_size=1)

        self.br1_conv33_1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride)
        self.br1_conv33_2 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride)
        self.br2_conv33   = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride)

        self.condition_conv  = nn.Conv2d(c_ch, 2*condition_ch, kernel_size=1)
        self.condition_conv1 = nn.Conv2d(condition_ch, condition_ch, kernel_size=1)
        self.condition_conv2 = nn.Conv2d(condition_ch, condition_ch, kernel_size=1)

    def forward(self, x, y):
        residual = x

        branch1 = F.relu(self.br1_conv11(x), inplace=True)
        branch1 = F.relu(self.br1_conv33_1(branch1))
        branch1 = F.relu(self.br1_conv33_2(branch1))

        branch2 = F.relu(self.br2_conv11(x), inplace=True)
        branch2 = F.relu(self.br2_conv33(branch2))

        branch3 = F.relu(self.br3_conv11(x), inplace=True)

        x = torch.cat([branch1, branch2, branch3], dim=1)
        x = F.relu(self.tail_conv11(x))

        y = F.relu(self.condition_conv(y))
        w = self.condition_conv1(y[:, :self.condition_ch, :,:])
        b = self.condition_conv1(y[:, self.condition_ch:, :,:])

        x = w*x + b

        return residual + x

class DBSN_Cond(nn.Module):
    def __init__(self, num_module=5, in_ch=1, out_ch=1, base_ch=64):
        super().__init__()

        assert base_ch%2 == 0, "base channel should be divided with 2"

        self.head_conv11 = nn.Conv2d(in_ch, base_ch, kernel_size=1)

        self.central_conv33 = CentralMaskedConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.central_conv55 = CentralMaskedConv2d(64, 64, kernel_size=5, stride=1, padding=2)

        self.mdc_branch1 = nn.ModuleList([MDC_Cond(stride=2, in_ch=base_ch, c_ch=in_ch, condition_ch=base_ch) for _ in range(num_module)])
        self.mdc_branch2 = nn.ModuleList([MDC_Cond(stride=3, in_ch=base_ch, c_ch=in_ch, condition_ch=base_ch) for _ in range(num_module)])

        t = []
        t.append(nn.Conv2d(base_ch*2, base_ch//2, kernel_size=1))
        t.append(nn.ReLU(inplace=True))
        for i in range(2):
            t.append(nn.Conv2d(base_ch//2, base_ch//2, kernel_size=1))
            t.append(nn.ReLU(inplace=True))
        t.append(nn.Conv2d(base_ch//2, out_ch, kernel_size=1))
        self.tail = nn.Sequential(*t)

    def forward(self, x, nlf=None):
        org = x
        x = self.head_conv11(x)

        br1 = self.central_conv33(x)
        br2 = self.central_conv55(x)

        for br1_module in self.mdc_branch1:
            x_noiser = org + torch.normal(mean=0., std=nlf, size=org.shape).cuda() if nlf is not None else org
            br1 = br1_module(br1, x_noiser.detach())

        for br2_module in self.mdc_branch2:
            x_noiser = org + torch.normal(mean=0., std=nlf, size=org.shape).cuda() if nlf is not None else org
            br2 = br2_module(br2, x_noiser.detach())

        x = torch.cat([br1, br2], dim=1)

        return self.tail(x)

if __name__ == "__main__":
    t = torch.randn(16,1,64,64)

    model = DBSN()

    print(model(t).shape)
    print(sum(p.numel() for p in model.parameters()))
    