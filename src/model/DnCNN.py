import torch
import torch.nn as nn

from . import regist_model, get_model_object

'''
modified version of original code
Ref : https://github.com/cszn/KAIR
'''

@regist_model
class DnCNN(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, n_layer=17):
        super().__init__()
        bias = True

        head = Block(in_ch, base_ch, kernel_size=3, bn=True, act='ReLU', bias=bias)
        body = [Block(base_ch, base_ch, kernel_size=3, bn=True, act='ReLU', bias=bias) for _ in range(n_layer-2)]
        tail = Block(base_ch, in_ch, kernel_size=3, bn=False, act=None, bias=bias)

        self.model = nn.Sequential(head, *body, tail)

    def forward(self, x):
        n = self.model(x)
        return x-n

    def _initialize_weights(self):
        # Liyong version
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
                clip_b = 0.025
                w = m.weight.data.shape[0]
                for j in range(w):
                    if m.weight.data[j] >= 0 and m.weight.data[j] < clip_b:
                        m.weight.data[j] = clip_b
                    elif m.weight.data[j] > -clip_b and m.weight.data[j] < 0:
                        m.weight.data[j] = -clip_b
                m.running_var.fill_(0.01)

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, bn, act, bias):
        super().__init__()
        model = []
        model.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size//2, bias=bias))
        if bn: model.append(nn.BatchNorm2d(out_ch, eps=1e-04, momentum=0.9, affine=True))
        if act == 'ReLU':
            model.append(nn.ReLU(inplace=True))
        elif act is not None:
            raise RuntimeError('undefined activation function %s'%act)

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

@regist_model
class DnCNN_B(DnCNN):
    def __init__(self, in_ch=1):
        super().__init__(in_ch=in_ch, n_layer=20)

# ========

@regist_model
class NarrowDnCNN(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=128, n_layer=44):
        super().__init__()
        bias = True

        head = Block(in_ch, base_ch, kernel_size=3, bn=True, act='ReLU', bias=bias)
        hbod = [Block(base_ch, base_ch, kernel_size=1, bn=True, act='ReLU', bias=bias) for _ in range((n_layer-3)//2)]
        midd = Block(base_ch, base_ch, kernel_size=3, bn=True, act='ReLU', bias=bias)
        mbod = [Block(base_ch, base_ch, kernel_size=1, bn=True, act='ReLU', bias=bias) for _ in range((n_layer-3) - (n_layer-3)//2)]
        tail = Block(base_ch, out_ch, kernel_size=3, bn=False, act=None, bias=bias)

        self.model = nn.Sequential(head, *hbod, midd, *mbod, tail)

    def forward(self, x):
        n = self.model(x)
        return x-n

    def _initialize_weights(self):
        # Liyong version
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
                clip_b = 0.025
                w = m.weight.data.shape[0]
                for j in range(w):
                    if m.weight.data[j] >= 0 and m.weight.data[j] < clip_b:
                        m.weight.data[j] = clip_b
                    elif m.weight.data[j] > -clip_b and m.weight.data[j] < 0:
                        m.weight.data[j] = -clip_b
                m.running_var.fill_(0.01)

if __name__ == "__main__":
    dn = NarrowDnCNN()
    print(sum(p.numel() for p in dn.parameters()))
