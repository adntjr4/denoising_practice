import torch
import torch.nn as nn

'''
modified version of original code
Ref : https://github.com/cszn/KAIR
'''


class DnCNN(nn.Module):
    def __init__(self, n_in_ch=1, n_out_ch=1, n_ch=64, n_layer=17):
        super().__init__()
        bias = True

        head = Block(n_in_ch, n_ch, bn=False, act='ReLU', bias=bias)
        body = [Block(n_ch, n_ch, bn=True, act='ReLU', bias=bias) for _ in range(n_layer-2)]
        tail = Block(n_ch, n_out_ch, bn=False, act=None, bias=bias)

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
    def __init__(self, n_in_ch, n_out_ch, bn, act, bias):
        super().__init__()
        model = []
        model.append(nn.Conv2d(n_in_ch, n_out_ch, kernel_size=3, padding=1, bias=bias))
        if bn: model.append(nn.BatchNorm2d(n_out_ch, eps=1e-04, momentum=0.9, affine=True))
        if act == 'ReLU':
            model.append(nn.ReLU(inplace=True))
        elif act is not None:
            raise RuntimeError('undefined activation function %s'%act)

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class CDnCNN_B(DnCNN):
    def __init__(self):
        super().__init__(n_in_ch=3, n_out_ch=3, n_layer=20)

class DnCNN_B(DnCNN):
    def __init__(self):
        super().__init__(n_in_ch=1, n_out_ch=1, n_layer=20)


if __name__ == "__main__":
    dn = DnCNN(n_in_ch=3, n_out_ch=3)
    i = torch.randn(10,3,28,28)
    o = dn(i)
    print(o.shape)