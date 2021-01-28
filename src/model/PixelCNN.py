import torch
import torch.nn as nn


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert mask_type in {'A', 'B'}

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()

        if mask_type == 'A':
            assert self.kernel_size[0] == 3
            self.mask.fill_(0)
            self.mask[:, :, kH // 2, :      ] = 1
            self.mask[:, :, :      , kW // 2] = 1
            self.mask[:, :, kH // 2, kW // 2] = 0
        else:
            assert self.kernel_size[0] == 5
            self.mask.fill_(1)
            self.mask[:, :, kH//2+1, kH//2+1] = 0
            self.mask[:, :, kH//2+1, kH//2-1] = 0
            self.mask[:, :, kH//2-1, kH//2+1] = 0
            self.mask[:, :, kH//2-1, kH//2-1] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)