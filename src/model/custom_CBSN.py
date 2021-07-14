import math

import torch
from torch._C import has_cuda
import torch.nn as nn
import torch.nn.functional as F

from ..util.util import pixel_shuffle_down_sampling, pixel_shuffle_up_sampling, random_PD_down, random_PD_up
from . import regist_model, get_model_object
from .DBSN import DBSN
from .common import ResBlock, LocalMeanNet


@regist_model
class CBSN(nn.Module):
    def __init__(self, pd=4, pd_pad=0):
        super().__init__()

        self.pd      = pd
        self.pd_pad  = pd_pad

        self.bsn = DBSN(in_ch=3, out_ch=3, base_ch=32)

        self.Rnet = Blocks(3,1)
        self.Gnet = Blocks(3,1)
        self.Bnet = Blocks(3,1)

    def forward(self, x):
        # pd, forward
        pd_x, indice = random_PD_down(x, f=self.pd, pad=self.pd_pad)
        pd_x_mean = self.bsn(pd_x)
        bsn_x = random_PD_up(pd_x_mean, indice, f=self.pd, pad=self.pd_pad)

        # forward main network
        R = self.Rnet(torch.cat((bsn_x[:,0:1], x[:,1:2], x[:,2:3]), dim=1))
        G = self.Gnet(torch.cat((x[:,0:1], bsn_x[:,1:2], x[:,2:3]), dim=1))
        B = self.Bnet(torch.cat((x[:,0:1], x[:,1:2], bsn_x[:,2:3]), dim=1))

        x = torch.cat((R, G, B), dim=1)

        return x

class Blocks(nn.Module):
    def __init__(self, in_ch, out_ch, n_block=5, base_ch=32):
        super().__init__()

        blocks = [nn.Conv2d(in_ch, base_ch, kernel_size=1)]

        for i in range(n_block):
            blocks.append(ResBlock(base_ch, kernel_size=3, act='ReLU', bias=True))

        blocks.append(nn.Conv2d(base_ch, out_ch, kernel_size=1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)
