import os, sys, argparse, cv2
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from src.datahandler import get_dataset_object
from src.util.util import pixel_shuffle_up_sampling, tensor2np

class LocalVarianceNet(nn.Module):
    def __init__(self, in_ch, window_size):
        super().__init__()

        self.in_ch = in_ch
        self.ws = window_size

        self.average_net = nn.Conv2d(in_ch, in_ch, kernel_size=self.ws, padding=self.ws//2, padding_mode='circular', bias=False, groups=in_ch)
        self.average_net.weight.data.fill_(1/(self.ws**2))
        for param in self.average_net.parameters():
            param.requires_grad = False

    def forward(self, x):
        squared = torch.square(x)
        return self.average_net(squared) - torch.square(self.average_net(x))        

class NLFNet(nn.Module):
    def __init__(self, window_size=9, real=False):
        super().__init__()

        self.gamma = 2.0
        self.real = real

        self.lvn3 = LocalVarianceNet(3, window_size)
        self.lvn1 = LocalVarianceNet(1, window_size)

    def forward(self, x):
        alpha_map = self.lvn3(x).sum(1, keepdim=True)/3
        beta_map = self.lvn1(x.sum(1, keepdim=True)/3)

        weight = self.lvn3(x-x[:, [2,1,0]]).sum(1, keepdim=True)/3
        _,_,w,h = weight.shape

        w_sum = weight.sum()
        weight = torch.exp(-self.gamma * w*h * weight / w_sum)

        # weight = F.softmax(weight.view(1,1,w*h), dim=2).view(1,1,w,h)

        w_sum = weight.sum()
        if self.real:
            return 9/4*(weight*(alpha_map-beta_map)).sum() / w_sum
        else:
            return 3/2*(weight*(alpha_map-beta_map)).sum() / w_sum


nlf_net = NLFNet()
dataset = get_dataset_object('Synthesized_CBSD432_25')(crop_size=(128,128))
data = dataset.__getitem__(5)
noisy_image = data['real_noisy']
nlf = 625.0

nlf_net = NLFNet(real=True)
dataset = get_dataset_object('prep_SIDD')(crop_size=(160,160))
ratio = []
for i in range(500):
    data = dataset.__getitem__(i)
    noisy_image = data['real_noisy']
    clean_image = data['clean']
    noise_map = noisy_image - clean_image
    nlf = torch.var(noise_map, dim=(-1,-2))
    ratio.append([nlf[0]/nlf[1], nlf[2]/nlf[1]])

ratio = torch.Tensor(ratio)

print('nlf: ', float(nlf_net(noisy_image.unsqueeze(0))))
print('gt: ', nlf)
print(sum(ratio[:,0])/len(ratio), sum(ratio[:,1])/len(ratio))


