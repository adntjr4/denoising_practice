import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from . import regist_loss
from ..util.pytorch_msssim import ssim, msssim


@regist_loss
class L1():
    def __call__(self, input_data, model_output, data, module):
        if type(model_output) is tuple: output = model_output[0]
        else: output = model_output
        return F.l1_loss(output, data['clean'])

@regist_loss
class L2():
    def __call__(self, input_data, model_output, data, module):
        if type(model_output) is tuple: output = model_output[0]
        else: output = model_output
        return F.mse_loss(output, data['clean'])

@regist_loss
class SSIM():
    def __call__(self, input_data, model_output, data, module):
        if type(model_output) is tuple: output = model_output[0]
        else: output = model_output
        return 1-ssim(output, data['clean'])

@regist_loss
class MSSSIM():
    def __call__(self, input_data, model_output, data, module):
        if type(model_output) is tuple: output = model_output[0]
        else: output = model_output
        return 1-msssim(output, data['clean'])

@regist_loss
class VGG22(nn.Module):
    '''
    Perceptual loss : conv2-2
    '''
    def __init__(self):
        vgg_features = torchvision.models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        self.vgg = nn.Sequential(*modules[:8])
        self.vgg.requires_grad = False

    def forward(self, img1, img2):
        vgg1 = self.vgg(img1)
        vgg2 = self.vgg(img2)
        return F.mse_loss(vgg1, vgg2)

@regist_loss
class VGG54(nn.Module):
    '''
    Perceptual loss : conv5-4
    '''
    def __init__(self):
        vgg_features = torchvision.models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        self.vgg = nn.Sequential(*modules[:35])
        self.vgg.requires_grad = False

    def forward(self, img1, img2):
        vgg1 = self.vgg(img1)
        vgg2 = self.vgg(img2)
        return F.mse_loss(vgg1, vgg2)
