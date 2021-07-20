import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from . import regist_loss
from ..util.util import get_gaussian_2d_filter, mean_conv2d, variance_conv2d


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
        return (1-ssim(output, data['clean'])).mean()

@regist_loss
class MSSSIM():
    def __call__(self, input_data, model_output, data, module):
        if type(model_output) is tuple: output = model_output[0]
        else: output = model_output
        return (1-msssim(output, data['clean'])).mean()

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

def ssim(img1, img2, sigma=3.0, include_cs=False):
    '''
    return custom ssim using gaussian kernel
    Args:
        img1 (Tensor) : (B, C, H, W)
        img2 (Tensor) : (B, C, H, W)
        sigma: std of gaussian kernel
    Return:
        ssim (Tensor): (B)
    image range : [0,255]
    '''
    # hyper-parameters
    C1 = (0.01*255) ** 2
    C2 = (0.03*255) ** 2
    # window_size = 11
    # I decide to use 6*sigma+1 size of kernel
    # refer here : https://kr.mathworks.com/matlabcentral/answers/231351-what-is-the-correct-number-of-pixels-for-a-gaussian-with-a-given-standard-deviation

    # get gaussian kernel
    tmp = int(6*sigma)
    window_size = tmp + (tmp+1)%2
    window = get_gaussian_2d_filter(window_size, sigma, channel=img1.shape[1]).to(img1.device)

    # calculate mean and variance of each image
    mu1 = mean_conv2d(img1, window=window, padd=False)
    mu2 = mean_conv2d(img2, window=window, padd=False)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = variance_conv2d(img1, window=window, padd=False)
    sigma2_sq = variance_conv2d(img2, window=window, padd=False)
    sigma12 = mean_conv2d(img1 * img2, window=window, padd=False) - mu1_mu2

    # calculate SSIM
    v1 = 2 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    # average over images in batch
    if include_cs: return ssim_map.mean((1,2,3)), (v1/v2).mean((1,2,3))
    return ssim_map.mean((1,2,3))

def msssim(img1, img2, sigma_list=[0.5, 1.0, 2.0, 4.0, 8.0]):
    ssims, cs = [], []
    for sigma in sigma_list:
        ssim_, cs_ = ssim(img1, img2, sigma=sigma, include_cs=True)
        ssims.append(ssim_)
        cs.append(cs_)
    
    return torch.prod(torch.stack(cs[:-1], dim=1), dim=1) * ssims[-1]
