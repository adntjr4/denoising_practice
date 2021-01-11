import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(true_img, test_img):
    '''
    image value range : [0 - 255]
    clipping for model output
    '''
    # tensor to numpy
    if isinstance(true_img, torch.Tensor):
        true_img = true_img.detach().cpu().numpy()
    if isinstance(test_img, torch.Tensor):
        test_img = test_img.detach().cpu().numpy()

    # numpy value cliping & chnage type to uint8
    true_img = np.clip(true_img, 0, 255).astype(np.uint8)
    test_img = np.clip(test_img, 0, 255).astype(np.uint8)

    return peak_signal_noise_ratio(true_img, test_img)

def ssim(img1, img2, **kargs):
    '''
    image value range : [0 - 255]
    clipping for model output
    '''
    # tensor to numpy
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()

    # numpy value cliping
    img1 = np.clip(img1, 0, 255).astype(np.uint8)
    img2 = np.clip(img2, 0, 255).astype(np.uint8)

    return structural_similarity(img1, img2, **kargs)