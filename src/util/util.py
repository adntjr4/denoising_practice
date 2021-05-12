import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def tensor2np(t):
    '''
    transform torch Tensor to numpy having opencv image form.
    # color
    (this function assumed tensor have RGB format and opencv have BGR format)
    # gray
    (just transform into numpy)
    '''
    # gray
    if len(t.shape) == 2:
        return t.numpy()
    elif len(t.shape) == 3: # RGB -> BGR
        return np.flip(t.permute(1,2,0).numpy(), axis=2)
    elif len(t.shape) == 4:
        raise RuntimeError('multiple images cannot be transformed to numpy array.') 
    else:
        raise RuntimeError('wrong tensor dimensions : %s'%(t.shape,))

def rot_hflip_img(img:torch.Tensor, rot_times:int=0, hflip:int=0):
    '''
    rotate '90 x times degree' & horizontal flip image 
    (shape of img: b,c,h,w or c,h,w)
    '''
    b=0 if len(img.shape)==3 else 1
    # no flip
    if hflip % 2 == 0:
        # 0 degrees
        if rot_times % 4 == 0:    
            return img
        # 90 degrees
        elif rot_times % 4 == 1:  
            return img.flip(b+1).transpose(b+1,b+2)
        # 180 degrees
        elif rot_times % 4 == 2:  
            return img.flip(b+2).flip(b+1)
        # 270 degrees
        else:               
            return img.flip(b+2).transpose(b+1,b+2)
    # horizontal flip
    else:
        # 0 degrees
        if rot_times % 4 == 0:    
            return img.flip(b+2)
        # 90 degrees
        elif rot_times % 4 == 1:  
            return img.flip(b+1).flip(b+2).transpose(b+1,b+2)
        # 180 degrees
        elif rot_times % 4 == 2:  
            return img.flip(b+1)
        # 270 degrees
        else:               
            return img.transpose(b+1,b+2)

def pixel_shuffle_down_sampling(x, f):
    '''
    pixel-shuffle down-sampling (PD) from "When AWGN-denoiser meets real-world noise." (AAAI 2019)
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
    '''
    # single image tensor
    if len(x.shape) == 3:
        c,h,w = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        return unshuffled.view(c,f,f,h//f,w//f).permute(0,1,3,2,4).reshape(c,h,w)
    # batched image tensor
    else:
        b,c,h,w = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        return unshuffled.view(b,c,f,f,h//f,w//f).permute(0,1,2,4,3,5).reshape(b,c,h,w)

def pixel_shuffle_up_sampling(x, f):
    '''
    reverse of pixel-shuffle down-sampling (PD)
    see more details about PD in pixel_shuffle_down_sampling()
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
    '''
    # single image tensor
    if len(x.shape) == 3:
        c,h,w = x.shape
        before_shuffle = x.view(c,f,h//f,f,w//f).permute(0,1,3,2,4).reshape(c*f*f,h//f,w//f)
        return F.pixel_shuffle(before_shuffle, f)   
    # batched image tensor
    else:
        b,c,h,w = x.shape
        before_shuffle = x.view(b,c,f,h//f,f,w//f).permute(0,1,2,4,3,5).reshape(b,c*f*f,h//f,w//f)
        return F.pixel_shuffle(before_shuffle, f)

def human_format(num):
    magnitude=0
    while abs(num)>=1000:
        magnitude+=1
        num/=1000.0
    return '%.1f%s'%(num,['','K','M','G','T','P'][magnitude])

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
