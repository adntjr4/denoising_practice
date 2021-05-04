import matplotlib as plt
import torch
import torch.nn.functional as F
import cv2
import numpy as np


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

def rot_hflip_img(img:torch.Tensor, rot_times:int, hflip:int):
    '''
    rotate '90 x times degree' & horizontal flip image 
    (shape of img: C, H, W)
    '''
    # no flip
    if hflip % 2 == 0:
        # 0 degrees
        if rot_times % 4 == 0:    
            return img
        # 90 degrees
        elif rot_times % 4 == 1:  
            return img.flip(1).transpose(1,2)
        # 180 degrees
        elif rot_times % 4 == 2:  
            return img.flip(2).flip(1)
        # 270 degrees
        else:               
            return img.flip(2).transpose(1,2)
    # horizontal flip
    else:
        # 0 degrees
        if rot_times % 4 == 0:    
            return img.flip(2)
        # 90 degrees
        elif rot_times % 4 == 1:  
            return img.flip(1).flip(2).transpose(1,2)
        # 180 degrees
        elif rot_times % 4 == 2:  
            return img.flip(1)
        # 270 degrees
        else:               
            return img.transpose(1,2)

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

