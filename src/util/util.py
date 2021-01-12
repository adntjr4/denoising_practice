import matplotlib as plt
import torch
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
