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
    ( )
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
