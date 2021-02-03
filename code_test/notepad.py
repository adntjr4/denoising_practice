import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter


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

def gaussian_kernel(size=21, sigma=3):
    """Returns a 2D Gaussian kernel.
    Parameters
    ----------
    size : float, the kernel size (will be square)

    sigma : float, the sigma Gaussian parameter

    Returns
    -------
    out : array, shape = (size, size)
      an array with the centered gaussian kernel
    """
    x = np.linspace(- (size // 2), size // 2)
    x /= np.sqrt(2)*sigma
    x2 = x**2
    kernel = np.exp(- x2[:, None] - x2[None, :])
    return kernel / kernel.sum()

gau_noise = torch.normal(mean=0., std=25, size=(3,64,64))

struc_gau = torch.Tensor(gaussian_filter(gau_noise, sigma=1))*9

cv2.imwrite('i.png', tensor2np(gau_noise))
cv2.imwrite('s.png', tensor2np(struc_gau))