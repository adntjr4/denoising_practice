from math import exp

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def np2tensor(n:np.array):
    '''
    transform numpy array (image) to torch Tensor
    BGR -> RGB
    (w,h,c) -> (c,w,h)
    '''
    # gray
    if len(n.shape) == 2:
        return torch.from_numpy(np.ascontiguousarray(np.transpose(n, (2,0,1))))
    # RGB -> BGR
    elif len(n.shape) == 3:
        return torch.from_numpy(np.ascontiguousarray(np.transpose(np.flip(n, axis=2), (2,0,1))))
    else:
        raise RuntimeError('wrong numpy dimensions : %s'%(n.shape,))

def tensor2np(t:torch.Tensor):
    '''
    transform torch Tensor to numpy having opencv image form.
    RGB -> BGR
    (c,w,h) -> (w,h,c)
    '''
    # gray
    if len(t.shape) == 2:
        return t.cpu().permute(1,2,0).numpy()
    # RGB -> BGR
    elif len(t.shape) == 3:
        return np.flip(t.cpu().permute(1,2,0).numpy(), axis=2)
    # image batch
    elif len(t.shape) == 4:
        return np.flip(t.cpu().permute(0,2,3,1).numpy(), axis=3)
    else:
        raise RuntimeError('wrong tensor dimensions : %s'%(t.shape,))

def imwrite_test(t, name='test'):
    cv2.imwrite('./%s.png'%name, tensor2np(t.cpu()))

def imread_tensor(name='test'):
    return np2tensor(cv2.imread('./%s'%name))

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

def pixel_shuffle_down_sampling(x:torch.Tensor, f:int, pad:int=0, pad_value:float=0.):
    '''
    pixel-shuffle down-sampling (PD) from "When AWGN-denoiser meets real-world noise." (AAAI 2019)
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad between each down-sampled images
        pad_value (float) : padding value
    Return:
        pd_x (Tensor) : down-shuffled image tensor with pad or not
    '''
    # single image tensor
    if len(x.shape) == 3:
        c,w,h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(c,f,f,w//f+2*pad,h//f+2*pad).permute(0,1,3,2,4).reshape(c, w+2*f*pad, h+2*f*pad)
    # batched image tensor
    else:
        b,c,w,h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(b,c,f,f,w//f+2*pad,h//f+2*pad).permute(0,1,2,4,3,5).reshape(b,c,w+2*f*pad, h+2*f*pad)

def pixel_shuffle_up_sampling(x:torch.Tensor, f:int, pad:int=0):
    '''
    reverse of pixel-shuffle down-sampling (PD)
    see more details about PD in pixel_shuffle_down_sampling()
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad will be removed
    '''
    # single image tensor
    if len(x.shape) == 3:
        c,w,h = x.shape
        before_shuffle = x.view(c,f,w//f,f,h//f).permute(0,1,3,2,4).reshape(c*f*f,w//f,h//f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)   
    # batched image tensor
    else:
        b,c,w,h = x.shape
        before_shuffle = x.view(b,c,f,w//f,f,h//f).permute(0,1,2,4,3,5).reshape(b,c*f*f,w//f,h//f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)

def random_PD_down(x:torch.Tensor, f:int, pad:int=0, pad_value:float=0.):
    '''
    Random PD process
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad between each down-sampled images
        pad_value (float) : padding value
    Return:
        pd_x (Tensor) : down-shuffled image tensor with pad or not
        indice (Tensor) : indice of down-shuffled image
    '''
    # single image tensor
    if len(x.shape) == 3:
        c,w,h = x.shape
        unshuffled = F.pixel_unshuffle(x, f).view(c, f*f, w//f, h//f)
        
        # make indice
        indice = torch.rand(1, f*f, w//f, h//f, device=x.device)
        indice = indice.argsort(dim=1)

        # random shuffle
        unshuffled = torch.gather(unshuffled, dim=1, index=indice.expand(c,f*f,w//f,h//f))

        # padding
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)

        pd_x = unshuffled.view(c,f,f,w//f+2*pad,h//f+2*pad).permute(0,1,3,2,4).reshape(c, w+2*f*pad, h+2*f*pad)

    # batched image tensor
    else:
        b,c,w,h = x.shape
        unshuffled = F.pixel_unshuffle(x, f).view(b, c, f*f, w//f, h//f)

        # make indice
        indice = torch.rand(b, 1, f*f, w//f, h//f, device=x.device)
        indice = indice.argsort(dim=2)

        # random shuffle
        unshuffled = torch.gather(unshuffled, dim=2, index=indice.expand(b,c,f*f,w//f,h//f))

        # padding
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)

        pd_x = unshuffled.view(b,c,f,f,w//f+2*pad,h//f+2*pad).permute(0,1,2,4,3,5).reshape(b,c,w+2*f*pad, h+2*f*pad)

    return pd_x, indice

def random_PD_up(x:torch.Tensor, indice:torch.Tensor, f:int, pad:int=0):
    '''
    reverse of random PD process
    Args:
        x (Tensor) : input tensor
        indice (Tensor) : indice of down-shuffled image
        f (int) : factor of PD
        pad (int) : number of pad will be removed
    '''
    # single image tensor
    if len(x.shape) == 3:
        c,w,h = x.shape
        before_shuffle = x.view(c,f,w//f,f,h//f).permute(0,1,3,2,4).reshape(c*f*f,w//f,h//f)

        # remove pad
        if pad != 0: 
            before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
            w, h = w-2*f*pad, h-2*f*pad
        before_shuffle = before_shuffle.reshape(c,f*f,w//f,h//f)

        # reverse indice
        r_indice = indice.argsort(dim=1)

        # unshuffle
        before_shuffle = torch.gather(before_shuffle, dim=1, index=r_indice.expand(c,f*f,w//f,h//f))
        before_shuffle = before_shuffle.reshape(c*f*f,w//f,h//f)

        return F.pixel_shuffle(before_shuffle, f)   
    # batched image tensor
    else:
        b,c,w,h = x.shape
        before_shuffle = x.view(b,c,f,w//f,f,h//f).permute(0,1,2,4,3,5).reshape(b,c*f*f,w//f,h//f)

        # remove pad
        if pad != 0: 
            before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
            w, h = w-2*f*pad, h-2*f*pad
        before_shuffle = before_shuffle.reshape(b,c,f*f,w//f,h//f)

        # reverse indice
        r_indice = indice.argsort(dim=2)

        # unshuffle
        before_shuffle = torch.gather(before_shuffle, dim=2, index=r_indice.expand(b,c,f*f,w//f,h//f))
        before_shuffle = before_shuffle.reshape(b,c*f*f,w//f,h//f)
        
        return F.pixel_shuffle(before_shuffle, f)

def human_format(num):
    magnitude=0
    while abs(num)>=1000:
        magnitude+=1
        num/=1000.0
    return '%.1f%s'%(num,['','K','M','G','T','P'][magnitude])

def psnr(img1, img2):
    '''
    image value range : [0 - 255]
    clipping for model output
    '''
    # tensor to numpy
    if isinstance(img1, torch.Tensor):
        img1 = tensor2np(img1)
    if isinstance(img2, torch.Tensor):
        img2 = tensor2np(img2)

    # numpy value cliping & chnage type to uint8
    img1 = np.clip(img1, 0, 255).astype(np.uint8)
    img2 = np.clip(img2, 0, 255).astype(np.uint8)

    return peak_signal_noise_ratio(img1, img2)

def ssim(img1, img2):
    '''
    image value range : [0 - 255]
    clipping for model output
    '''
    def remove_batch_demension(img):
        assert len(img.shape) == 4
        return img[0]
    def any2np(img):
        if isinstance(img, torch.Tensor):
            img = tensor2np(img)
        return img

    # convert to single image, tensor to numpy
    img1 = any2np(remove_batch_demension(img1))
    img2 = any2np(remove_batch_demension(img2))

    # numpy value cliping
    img1 = np.clip(img1, 0, 255).astype(np.uint8)
    img2 = np.clip(img2, 0, 255).astype(np.uint8)

    return structural_similarity(img1, img2, multichannel=True)

def get_gaussian_2d_filter(window_size, sigma, channel=1, device=torch.device('cpu')):
    '''
    return 2d gaussian filter window as tensor form
    Arg:
        window_size : filter window size
        sigma : standard deviation
    '''
    gauss = torch.ones(window_size, device=device)
    for x in range(window_size): gauss[x] = exp(-(x - window_size//2)**2/float(2*sigma**2))
    gauss = gauss.unsqueeze(1)
    #gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)], device=device).unsqueeze(1)
    filter2d = gauss.mm(gauss.t()).float()
    filter2d = (filter2d/filter2d.sum()).unsqueeze(0).unsqueeze(0)
    return filter2d.expand(channel, 1, window_size, window_size)

def get_mean_2d_filter(window_size, channel=1, device=torch.device('cpu')):
    '''
    return 2d mean filter as tensor form
    Args:
        window_size : filter window size
    '''
    window = torch.ones((window_size, window_size), device=device)
    window = (window/window.sum()).unsqueeze(0).unsqueeze(0)
    return window.expand(channel, 1, window_size, window_size)

def mean_conv2d(x, window_size=None, window=None, filter_type='gau', sigma=1.0, padd=True):
    '''
    color channel-wise 2d mean or gaussian convolution
    Args:
        x : input image
        window_size : filter window size
        filter_type(opt) : 'gau' or 'mean'
        sigma : standard deviation of gaussian filter
    '''    
    if window is None:
        if filter_type == 'gau':
            window = get_gaussian_2d_filter(window_size, sigma=sigma, channel=x.shape[1], device=x.device)
        else:
            window = get_mean_2d_filter(window_size, channel=x.shape[1], device=x.device)

    padding = window.shape[2]//2 if padd else 0
    return F.conv2d(x, window, padding=padding, groups=x.shape[1])

def variance_conv2d(x, window_size=None, window=None, filter_type='gau', sigma=1.0, padd=True):
    '''
    calculate variance using mean filter(mean_conv2d)
    '''
    if window is None:
        if filter_type == 'gau':
            window = get_gaussian_2d_filter(window_size, sigma=sigma, channel=x.shape[1], device=x.device)
        else:
            window = get_mean_2d_filter(window_size, channel=x.shape[1], device=x.device)

    mean = mean_conv2d(x, window=window, padd=padd)
    return mean_conv2d(x.pow(2), window=window, padd=padd) - mean.pow(2)

if __name__ == '__main__':
    t = torch.randn(1,3,5,5).cuda()
    print(mean_conv2d(t, window_size=3, filter_type='mean'))
