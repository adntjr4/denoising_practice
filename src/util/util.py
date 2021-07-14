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
    else:
        raise RuntimeError('wrong tensor dimensions : %s'%(t.shape,))

def imwrite_test(t, name='test'):
    cv2.imwrite('./%s.png'%name, tensor2np(t.cpu()))

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
