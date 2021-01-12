import math

import torch


class Sampler():
    def __init__(self):
        pass

    def __call__(self, shape):
        raise NotImplementedError

class RandomSampler(Sampler):
    def __init__(self, N):
        super().__init__()
        self.N = N

    def __call__(self, shape):
        '''
        Args:
            shape : C, H, W
        '''
        C, H, W = shape
        perm = torch.randperm(H*W)[:self.N]
        mask = torch.zeros((H, W), dtype=torch.bool)
        mask[perm//W, perm%W] = True
        return mask

class StratifiedSampler(Sampler):
    def __init__(self, N):
        super().__init__()
        self.N = N

        self.n_split = int(math.sqrt(N))

    def __call__(self, shape):
        '''
        Args:
            shape : C, H, W
        '''
        raise NotImplementedError


class Replacer():
    def __init__(self):
        pass

    def __call__(self, noisy_img:torch.Tensor, mask:torch.Tensor):
        raise NotImplementedError
        
    def void_mask(self, noisy_img:torch.Tensor, mask:torch.Tensor):
        return noisy_img - noisy_img*mask

class VoidReplacer(Replacer):
    def __init__(self):
        super().__init__()

    def __call__(self, noisy_img:torch.Tensor, mask:torch.Tensor):
        return self.void_mask(noisy_img, mask)

class RandomReplacer(Replacer):
    def __init__(self):
        super().__init__()

    def __call__(self, noisy_img:torch.Tensor, mask:torch.Tensor):
        H, W = mask.shape
        N = mask.sum()
        perm = torch.randperm(H*W)[:N]

        masked = noisy_img.clone()

        masked[:, mask] = noisy_img[:, perm//W, perm%W]
        return masked
