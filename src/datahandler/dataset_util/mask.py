import math

import torch


class Sampler():
    def __init__(self):
        pass

    def __call__(self, shape):
        raise NotImplementedError

class RandomSampler(Sampler):
    def __init__(self, N_ratio):
        super().__init__()
        self.N_ratio = N_ratio

    def __call__(self, shape):
        '''
        Args:
            shape(tuple) : (C, H, W)
        Return:
            mask(Tensor) : (C, H, W)
        '''
        C, H, W = shape
        N = int(H*W*self.N_ratio)
        mask = torch.zeros((C, H, W), dtype=torch.bool)

        for ch in range(C):
            perm = torch.randperm(H*W)[:N]
            mask[ch, perm//W, perm%W] = 1.

        return mask

class StratifiedSampler(Sampler):
    def __init__(self, N_ratio):
        super().__init__()
        self.N_ratio = N_ratio
        raise RuntimeError('there is a bug need to fix on __call__() of StratifiedSampler. (on using N & is not exactly stratified sampling.')

    def __call__(self, shape, box_size=20):
        '''
        Args:
            shape : C, H, W
        '''
        C, H, W = shape
        N = int(H*W*self.N_ratio)
        mask = torch.zeros((C, H, W), dtype=torch.bool)

        for ch in range(C):
            perm_x = torch.randperm(W)[:N]
            perm_y = torch.randperm(H)[:N]

            mask[ch, perm_y, perm_x] = 1.

        return mask

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
    def __init__(self, range=None):
        super().__init__()

        self.range = range

    def __call__(self, noisy_img:torch.Tensor, mask:torch.Tensor):
        C, H, W = mask.shape
        N = int(mask[0].sum())

        if self.range is None:
            # find replacement value from whole image
            masked = noisy_img.clone()

            for ch in range(C):
                perm = torch.randperm(H*W)[:N]
                masked[ch, mask[ch]] = noisy_img[ch, perm//W, perm%W]

            return masked
        else:
            # fine value from in ranged patch of each masked pixel
            masked = noisy_img.clone()

            x = torch.IntTensor(range(W)).repeat(H,1)
            y = torch.IntTensor(range(H)).repeat(W,1).transpose(0,1)

            for ch in range(C):
                mask_x = x[mask[ch]]
                mask_y = y[mask[ch]]

                x_shift = torch.randint(-self.range, self.range+1, [int(N)])
                y_shift = torch.randint(-self.range, self.range+1, [int(N)])

                replaced_x = (mask_x + x_shift).clip(0, W-1)
                replaced_y = (mask_y + y_shift).clip(0, H-1)

                masked[ch, mask[ch]] = noisy_img[ch, replaced_y, replaced_x]

            return masked

if __name__ == "__main__":
    ss = StratifiedSampler(3)

    print(ss((3,8,8)))