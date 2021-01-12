import random, math
from importlib import import_module

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..util.util import rot_hflip_img
from .dataset_util.mask import RandomSampler, StratifiedSampler, VoidReplacer, RandomReplacer

dataset_module = {
                    # BSD
                    'BSD68'     : 'BSD',
                    'BSD400'    : 'BSD',
                    'CBSD68'    : 'BSD',
                    'CBSD432'   : 'BSD',

                    # SIDD
                    'SIDD'      : 'SIDD', 


                }
     
def get_dataset_object(dataset_name):
    if dataset_name is None:
        return None
    elif len(dataset_name.split('+')) > 1:
        raise NotImplementedError
    else:
        module_dset = import_module('src.datahandler.{}'.format(dataset_module[dataset_name]))
        return getattr(module_dset, dataset_name)
    
class DenoiseDataSet(Dataset):
    def __init__(self, add_noise=None, mask=None, crop_size=None, aug=None, n_repeat=1):
        self.dataset_dir = './dataset'
        
        self.add_noise_type, self.add_noise_opt = self._parse_add_noise(add_noise)
        self.sampler, self.replacer = self._parse_mask(mask)

        self.crop_size = crop_size
        self.aug = aug
        self.n_repeat = n_repeat

        self._scan()

    def __len__(self):
        return len(self.img_paths) * self.n_repeat

    def __getitem__(self, idx):
        '''
        final dictionary shape of data:
        data{'clean', 'syn_noisy', 'real_noisy', etc}
        '''
        data_idx = idx % len(self.img_paths)
        data = self._load_data(data_idx)

        # preprocessing include extract patch
        data = self._pre_processing(data)

        # synthesize noise
        if self.add_noise_type is not None:
            if 'clean' in data:
                data['syn_noisy'] = self._add_noise(data['clean'], self.add_noise_type, self.add_noise_opt)
            else:
                raise RuntimeError('there is no clean image to synthesize. (synthetic noise type: %s)'%self.add_noise_type)

        # mask on noisy image
        if self.sampler is not None or self.replacer is not None:
            # mask on noisy image (real_noise is higher priority)
            if 'real_noisy' in data:
                data['masked'] = self._mask(data['real_noisy'])
            elif 'syn_noisy' in data:
                data['masked'] = self._mask(data['syn_noisy'])
            else:
                raise RuntimeError('there is no noisy image for masking.')

        # data augmentation
        if self.aug != None:
            data = self._augmentation(data, self.aug)

        return data

    def _load_data(self, data_idx):
        raise NotImplementedError
        # TODO load possible data as dictionary 

    def _scan(self):
        raise NotImplementedError
        # TODO init and fill in self.img_paths

    def _load_img(self, img_name):
        img = cv2.imread(img_name, -1)
        assert img is not None, "failure on loading image - %s"%img_name
        if len(img.shape) != 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2,0,1))
        else:
            img = np.expand_dims(img, axis=0)
        img = torch.Tensor(img)
        return img

    def _pre_processing(self, data):
        # get patches from image
        if self.crop_size != None:
            data = self._get_patch(self.crop_size, data)
        
        # another pre-processing??

        return data

    def _get_patch(self, crop_size, data):
        # check all image size is same
        if 'clean' in data and 'real_noisy' in data:
            assert data['clean'].shape[1] == data['clean'].shape[1] and data['real_noisy'].shape[2] == data['real_noisy'].shape[2], \
            'img shape should be same. (%d, %d) != (%d, %d)' % (data['clean'].shape[1], data['clean'].shape[1], data['real_noisy'].shape[2], data['real_noisy'].shape[2])

        # get image shape and select random crop location
        if 'clean' in data:
            max_x = data['clean'].shape[2] - crop_size[0]
            max_y = data['clean'].shape[1] - crop_size[1]
        else:
            max_x = data['real_noisy'].shape[2] - crop_size[0]
            max_y = data['real_noisy'].shape[1] - crop_size[1]
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)

        # crop
        if 'clean' in data:
            data['clean'] = data['clean'][:, y:y+crop_size[1], x:x+crop_size[0]]
        if 'real_noisy' in data:
            data['real_noisy'] = data['real_noisy'][:, y:y+crop_size[1], x:x+crop_size[0]]
        
        return data

    def _parse_add_noise(self, add_noise_str):
        if add_noise_str != None:
            add_noise_type = add_noise_str.split('-')[0]
            add_noise_opt = [float(v) for v in add_noise_str.split('-')[1].split(':')]
            return add_noise_type, add_noise_opt
        return None, None

    def _add_noise(self, clean_img:torch.Tensor, add_noise_type:str, opt) -> torch.Tensor:
        if add_noise_type == 'uni':
            # add uniform noise
            return clean_img + 2*opt[0] * torch.rand(clean_img.shape) - opt[0]
        elif add_noise_type == 'gau':
            # add AWGN
            return clean_img + torch.normal(mean=0., std=opt[0], size=clean_img.shape)
        elif add_noise_type == 'gau_blind':
            # add blind gaussian noise
            return clean_img + torch.normal(mean=0., std=random.uniform(opt[0], opt[1]), size=clean_img.shape)
        elif add_noise_type == 'poi_gau':
            # add poisson guassian noise
            poi_gau_std = (clean_img * (opt[0]**2) + torch.ones(clean_img.shape) * (opt[1]**2)).sqrt()
            return clean_img + torch.normal(mean=0., std=poi_gau_std)
        else:
            raise RuntimeError('undefined additive noise type : %s'%add_noise_type)

    def _parse_mask(self, mask_str):
        if mask_str == None:
            return None, None

        mask_sampling = mask_str.split('-')[0]
        mask_sampling_method = mask_sampling.split('_')[0]
        n_mask_sampling = int(mask_sampling.split('_')[1]) if '_' in mask_sampling else None
        mask_selection_method = mask_str.split('-')[1] if '-' in mask_str else None

        # get sampler for sampling pixels from noisy image as much as number of sampling
        if mask_sampling_method == 'bypass':
            # no mask or replacement on image.
            return None, lambda x: x.clone()
        elif mask_sampling_method == 'rnd':
            # random sampling.
            sampler = RandomSampler(N=n_mask_sampling)
        elif mask_sampling_method == 'stf':
            # stratified sampling.
            sampler = StratifiedSampler(N=n_mask_sampling)
        else:
            raise RuntimeError('undefined mask sampling method type : %s'%mask_sampling_method)

        # get replacer for replacing pixels at masked location.
        if mask_selection_method == 'void':
            # fill with zero value
            replacer = VoidReplacer()
        elif mask_selection_method == 'rnd':
            # random selection.
            replacer = RandomReplacer()
        else:
            raise RuntimeError('undefined mask replacing method type : %s'%mask_selection_method)

        return sampler, replacer

    def _mask(self, noisy_img:torch.Tensor):
        if self.sampler is None:
            return self.replacer(noisy_img)
        else:
            mask_map = self.sampler(noisy_img.shape)
            return self.replacer(noisy_img, mask_map)

    def _augmentation(self, data:dict, aug:list):
        # random choice for rotation and flipping
        data_aug = {'rot':0, 'hflip':0}
        for one_aug in aug:
            if one_aug == 'rot':
                data_aug['rot'] = random.randint(0,3)
            elif one_aug == 'hflip':
                data_aug['hflip'] = random.randint(0,1)
        
        # for every data(only image), apply data augmentation.
        for key in data:
            # is image (if data length of dimension is 3.)
            if isinstance(data[key], torch.Tensor):
                if len(data[key].shape) == 3:
                    data[key] = rot_hflip_img(data[key], data_aug['rot'], data_aug['hflip'])

        return data

        
            
