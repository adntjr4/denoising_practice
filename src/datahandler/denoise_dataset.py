import random, math, os
from importlib import import_module

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from torch.utils.data import Dataset


from ..util.util import rot_hflip_img, tensor2np, pixel_shuffle_down_sampling
from .dataset_util.mask import RandomSampler, StratifiedSampler, VoidReplacer, RandomReplacer

    
class DenoiseDataSet(Dataset):
    def __init__(self, add_noise=None, mask=None, crop_size=None, aug=None, norm:bool=None, n_repeat:int=1, n_data:int=None, **kwargs):
        '''
        basic denoising dataset class for various dataset.
        for build specific dataset class below instruction must be implemented. (or see other dataset class already implemented.)
            - self._scan(self) : scan & save path, image name to self.img_paths
            - self._load_data(self, data_idx) : load all data you want as dictionary form with defined keys (see function for detail.)

        Args:
            add_noise (str or None)  : configuration of addictive noise to synthesize noisy image. (see details in yaml file)
            mask (str or None)       : configuration of mask method for self-supervised training.
            crop_size (list or None) : crop size [W, H]
            aug (list or None)       : data augmentation (see details in yaml file)
            norm (bool)              : flag of data normalization. (see datails in function)
            n_repeat (int)           : data repeating count
            n_data (int)             : set length of dataset to this number (using front data)
            kwargs:
                multiple_cliping(int)  : clipping height and width of image to be multiple of multiple_cliping (for UNet or PD).
                keep_on_mem (bool)   : flag of keeping all image data in memory (before croping image)
                pd (int)             : factor of pixel-shuffle down-sampling for images (1,2,3,4 are available. for more details see util.py)
        '''

        self.dataset_dir = './dataset'
        
        self.add_noise_type, self.add_noise_opt, self.add_noise_clamp = self._parse_add_noise(add_noise)
        self.sampler, self.replacer = self._parse_mask(mask)

        # set parameters for dataset.
        self.crop_size = crop_size
        self.aug = aug
        self.norm = norm
        self.n_repeat = n_repeat
        self.n_data = n_data
        self.kwargs = kwargs
        self.unshuffle = kwargs['pixel_unshuffle'] if 'pixel_unshuffle' in kwargs else None

        # scan all data and fill in self.img_paths
        self.img_paths = []
        self._scan()

        # check special options (keep_on_mem etc)
        self._do_others()

        # normalization factor
            # gray values from BSD400
        self.gray_means = torch.Tensor([110.73])
        self.gray_stds  = torch.Tensor([63.66])

        self.color_means = torch.Tensor([128.])
        self.color_stds  = torch.Tensor([64.])

    def __len__(self):
        if self.n_data is not None:
            return self.n_data * self.n_repeat
        return len(self.img_paths) * self.n_repeat

    def __getitem__(self, idx):
        '''
        final dictionary shape of data:
        data{'clean', 'syn_noisy', 'real_noisy', etc}
        '''
        data_idx = idx % len(self.img_paths)

        if not self._need_to_be_mem():
            data = self._load_data(data_idx)
        else:
            data = self.pre_loaded[data_idx]

        # pre-processing include extract patch
        data = self._pre_processing(data)

        # synthesize noise
        if not 'syn_noisy' in data:
            if self.add_noise_type is not None:
                if 'clean' in data:
                    syn_noisy_img, nlf = self._add_noise(data['clean'], self.add_noise_type, self.add_noise_opt, self.add_noise_clamp)
                    data['syn_noisy'] = syn_noisy_img
                    data['nlf'] = nlf
                elif 'real_noisy' in data:
                    syn_noisy_img, nlf = self._add_noise(data['real_noisy'], self.add_noise_type, self.add_noise_opt, self.add_noise_clamp)
                    data['syn_noisy'] = syn_noisy_img
                    data['nlf'] = nlf
                else:
                    raise RuntimeError('there is no clean or real image to synthesize. (synthetic noise type: %s)'%self.add_noise_type)

        # mask on noisy image
        if self.sampler is not None or self.replacer is not None:
            # mask on noisy image (real_noise is higher priority)
            if 'real_noisy' in data:
                mask, data['masked'] = self._mask(data['real_noisy'])
            elif 'syn_noisy' in data:
                mask, data['masked'] = self._mask(data['syn_noisy'])
            elif 'clean' in data:
                mask, data['masked'] = self._mask(data['clean'])
            else:
                raise RuntimeError('there is no noisy image for masking.')
            if mask is not None:
                data['mask'] = mask

        # data normalization
        if self.norm:
            data = self.normalize_data(data)

        # data augmentation
        if self.aug != None:
            data = self._augmentation(data, self.aug)

        return data

    def _load_data(self, data_idx):
        raise NotImplementedError
        # TODO load possible data as dictionary
        # dictionary key list :
        #   'clean' : clean image without noise (gt or anything).
        #   'real_noisy' : real noisy image or already synthesized noisy image.
        #   'instances' : any other information of capturing situation.

    def _scan(self):
        raise NotImplementedError
        # TODO fill in self.img_paths (include path from project directory)

    def _do_others(self):
        # keep on memory
        if self._need_to_be_mem():
            self._keep_on_mem()

    ##### Image handling functions #####

    def _load_img(self, img_name, gray=False):
        img = cv2.imread(img_name, -1)
        assert img is not None, "failure on loading image - %s"%img_name
        return self._load_img_from_np(img, gray)

    def _load_img_from_np(self, img, gray=False):
        if len(img.shape) != 2:
            if gray:
                # follows definition of sRBG in terms of the CIE 1931 linear luminance.
                # because calculation opencv color conversion and imread grayscale mode is a bit different.
                # https://en.wikipedia.org/wiki/Grayscale
                img = np.average(img, axis=2, weights=[0.0722, 0.7152, 0.2126])
                img = np.expand_dims(img, axis=0)
            else:
                img = np.flip(img, axis=2)
                img = np.transpose(img, (2,0,1))
        else:
            img = np.expand_dims(img, axis=0)
        return torch.from_numpy(np.ascontiguousarray(img)).float()

    def _pre_processing(self, data):
        C, H, W = data['clean'].shape if 'clean' in data else data['real_noisy'].shape

        # clipping edges for Unet input (make image size as multiple of parameter)
        if 'multiple_cliping' in self.kwargs:
            multiple = self.kwargs['multiple_cliping']
            cliped_H = H - (H % multiple)
            cliped_W = W - (W % multiple)
            data = self._get_patch([cliped_W, cliped_H], data, rnd=False)

        # pixel-shuffle down-sampling
        if 'pd' in self.kwargs:
            for key in data:
                if self._is_image_tensor(data[key]):
                    data[key] = pixel_shuffle_down_sampling(data[key], self.kwargs['pd'])

        # get patches from image
        if self.crop_size != None:
            data = self._get_patch(self.crop_size, data)

        # any other pre-processing??

        return data

    def _get_patch(self, crop_size, data, rnd=True):
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

        if rnd:
            x = np.random.randint(0, max_x)
            y = np.random.randint(0, max_y)
        else:
            x, y = 0, 0

        # crop
        if 'clean' in data:
            data['clean'] = data['clean'][:, y:y+crop_size[1], x:x+crop_size[0]]
        if 'real_noisy' in data:
            data['real_noisy'] = data['real_noisy'][:, y:y+crop_size[1], x:x+crop_size[0]]
        
        return data

    def normalize_data(self, data, cuda=False):
        # for all image
        for key in data:
            if self._is_image_tensor(data[key]):
                data[key] = self.normalize(data[key], cuda)
        return data

    def inverse_normalize_data(self, data, cuda=False):
        # for all image
        for key in data:
            # is image 
            if self._is_image_tensor(data[key]):
                data[key] = self.inverse_normalize(data[key], cuda)
        return data

    def normalize(self, img, cuda=False):
        if img.shape[0] == 1:
            stds = self.gray_stds
            means = self.gray_means
        elif img.shape[0] == 3:
            stds = self.color_stds
            means = self.color_means
        else:
            raise RuntimeError('undefined image channel length : %d'%img.shape[0])
        
        if cuda:
            means, stds = means.cuda(), stds.cuda() 

        return (img-means) / stds

    def inverse_normalize(self, img, cuda=False):
        if img.shape[0] == 1:
            stds = self.gray_stds
            means = self.gray_means
        elif img.shape[0] == 3:
            stds = self.color_stds
            means = self.color_means
        else:
            raise RuntimeError('undefined image channel length : %d'%img.shape[0])
        
        if cuda:
            means, stds = means.cuda(), stds.cuda() 

        return (img*stds) + means

    def _parse_add_noise(self, add_noise_str):
        if add_noise_str == 'bypass':
            return 'bypass', None, None
        elif add_noise_str != None:
            add_noise_type = add_noise_str.split('-')[0]
            add_noise_opt = [float(v) for v in add_noise_str.split('-')[1].split(':')]
            add_noise_clamp = len(add_noise_str.split('-'))>2 and add_noise_str.split('-')[2] == 'clamp'
            return add_noise_type, add_noise_opt, add_noise_clamp
        return None, None, None

    def _add_noise(self, clean_img:torch.Tensor, add_noise_type:str, opt, clamp=False) -> torch.Tensor:
        '''
        add various noise to clean image.
        Args:
            clean_img (Tensor) : clean image to synthesize on.
            add_noise_type : below types are available.
            opt (list) : optional args for synthsize noise.
        Return:
            synthesized_img
        Noise_types
            - uni (uniform distribution noise from -opt[0] ~ opt[0]
            - gau (gaussian distribution noise with zero-mean & opt[0] variance)
            - gau_bline (blind gaussian distribution with zero-mean, variance is uniformly selected from opt[0] ~ opt[1])
            - struc_gau (structured gaussian noise. gaussian filter is applied to above gaussian noise. opt[0] is variance of gaussian)
            - het_gau (heteroscedastic gaussian noise with indep weight:opt[0], dep weight:opt[1].)
        '''
        nlf = None

        if add_noise_type == 'bypass':
            # bypass clean image
            synthesized_img = clean_img

        elif add_noise_type == 'uni':
            # add uniform noise
            synthesized_img = clean_img + 2*opt[0] * torch.rand(clean_img.shape) - opt[0]

        elif add_noise_type == 'gau':
            # add AWGN
            nlf = opt[0]
            synthesized_img = clean_img + torch.normal(mean=0., std=nlf, size=clean_img.shape)
            
        elif add_noise_type == 'gau_blind':
            # add blind gaussian noise
            nlf = random.uniform(opt[0], opt[1])
            synthesized_img = clean_img + torch.normal(mean=0., std=nlf, size=clean_img.shape)

        elif add_noise_type == 'struc_gau':
            # add structured gaussian noise (saw in the paper "Noiser2Noise": https://arxiv.org/pdf/1910.11908.pdf)
            # TODO: is this gaussian filter correct?
            gau_noise = torch.normal(mean=0., std=opt[0], size=clean_img.shape)
            struc_gau = torch.Tensor(gaussian_filter(gau_noise, sigma=1))
            synthesized_img = clean_img + struc_gau

        elif add_noise_type == 'het_gau':
            # add heteroscedastic  guassian noise
            poi_gau_std = (clean_img * (opt[0]**2) + torch.ones(clean_img.shape) * (opt[1]**2)).sqrt()
            nlf = poi_gau_std
            synthesized_img = clean_img + torch.normal(mean=0., std=nlf)

        else:
            raise RuntimeError('undefined additive noise type : %s'%add_noise_type)

        if clamp:
            synthesized_img = torch.clamp(synthesized_img, 0, 255)

        return synthesized_img, nlf

    def _parse_mask(self, mask_str):
        if mask_str == None:
            return None, None

        mask_sampling = mask_str.split('-')[0]
        mask_sampling_method = mask_sampling.split('_')[0]
        n_mask_sampling = float(mask_sampling.split('_')[1]) if '_' in mask_sampling else None
        mask_selection_method = mask_str.split('-')[1] if '-' in mask_str else None

        # get sampler for sampling pixels from noisy image as much as number of sampling
        if mask_sampling_method == 'bypass':
            # no mask or replacement on image.
            return lambda shape: torch.ones(shape), lambda img, mask: img.clone()
        elif mask_sampling_method == 'rnd':
            # random sampling.
            sampler = RandomSampler(N_ratio=n_mask_sampling)
        elif mask_sampling_method == 'stf':
            # stratified sampling.
            sampler = StratifiedSampler(N_ratio=n_mask_sampling)
        else:
            raise RuntimeError('undefined mask sampling method type : %s'%mask_sampling_method)

        # get replacer for replacing pixels at masked location.
        if mask_selection_method == 'void':
            # fill with zero value
            replacer = VoidReplacer()
        elif mask_selection_method == 'rnd':
            # random selection.
            replacer = RandomReplacer()
        elif 'rnd' in mask_selection_method:
            # random selection with range
            selection_range = int(mask_selection_method.split('_')[1])
            replacer = RandomReplacer(range=selection_range)
        else:
            raise RuntimeError('undefined mask replacing method type : %s'%mask_selection_method)

        return sampler, replacer

    def _mask(self, noisy_img:torch.Tensor):
        mask_map = self.sampler(noisy_img.shape)
        return mask_map, self.replacer(noisy_img, mask_map)

    def _augmentation(self, data:dict, aug:list):
        '''
        Parsing augmentation list and apply it to the data images.
        '''
        # parsign augmentation
        rot, hflip = 0, 0
        for aug_name in aug:
            # aug : random rotation
            if aug_name == 'rot':
                rot = random.randint(0,3)
            # aug : random flip
            elif aug_name == 'hflip':
                hflip = random.randint(0,1)
            # random RGB channel shuffle
            elif aug_name == 'RGB_shuf':
                color_ch = [0,1,2]
                random.shuffle(color_ch)
            # random RB channdel shuffle
            elif aug_name == 'RB_shuf':
                color_ch = [random.choice([0,2])]
                color_ch.append(1)
                color_ch.append(2-color_ch[0])
            else:
                raise RuntimeError('undefined augmentation option : %s'%aug_name)
        
        # for every data(only image), apply rotation and flipping augmentation.
        for key in data:
            if self._is_image_tensor(data[key]):
                # random rotation and flip
                if rot != 0 or hflip != 0:
                    data[key] = rot_hflip_img(data[key], rot, hflip)

                # random color channel shuffle
                if 'color_ch' in locals():
                    data[key] = data[key][color_ch]
            
        return data

    def _need_to_be_mem(self):
        if 'keep_on_mem' in self.kwargs:
            return self.kwargs['keep_on_mem']
        return False

    def _keep_on_mem(self):
        self.pre_loaded = []
        for idx in range(len(self.img_paths)):
            data = self._load_data(idx)

            # make synthesized noisy image when pre-load image
            if self.add_noise_type is not None:
                if 'clean' in data:
                    data['syn_noisy'] = self._add_noise(data['clean'], self.add_noise_type, self.add_noise_opt)

            self.pre_loaded.append(data)

    #----------------------------#
    #   Image saving functions   #
    #----------------------------#
    def save_all_image(self, dir, clean=False, syn_noisy=False, real_noisy=False):
        for idx in range(len(self.img_paths)):
            data = self.__getitem__(idx)

            if clean and 'clean' in data:
                cv2.imwrite(os.path.join(dir, '%04d_CL.png'%idx), tensor2np(data['clean']))
            if syn_noisy and 'syn_noisy' in data:
                cv2.imwrite(os.path.join(dir, '%04d_SN.png'%idx), tensor2np(data['syn_noisy']))
            if real_noisy and 'real_noisy' in data:
                cv2.imwrite(os.path.join(dir, '%04d_RN.png'%idx), tensor2np(data['real_noisy']))

            print('image %04d saved!'%idx)

    def prep_save(self, img_size:int, overlap:int, clean=False, syn_noisy=False, real_noisy=False):
        d_name = '%s_s%d_o%d'%(self.__class__.__name__, img_size, overlap)
        os.makedirs(os.path.join(self.dataset_dir, 'prep', d_name), exist_ok=True)

        assert overlap < img_size
        overlap = img_size - overlap

        if clean:
            clean_dir = os.path.join(self.dataset_dir, 'prep', d_name, 'CL')
            os.makedirs(clean_dir, exist_ok=True)
        if syn_noisy: 
            syn_noisy_dir = os.path.join(self.dataset_dir, 'prep', d_name, 'SN')
            os.makedirs(syn_noisy_dir, exist_ok=True)
        if real_noisy:
            real_noisy_dir = os.path.join(self.dataset_dir, 'prep', d_name, 'RN')
            os.makedirs(real_noisy_dir, exist_ok=True)

        for img_idx in range(self.__len__()):
            data = self.__getitem__(img_idx)

            c,w,h = data['clean'].shape if 'clean' in data else data['real_noisy'].shape
            for w_idx in range((w-img_size)//overlap):
                for h_idx in range((h-img_size)//overlap):
                    wl, wr = w_idx*overlap, w_idx*overlap+img_size
                    hl, hr = h_idx*overlap, h_idx*overlap+img_size
                    if clean:      cv2.imwrite(os.path.join(clean_dir,      '%d_%d_%d.png'%(img_idx, w_idx, h_idx)), tensor2np(data['clean'][:,wl:wr,hl:hr])) 
                    if syn_noisy:  cv2.imwrite(os.path.join(syn_noisy_dir,  '%d_%d_%d.png'%(img_idx, w_idx, h_idx)), tensor2np(data['syn_noisy'][:,wl:wr,hl:hr])) 
                    if real_noisy: cv2.imwrite(os.path.join(real_noisy_dir, '%d_%d_%d.png'%(img_idx, w_idx, h_idx)), tensor2np(data['real_noisy'][:,wl:wr,hl:hr])) 

            print('img%d'%img_idx)
        return

    #----------------------------#
    #            etc             #
    #----------------------------#
    def _is_image_tensor(self, x):
        '''
        return input tensor has image shape. (include batched image)
        '''
        if isinstance(x, torch.Tensor):
            if len(x.shape) == 3 or len(x.shape) == 4:
                if x.dtype != torch.bool:
                    return True
        return False


