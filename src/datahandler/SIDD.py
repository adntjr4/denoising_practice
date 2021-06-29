import os, yaml

import scipy.io
import numpy as np

from src.datahandler.denoise_dataset import DenoiseDataSet
from . import regist_dataset


@regist_dataset
class SIDD(DenoiseDataSet):
    '''
    SIDD datatset class using original images.
    '''
    def __init__(self, add_noise=None, mask=None, crop_size=None, aug=None, n_repeat=1, **kwargs):
        super().__init__(add_noise=add_noise, mask=mask, crop_size=crop_size, aug=aug, n_repeat=n_repeat, **kwargs)

    def _scan(self):
        self.dataset_path = os.path.join(self.dataset_dir, 'SIDD/SIDD_Medium_Srgb/Data')

        # scan all image path & info in dataset path
        for folder_name in os.listdir(self.dataset_path):
            # parse folder name of each shot
            parsed_name = self._parse_folder_name(folder_name)

            # add path & information of image 0
            info0 = {}
            info0['instances'] = parsed_name
            info0['clean_img_path'] = os.path.join(self.dataset_path, folder_name, '%s_GT_SRGB_010.PNG'%parsed_name['scene_instance_number'])
            info0['noisy_img_path'] = os.path.join(self.dataset_path, folder_name, '%s_NOISY_SRGB_010.PNG'%parsed_name['scene_instance_number'])
            self.img_paths.append(info0)

            # add path & information of image 1
            info1 = {}
            info1['instances'] = parsed_name
            info1['clean_img_path'] = os.path.join(self.dataset_path, folder_name, '%s_GT_SRGB_011.PNG'%parsed_name['scene_instance_number'])
            info1['noisy_img_path'] = os.path.join(self.dataset_path, folder_name, '%s_NOISY_SRGB_011.PNG'%parsed_name['scene_instance_number'])
            self.img_paths.append(info1)

    def _load_data(self, data_idx):
        info = self.img_paths[data_idx]

        clean_img = self._load_img(info['clean_img_path'])
        noisy_img = self._load_img(info['noisy_img_path'])

        return {'clean': clean_img, 'real_noisy': noisy_img, 'instances': info['instances'] }

    def _parse_folder_name(self, name):
        parsed = {}
        splited = name.split('_')
        parsed['scene_instance_number']      = splited[0]
        parsed['scene_number']               = splited[1]
        parsed['smartphone_camera_code']     = splited[2]
        parsed['ISO_speed']                  = splited[3]
        parsed['shutter_speed']              = splited[4]
        parsed['illuminant_temperature']     = splited[5]
        parsed['illuminant_brightness_code'] = splited[6]
        return parsed

@regist_dataset
class prep_SIDD(DenoiseDataSet):
    '''
    SIDD dataset class for using prepared SIDD data from GeunWung.
    here we use prep-SIDD_Medium_sRGB-cut512-ov128
    '''
    def __init__(self, add_noise=None, mask=None, crop_size=None, aug=None, n_repeat=1, **kwargs):
        super().__init__(add_noise=add_noise, mask=mask, crop_size=crop_size, aug=aug, n_repeat=n_repeat, **kwargs)

    def _scan(self):
        self.dataset_path = os.path.join(self.dataset_dir, 'prep/prep-SIDD_Medium_sRGB-cut512-ov128')

        for root, _, files in os.walk(os.path.join(self.dataset_path, 'GT')):
            self.img_paths = files

    def _load_data(self, data_idx):
        file_name = self.img_paths[data_idx]

        instance  = self._get_instance(file_name.replace('.png', ''))
        clean_img = self._load_img(os.path.join(self.dataset_path, 'GT', file_name))
        noisy_img = self._load_img(os.path.join(self.dataset_path, 'N' , file_name))

        return {'clean': clean_img, 'real_noisy': noisy_img,} #'instances': instance }

    def _get_instance(self, name):
        with open(os.path.join(self.dataset_path, 'info_GT', name+'.yml')) as f:
            instance = yaml.load(f, Loader=yaml.FullLoader)
        return instance

@regist_dataset
class part_SIDD(DenoiseDataSet):
    '''
    part of SIDD dataset class for using prepared SIDD data from GeunWung.
    here we use part_SIDD_Medium_sRGB
    '''
    def __init__(self, add_noise=None, mask=None, crop_size=None, aug=None, n_repeat=1, **kwargs):
        super().__init__(add_noise=add_noise, mask=mask, crop_size=crop_size, aug=aug, n_repeat=n_repeat, **kwargs)

    def _scan(self):
        self.dataset_path = os.path.join(self.dataset_dir, 'part_SIDD_Medium_sRGB')

        for root, _, files in os.walk(os.path.join(self.dataset_path, 'GT')):
            self.img_paths = files

    def _load_data(self, data_idx):
        file_name = self.img_paths[data_idx]

        instance  = self._get_instance(file_name.replace('.png', ''))
        clean_img = self._load_img(os.path.join(self.dataset_path, 'GT', file_name))
        noisy_img = self._load_img(os.path.join(self.dataset_path, 'N' , file_name))

        return {'clean': clean_img, 'real_noisy': noisy_img,} #'instances': instance }

    def _get_instance(self, name):
        with open(os.path.join(self.dataset_path, 'info_GT', name+'.yml')) as f:
            instance = yaml.load(f, Loader=yaml.FullLoader)
        return instance

@regist_dataset
class SIDD_val(DenoiseDataSet):
    '''
    SIDD validation dataset class 
    '''
    def __init__(self, add_noise=None, mask=None, crop_size=None, aug=None, n_repeat=1, **kwargs):
        super().__init__(add_noise=add_noise, mask=mask, crop_size=crop_size, aug=aug, n_repeat=n_repeat, **kwargs)

    def _scan(self):
        self.dataset_path = os.path.join(self.dataset_dir, 'SIDD')

        clean_mat_file_path = os.path.join(self.dataset_path, 'ValidationGtBlocksSrgb.mat')
        noisy_mat_file_path = os.path.join(self.dataset_path, 'ValidationNoisyBlocksSrgb.mat')

        self.clean_patches = np.array(scipy.io.loadmat(clean_mat_file_path, appendmat=False)['ValidationGtBlocksSrgb'])
        self.noisy_patches = np.array(scipy.io.loadmat(noisy_mat_file_path, appendmat=False)['ValidationNoisyBlocksSrgb'])

        # for __len__(), make img_paths have same length
        # number of all possible patch is 1280
        for _ in range(1280):
            self.img_paths.append(None)

    def _load_data(self, data_idx):
        img_id   = data_idx // 32
        patch_id = data_idx  % 32

        clean_img = self.clean_patches[img_id, patch_id, :].astype(float)
        noisy_img = self.noisy_patches[img_id, patch_id, :].astype(float)

        clean_img = self._load_img_from_np(clean_img)
        noisy_img = self._load_img_from_np(noisy_img)

        return {'clean': clean_img, 'real_noisy': noisy_img }

@regist_dataset
class SIDD_benchmark(DenoiseDataSet):
    '''
    SIDD benchmark dataset class
    '''
    def __init__(self, add_noise=None, mask=None, crop_size=None, aug=None, n_repeat=1, **kwargs):
        super().__init__(add_noise=add_noise, mask=mask, crop_size=crop_size, aug=aug, n_repeat=n_repeat, **kwargs)

    def _scan(self):
        self.dataset_path = os.path.join(self.dataset_dir, 'SIDD')

        mat_file_path = os.path.join(self.dataset_path, 'BenchmarkNoisyBlocksSrgb.mat')

        self.noisy_patches = np.array(scipy.io.loadmat(mat_file_path, appendmat=False)['BenchmarkNoisyBlocksSrgb'])

        # for __len__(), make img_paths have same length
        # number of all possible patch is 1280
        for _ in range(1280):
            self.img_paths.append(None)

    def _load_data(self, data_idx):
        img_id   = data_idx // 32
        patch_id = data_idx  % 32

        noisy_img = self.noisy_patches[img_id, patch_id, :].astype(float)

        noisy_img = self._load_img_from_np(noisy_img)

        return {'real_noisy': noisy_img}
