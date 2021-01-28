import os

from src.datahandler.denoise_dataset import DenoiseDataSet


class SIDD(DenoiseDataSet):
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

class SIDD_val(DenoiseDataSet):
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