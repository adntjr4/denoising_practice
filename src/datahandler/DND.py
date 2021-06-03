import os

import h5py

from src.datahandler.denoise_dataset import DenoiseDataSet


class DND(DenoiseDataSet):
    def __init__(self, add_noise=None, mask=None, crop_size=None, aug=None, n_repeat=1, **kwargs):
        super().__init__(add_noise=add_noise, mask=mask, crop_size=crop_size, aug=aug, n_repeat=n_repeat, **kwargs)

    def _scan(self):
        self.dataset_path = os.path.join(self.dataset_dir, 'DND/dnd_2017/png_srgb')

        for root, _, files in os.walk(self.dataset_path):
            for file_name in files:
                # TODO parse folder name of each shot
                # parsed_name = self._parse_folder_name(folder_name)
                self.img_paths.append(os.path.join(root, file_name))

    def _load_data(self, data_idx):
        noisy_img = self._load_img(self.img_paths[data_idx])
        return {'real_noisy': noisy_img}

    def _parse_folder_name(self, name):
        raise NotImplementedError
        parsed = {}
        return parsed

class prep_DND(DenoiseDataSet):
    '''
    dataset class for prepared DND dataset which is cropped with overlap.
    '''
    def __init__(self, add_noise=None, mask=None, crop_size=None, aug=None, n_repeat=1, **kwargs):
        super().__init__(add_noise=add_noise, mask=mask, crop_size=crop_size, aug=aug, n_repeat=n_repeat, **kwargs)

    def _scan(self):
        self.dataset_path = os.path.join(self.dataset_dir, 'prep/prep-DND-cut512-ov128')

        for root, _, files in os.walk(os.path.join(self.dataset_path, 'N')):
            self.img_paths = files

    def _load_data(self, data_idx):
        file_name = self.img_paths[data_idx]

        noisy_img = self._load_img(os.path.join(self.dataset_path, 'N' , file_name))

        return {'real_noisy': noisy_img} #'instances': instance }

class DND_benchmark(DenoiseDataSet):
    '''
    dumpy dataset class for DND benchmark
    '''
    def __init__(self, add_noise=None, mask=None, crop_size=None, aug=None, n_repeat=1, **kwargs):
        super().__init__(add_noise=add_noise, mask=mask, crop_size=crop_size, aug=aug, n_repeat=n_repeat, **kwargs)

    def _scan(self):
        pass

    def _load_data(self, data_idx):
        pass