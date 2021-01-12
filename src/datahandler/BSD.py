import os

from src.datahandler.denoise_dataset import DenoiseDataSet


class BSD68(DenoiseDataSet):
    def __init__(self, add_noise=None, mask=None, crop_size=None, aug=None, n_repeat=1):
        super().__init__(add_noise=add_noise, mask=mask, crop_size=crop_size, aug=aug, n_repeat=n_repeat)

    def _scan(self):
        self.img_paths = []
        for root, _, files in os.walk(os.path.join(self.dataset_dir, 'BSD68')):
            for file_name in files:
                self.img_paths.append(os.path.join(root, file_name))

    def _load_data(self, data_idx):
        img_name = self.img_paths[data_idx]
        img = self._load_img(img_name)
        return {'clean': img}

class BSD400(DenoiseDataSet):
    def __init__(self, add_noise=None, mask=None, crop_size=None, aug=None, n_repeat=1):
        super().__init__(add_noise=add_noise, mask=mask, crop_size=crop_size, aug=aug, n_repeat=n_repeat)

    def _scan(self):
        self.img_paths = []
        for root, _, files in os.walk(os.path.join(self.dataset_dir, 'BSD400')):
            for file_name in files:
                self.img_paths.append(os.path.join(root, file_name))

    def _load_data(self, data_idx):
        img_name = self.img_paths[data_idx]
        img = self._load_img(img_name)
        return {'clean': img}

class CBSD68(DenoiseDataSet):
    def __init__(self, add_noise=None, mask=None, crop_size=None, aug=None, n_repeat=1):
        super().__init__(add_noise=add_noise, mask=mask, crop_size=crop_size, aug=aug, n_repeat=n_repeat)

    def _scan(self):
        self.img_paths = []
        for root, _, files in os.walk(os.path.join(self.dataset_dir, 'CBSD68')):
            for file_name in files:
                self.img_paths.append(os.path.join(root, file_name))

    def _load_data(self, data_idx):
        img_name = self.img_paths[data_idx]
        img = self._load_img(img_name)
        return {'clean': img}

class CBSD432(DenoiseDataSet):
    def __init__(self, add_noise=None, mask=None, crop_size=None, aug=None, n_repeat=1):
        super().__init__(add_noise=add_noise, mask=mask, crop_size=crop_size, aug=aug, n_repeat=n_repeat)

    def _scan(self):
        self.img_paths = []
        for root, _, files in os.walk(os.path.join(self.dataset_dir, 'CBSD432')):
            for file_name in files:
                self.img_paths.append(os.path.join(root, file_name))

    def _load_data(self, data_idx):
        img_name = self.img_paths[data_idx]
        img = self._load_img(img_name)
        return {'clean': img}
