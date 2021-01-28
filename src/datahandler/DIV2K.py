import os

from src.datahandler.denoise_dataset import DenoiseDataSet


class DIV2K_train(DenoiseDataSet):
    def __init__(self, add_noise=None, mask=None, crop_size=None, aug=None, n_repeat=1, **kwargs):
        super().__init__(add_noise=add_noise, mask=mask, crop_size=crop_size, aug=aug, n_repeat=n_repeat, **kwargs)

    def _scan(self):
        self.dataset_path = os.path.join(self.dataset_dir, 'DIV2K/DIV2K_train_HR')

        for root, _, files in os.walk(self.dataset_path):
            for file_name in files:
                self.img_paths.append(os.path.join(root, file_name))

    def _load_data(self, data_idx):
        clean_img = self._load_img(self.img_paths[data_idx])
        return {'clean': clean_img}

class DIV2K_val(DenoiseDataSet):
    def __init__(self, add_noise=None, mask=None, crop_size=None, aug=None, n_repeat=1, **kwargs):
        super().__init__(add_noise=add_noise, mask=mask, crop_size=crop_size, aug=aug, n_repeat=n_repeat, **kwargs)

    def _scan(self):
        self.dataset_path = os.path.join(self.dataset_dir, 'DIV2K/DIV2K_valid_HR')

        for root, _, files in os.walk(self.dataset_path):
            for file_name in files:
                self.img_paths.append(os.path.join(root, file_name))

    def _load_data(self, data_idx):
        clean_img = self._load_img(self.img_paths[data_idx])
        return {'clean': clean_img}
