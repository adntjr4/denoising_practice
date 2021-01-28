import os

from src.datahandler.denoise_dataset import DenoiseDataSet


class RNI15(DenoiseDataSet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _scan(self):
        for root, _, files in os.walk(os.path.join(self.dataset_dir, 'RNI15')):
            for file_name in files:
                self.img_paths.append(os.path.join(root, file_name))

    def _load_data(self, data_idx):
        img_name = self.img_paths[data_idx]
        img = self._load_img(img_name)
        return {'real_noisy': img}
        