import os

from src.datahandler.denoise_dataset import DenoiseDataSet
from . import regist_dataset


@regist_dataset
class BSD68(DenoiseDataSet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _scan(self):
        for root, _, files in os.walk(os.path.join(self.dataset_dir, 'CBSD68')):
            for file_name in files:
                self.img_paths.append(os.path.join(root, file_name))

    def _load_data(self, data_idx):
        img_name = self.img_paths[data_idx]
        img = self._load_img(img_name, gray=True)
        return {'clean': img}

@regist_dataset
class BSD432(DenoiseDataSet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _scan(self):
        for root, _, files in os.walk(os.path.join(self.dataset_dir, 'CBSD432')):
            for file_name in files:
                self.img_paths.append(os.path.join(root, file_name))

    def _load_data(self, data_idx):
        img_name = self.img_paths[data_idx]
        img = self._load_img(img_name, gray=True)
        return {'clean': img}

@regist_dataset
class CBSD68(DenoiseDataSet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _scan(self):
        for root, _, files in os.walk(os.path.join(self.dataset_dir, 'CBSD68')):
            for file_name in files:
                self.img_paths.append(os.path.join(root, file_name))

    def _load_data(self, data_idx):
        img_name = self.img_paths[data_idx]
        img = self._load_img(img_name)
        return {'clean': img}

@regist_dataset
class CBSD432(DenoiseDataSet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _scan(self):
        for root, _, files in os.walk(os.path.join(self.dataset_dir, 'CBSD432')):
            for file_name in files:
                self.img_paths.append(os.path.join(root, file_name))

    def _load_data(self, data_idx):
        img_name = self.img_paths[data_idx]
        img = self._load_img(img_name)
        return {'clean': img}

@regist_dataset
class Synthesized_BSD(DenoiseDataSet):
    def __init__(self, folder_name, **kwargs):
        self.folder_name = folder_name
        super().__init__(**kwargs)

    def _scan(self):
        for root, _, files in os.walk(os.path.join(self.dataset_dir, 'synthesized_BSD', self.folder_name)):
            for file_name in files:
                self.img_paths.append(os.path.join(root, file_name))

    def _load_data(self, data_idx):
        img_name = self.img_paths[data_idx]
        img = self._load_img(img_name)
        return {'real_noisy': img}

@regist_dataset
class Synthesized_BSD68_15(Synthesized_BSD):
    def __init__(self, **kwargs):
        super().__init__(folder_name='BSD68_15', **kwargs)

@regist_dataset
class Synthesized_BSD68_25(Synthesized_BSD):
    def __init__(self, **kwargs):
        super().__init__(folder_name='BSD68_25', **kwargs)

@regist_dataset
class Synthesized_BSD68_50(Synthesized_BSD):
    def __init__(self, **kwargs):
        super().__init__(folder_name='BSD68_50', **kwargs)

@regist_dataset
class Synthesized_BSD432_15(Synthesized_BSD):
    def __init__(self, **kwargs):
        super().__init__(folder_name='BSD432_15', **kwargs)

@regist_dataset
class Synthesized_BSD432_25(Synthesized_BSD):
    def __init__(self, **kwargs):
        super().__init__(folder_name='BSD432_25', **kwargs)

@regist_dataset
class Synthesized_BSD432_50(Synthesized_BSD):
    def __init__(self, **kwargs):
        super().__init__(folder_name='BSD432_50', **kwargs)

@regist_dataset
class Synthesized_CBSD68_15(Synthesized_BSD):
    def __init__(self, **kwargs):
        super().__init__(folder_name='CBSD68_15', **kwargs)

@regist_dataset
class Synthesized_CBSD68_25(Synthesized_BSD):
    def __init__(self, **kwargs):
        super().__init__(folder_name='CBSD68_25', **kwargs)

@regist_dataset
class Synthesized_CBSD68_50(Synthesized_BSD):
    def __init__(self, **kwargs):
        super().__init__(folder_name='CBSD68_50', **kwargs)

@regist_dataset
class Synthesized_CBSD432_15(Synthesized_BSD):
    def __init__(self, **kwargs):
        super().__init__(folder_name='CBSD432_15', **kwargs)

@regist_dataset
class Synthesized_CBSD432_25(Synthesized_BSD):
    def __init__(self, **kwargs):
        super().__init__(folder_name='CBSD432_25', **kwargs)

@regist_dataset
class Synthesized_CBSD432_50(Synthesized_BSD):
    def __init__(self, **kwargs):
        super().__init__(folder_name='CBSD432_50', **kwargs)

@regist_dataset
class Synthesized_BSD432_25_struc(Synthesized_BSD):
    def __init__(self, **kwargs):
        super().__init__(folder_name='BSD432_25_s', **kwargs)
