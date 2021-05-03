import os

from src.datahandler.denoise_dataset import DenoiseDataSet


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

class Synthesized_BSD68_15(Synthesized_BSD):
    def __init__(self, **kwargs):
        super().__init__(folder_name='BSD68_15', **kwargs)

class Synthesized_BSD68_25(Synthesized_BSD):
    def __init__(self, **kwargs):
        super().__init__(folder_name='BSD68_25', **kwargs)

class Synthesized_BSD68_50(Synthesized_BSD):
    def __init__(self, **kwargs):
        super().__init__(folder_name='BSD68_50', **kwargs)

class Synthesized_BSD432_15(Synthesized_BSD):
    def __init__(self, **kwargs):
        super().__init__(folder_name='BSD432_15', **kwargs)

class Synthesized_BSD432_25(Synthesized_BSD):
    def __init__(self, **kwargs):
        super().__init__(folder_name='BSD432_25', **kwargs)

class Synthesized_BSD432_50(Synthesized_BSD):
    def __init__(self, **kwargs):
        super().__init__(folder_name='BSD432_50', **kwargs)

class Synthesized_CBSD68_15(Synthesized_BSD):
    def __init__(self, **kwargs):
        super().__init__(folder_name='CBSD68_15', **kwargs)

class Synthesized_CBSD68_25(Synthesized_BSD):
    def __init__(self, **kwargs):
        super().__init__(folder_name='CBSD68_25', **kwargs)

class Synthesized_CBSD68_50(Synthesized_BSD):
    def __init__(self, **kwargs):
        super().__init__(folder_name='CBSD68_50', **kwargs)

class Synthesized_CBSD432_15(Synthesized_BSD):
    def __init__(self, **kwargs):
        super().__init__(folder_name='CBSD432_15', **kwargs)

class Synthesized_CBSD432_25(Synthesized_BSD):
    def __init__(self, **kwargs):
        super().__init__(folder_name='CBSD432_25', **kwargs)

class Synthesized_CBSD432_50(Synthesized_BSD):
    def __init__(self, **kwargs):
        super().__init__(folder_name='CBSD432_50', **kwargs)

class Synthesized_BSD432_25_struc(Synthesized_BSD):
    def __init__(self, **kwargs):
        super().__init__(folder_name='BSD432_25_s', **kwargs)

class PD_BSD(DenoiseDataSet):
    def __init__(self, pd, ds, nlf, **kwargs):
        self.pd = pd
        self.ds = ds
        self.nlf = nlf
        self.dataset_path = os.path.join('synthesized_PD_BSD', 'pd%d'%self.pd, '%s_%d'%(self.ds, self.nlf))
        super().__init__(**kwargs)
        
    def _scan(self):
        for root, _, files in os.walk(os.path.join(self.dataset_dir, self.dataset_path, 'C')):
            for file_name in files:
                self.img_paths.append(file_name)

    def _load_data(self, data_idx):
        file_name = self.img_paths[data_idx]
        clean_img = self._load_img(os.path.join(self.dataset_dir, self.dataset_path, 'C', file_name))
        noisy_img = self._load_img(os.path.join(self.dataset_dir, self.dataset_path, 'N', file_name.replace('CL', 'SN')))
        return {'clean': clean_img, 'real_noisy': noisy_img}

class PD1_CBSD68_15(PD_BSD):
    def __init__(self, **kwargs):
        super().__init__(pd=1, ds='CBSD68', nlf=15, **kwargs)

class PD1_CBSD68_25(PD_BSD):
    def __init__(self, **kwargs):
        super().__init__(pd=1, ds='CBSD68', nlf=25, **kwargs)

class PD1_CBSD68_50(PD_BSD):
    def __init__(self, **kwargs):
        super().__init__(pd=1, ds='CBSD68', nlf=50, **kwargs)

class PD2_CBSD68_15(PD_BSD):
    def __init__(self, **kwargs):
        super().__init__(pd=2, ds='CBSD68', nlf=15, **kwargs)

class PD2_CBSD68_25(PD_BSD):
    def __init__(self, **kwargs):
        super().__init__(pd=2, ds='CBSD68', nlf=25, **kwargs)

class PD2_CBSD68_50(PD_BSD):
    def __init__(self, **kwargs):
        super().__init__(pd=2, ds='CBSD68', nlf=50, **kwargs)

class PD3_CBSD68_15(PD_BSD):
    def __init__(self, **kwargs):
        super().__init__(pd=3, ds='CBSD68', nlf=15, **kwargs)

class PD3_CBSD68_25(PD_BSD):
    def __init__(self, **kwargs):
        super().__init__(pd=3, ds='CBSD68', nlf=25, **kwargs)

class PD3_CBSD68_50(PD_BSD):
    def __init__(self, **kwargs):
        super().__init__(pd=3, ds='CBSD68', nlf=50, **kwargs)

class PD4_CBSD68_15(PD_BSD):
    def __init__(self, **kwargs):
        super().__init__(pd=4, ds='CBSD68', nlf=15, **kwargs)

class PD4_CBSD68_25(PD_BSD):
    def __init__(self, **kwargs):
        super().__init__(pd=4, ds='CBSD68', nlf=25, **kwargs)

class PD4_CBSD68_50(PD_BSD):
    def __init__(self, **kwargs):
        super().__init__(pd=4, ds='CBSD68', nlf=50, **kwargs)

class PD1_CBSD432_15(PD_BSD):
    def __init__(self, **kwargs):
        super().__init__(pd=1, ds='CBSD432', nlf=15, **kwargs)

class PD1_CBSD432_25(PD_BSD):
    def __init__(self, **kwargs):
        super().__init__(pd=1, ds='CBSD432', nlf=25, **kwargs)

class PD1_CBSD432_50(PD_BSD):
    def __init__(self, **kwargs):
        super().__init__(pd=1, ds='CBSD432', nlf=50, **kwargs)

class PD2_CBSD432_15(PD_BSD):
    def __init__(self, **kwargs):
        super().__init__(pd=2, ds='CBSD432', nlf=15, **kwargs)

class PD2_CBSD432_25(PD_BSD):
    def __init__(self, **kwargs):
        super().__init__(pd=2, ds='CBSD432', nlf=25, **kwargs)

class PD2_CBSD432_50(PD_BSD):
    def __init__(self, **kwargs):
        super().__init__(pd=2, ds='CBSD432', nlf=50, **kwargs)

class PD3_CBSD432_15(PD_BSD):
    def __init__(self, **kwargs):
        super().__init__(pd=3, ds='CBSD432', nlf=15, **kwargs)

class PD3_CBSD432_25(PD_BSD):
    def __init__(self, **kwargs):
        super().__init__(pd=3, ds='CBSD432', nlf=25, **kwargs)

class PD3_CBSD432_50(PD_BSD):
    def __init__(self, **kwargs):
        super().__init__(pd=3, ds='CBSD432', nlf=50, **kwargs)

class PD4_CBSD432_15(PD_BSD):
    def __init__(self, **kwargs):
        super().__init__(pd=4, ds='CBSD432', nlf=15, **kwargs)

class PD4_CBSD432_25(PD_BSD):
    def __init__(self, **kwargs):
        super().__init__(pd=4, ds='CBSD432', nlf=25, **kwargs)

class PD4_CBSD432_50(PD_BSD):
    def __init__(self, **kwargs):
        super().__init__(pd=4, ds='CBSD432', nlf=50, **kwargs)
