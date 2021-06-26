
from importlib import import_module

dataset_module = {
                    # BSD
                    'BSD68'     : 'BSD',
                    'BSD432'    : 'BSD',
                    'CBSD68'    : 'BSD',
                    'CBSD432'   : 'BSD',

                    # DND
                    'DND'       : 'DND',
                    'prep_DND'  : 'DND',
                    'DND_benchmark'  : 'DND',

                    # SIDD
                    'SIDD'      : 'SIDD',
                    'SIDD_val'  : 'SIDD',
                    'prep_SIDD' : 'SIDD',
                    'SIDD_benchmark' : 'SIDD',

                    # RNI15
                    'RNI15'     : 'RNI15',

                    # DIV2K
                    'DIV2K_train' : 'DIV2K',
                    'DIV2K_val'   : 'DIV2K',

                    # prep & part
                    'prep_SIDD' : 'SIDD',
                    'part_SIDD' : 'SIDD',

                    # pre-generated synthetic noisy image
                    'Synthesized_BSD68_15'   : 'BSD',
                    'Synthesized_BSD68_25'   : 'BSD',
                    'Synthesized_BSD68_50'   : 'BSD',
                    'Synthesized_BSD432_15'  : 'BSD',
                    'Synthesized_BSD432_25'  : 'BSD',
                    'Synthesized_BSD432_50'  : 'BSD',
                    'Synthesized_CBSD68_15'  : 'BSD',
                    'Synthesized_CBSD68_25'  : 'BSD',
                    'Synthesized_CBSD68_50'  : 'BSD',
                    'Synthesized_CBSD432_15' : 'BSD',
                    'Synthesized_CBSD432_25' : 'BSD',
                    'Synthesized_CBSD432_50' : 'BSD',

                    'Synthesized_BSD432_25_struc'  : 'BSD',

                    'PD1_CBSD68_15' : 'BSD',
                    'PD1_CBSD68_25' : 'BSD',
                    'PD1_CBSD68_50' : 'BSD',
                    'PD2_CBSD68_15' : 'BSD',
                    'PD2_CBSD68_25' : 'BSD',
                    'PD2_CBSD68_50' : 'BSD',
                    'PD3_CBSD68_15' : 'BSD',
                    'PD3_CBSD68_25' : 'BSD',
                    'PD3_CBSD68_50' : 'BSD',
                    'PD4_CBSD68_15' : 'BSD',
                    'PD4_CBSD68_25' : 'BSD',
                    'PD4_CBSD68_50' : 'BSD',
                    'PD1_CBSD432_15' : 'BSD',
                    'PD1_CBSD432_25' : 'BSD',
                    'PD1_CBSD432_50' : 'BSD',
                    'PD2_CBSD432_15' : 'BSD',
                    'PD2_CBSD432_25' : 'BSD',
                    'PD2_CBSD432_50' : 'BSD',
                    'PD3_CBSD432_15' : 'BSD',
                    'PD3_CBSD432_25' : 'BSD',
                    'PD3_CBSD432_50' : 'BSD',
                    'PD4_CBSD432_15' : 'BSD',
                    'PD4_CBSD432_25' : 'BSD',
                    'PD4_CBSD432_50' : 'BSD',

                }

def get_dataset_object(dataset_name):
    if dataset_name is None:
        return None
    elif len(dataset_name.split('+')) > 1:
        raise NotImplementedError
    else:
        module_dset = import_module('src.datahandler.{}'.format(dataset_module[dataset_name]))
        return getattr(module_dset, dataset_name)
