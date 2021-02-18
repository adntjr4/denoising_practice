
from importlib import import_module

model_module_dict = {
                # DnCNN
                'DnCNN_B':  'DnCNN',
                'CDnCNN_B': 'DnCNN',

                # N2V
                'N2V_UNet': 'UNet',
                'C_N2V_UNet': 'UNet',

                # D-BSN
                'DBSN': 'DBSN',
                'C_DBSN': 'DBSN',

                # Effective Blind-Spot Network
                'EBSN' : 'EBSN',
                'C_EBSN' : 'EBSN',

                # CLtoN
                'CLtoN_G': 'CLtoN',
                'CLtoN_D': 'CLtoN',

                # Custom
                'RBSN_sep': 'RBSN',
                'C_RBSN_sep': 'RBSN',
                'RBSN_cond': 'RBSN',
                'NarrowDnCNN': 'DnCNN',
                'FBSNet': 'FBSNet',
                'DBSN_Likelihood': 'DBSN',
                'DBSN_Cond': 'DBSN',

                # temporal networks
                'EBSN_Edge': 'RBSN',
                'DBSN_Alt': 'DBSN',
                'EBSN_Wide': 'EBSN',
                'C_EBSN_Wide': 'EBSN',
                }

def get_model_object(model_name):
    model_module = import_module('src.model.{}'.format(model_module_dict[model_name]))
    return getattr(model_module, model_name)