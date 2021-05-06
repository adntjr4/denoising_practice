
from importlib import import_module

model_module_dict = {
                # DnCNN
                'DnCNN_B':  'DnCNN',

                # N2V
                'N2V_UNet': 'UNet',

                # Laine19
                'Laine19': 'Laine19',

                # D-BSN
                'DBSN': 'DBSN',
                'DBSN_Likelihood': 'DBSN',
                'DBSN_Likelihood3': 'DBSN',

                # Effective Blind-Spot Network
                'EBSN' : 'EBSN',

                # CLtoN
                'CLtoN_G': 'CLtoN',
                'CLtoN_D': 'CLtoN',
                'LtoN_G': 'CLtoN',
                'LtoN_D': 'CLtoN',
                'CLtoN_G_indep_1': 'CLtoN',
                'CLtoN_G_indep_3': 'CLtoN',
                'CLtoN_G_indep_13': 'CLtoN',
                'CLtoN_G_dep_1': 'CLtoN',
                'CLtoN_G_dep_3': 'CLtoN',
                'CLtoN_G_dep_13': 'CLtoN',
                'CLtoN_G_indep_dep_1': 'CLtoN',
                'CLtoN_G_indep_dep_3': 'CLtoN',

                'CLtoN_D_one_out': 'CLtoN',
                'LtoN_D_one_out': 'CLtoN',
                }

def get_model_object(model_name):
    model_module = import_module('src.model.{}'.format(model_module_dict[model_name]))
    return getattr(model_module, model_name)