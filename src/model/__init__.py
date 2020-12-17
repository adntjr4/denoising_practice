
from importlib import import_module

model_module_dict = {
                'DnCNN_B':  'DnCNN',
                'CDnCNN_B': 'DnCNN',
                }

def get_model_object(model_name):
    model_module = import_module('src.model.{}'.format(model_module_dict[model_name]))
    return getattr(model_module, model_name)