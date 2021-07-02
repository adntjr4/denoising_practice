import os
from importlib import import_module


status_len = 13

trainer_dict = {}

def regist_trainer(trainer):
    trainer_name = trainer.__name__
    assert not trainer_name in trainer_dict, 'there is already registered dataset: %s in trainer_dict.' % trainer_name
    trainer_dict[trainer_name] = trainer

    return trainer

# import all python files in trainer folder
for module in os.listdir(os.path.dirname(__file__)):
    if module == '__init__.py' or module[-3:] != '.py':
        continue
    import_module('src.trainer.{}'.format(module[:-3]))
del module
