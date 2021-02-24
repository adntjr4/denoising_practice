import argparse, os
from importlib import import_module

import torch

from src.util.config_parse import ConfigParser
import src.trainer.trainer as Trainer


def main():
    # parsing configuration
    args = argparse.ArgumentParser()
    args.add_argument('--session_name', default=None,  type=str)
    args.add_argument('--config',  default=None,  type=str)
    args.add_argument('--resume',  default=False, type=lambda x:x.lower()!='false')
    args.add_argument('--gpu',     default=None,  type=str)
    args.add_argument('--thread',  default=4,     type=int)

    args = args.parse_args()

    assert args.session_name is not None, 'session name is required'
    assert args.config is not None, 'config file path is needed'

    cfg = ConfigParser(args)

    # device setting
    if cfg['gpu'] is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu']

    # intialize trainer
    trainer = getattr(Trainer, cfg['trainer'])(cfg)

    # train
    trainer.train()


if __name__ == '__main__':
    main()
