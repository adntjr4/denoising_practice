import argparse, os

import torch

from src.util.config_parse import ConfigParser
from src.trainer.trainer import Trainer


def main():
    # parsing configuration
    args = argparse.ArgumentParser()
    args.add_argument('--session_name', default=None,  type=str)
    args.add_argument('--config',       default=None,  type=str)
    args.add_argument('--ckpt_epoch',   default=None,  type=int)
    args.add_argument('--gpu',          default=None,  type=str)
    args.add_argument('--thread',       default=4,     type=int)

    args = args.parse_args()

    assert args.session_name is not None, 'session name is required'
    assert args.config is not None, 'config file path is needed'
    assert args.ckpt_epoch is not None, 'checkpoint epoch is needed'

    cfg = ConfigParser(args)

    # device setting
    if cfg['gpu'] is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu']

    # intialize trainer
    trainer = Trainer(cfg)

    # test
    trainer.test()


if __name__ == '__main__':
    main()
