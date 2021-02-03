import os, math
from importlib import import_module

import cv2
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from .output import Output
from ..loss.loss import Loss
from ..datahandler.denoise_dataset import get_dataset_object
from ..util.logger import Logger

status_len = 13


class BasicTrainer(Output):
    def test(self):
        raise NotImplementedError('define this function for each trainer')
    def validation(self):
        raise NotImplementedError('define this function for each trainer')
    def _set_module(self):
        raise NotImplementedError('define this function for each trainer')
    def _set_optimizer(self):
        raise NotImplementedError('define this function for each trainer')
    def _step(self, data):
        raise NotImplementedError('define this function for each trainer')
    '''
    def add_data_set(self):
        # use this function to add more data set.
    '''

    # =================================================================== #

    def __init__(self, cfg):
        self.session_name = cfg['session_name']

        self.checkpoint_folder = 'checkpoint'

        dir_list = ['checkpoint', 'img', 'tboard']
        super().__init__(self.session_name, dir_list)
        
        self.cfg = cfg
        self.train_cfg = cfg['training']
        self.val_cfg   = cfg['validation']
        self.ckpt_cfg  = cfg['checkpoint']

    def train(self):
        # initializing
        self._before_train()

        # warmup
        if self.epoch == 1 and self.train_cfg['warmup']:
            self._warmup()

        # training
        for self.epoch in range(self.epoch, self.max_epoch+1):
            self._before_epoch()
            self._run_epoch()
            self._after_epoch()
        
        self._after_train()

    def _warmup(self):
        self.status = 'warmup'.ljust(status_len) #.center(status_len)

        self.train_data_loader_iter = iter(self.train_data_loader)
        warmup_iter = self.train_cfg['warmup_iter']
        if warmup_iter > self.train_data_loader.__len__():
            self.logger.info('currently warmup support 1 epoch as maximum. warmup iter is replaced to 1 epoch iteration. %d -> %d' \
                % (warmup_iter, self.train_data_loader.__len__()))
            warmup_iter = self.train_data_loader.__len__()

        for self.iter in range(1, warmup_iter+1):
            self._adjust_warmup_lr(warmup_iter)
            self._before_step()
            self._run_step()
            self._after_step()

    def _before_train(self):
        # initialing
        self.module = self._set_module()

        # training dataset loader
        train_other_args = self._set_other_args(self.train_cfg)
        self.train_data_set = get_dataset_object(self.train_cfg['dataset'])(crop_size = self.train_cfg['crop_size'], 
                                                                            add_noise = self.train_cfg['add_noise'], 
                                                                            mask      = self.train_cfg['mask'],
                                                                            aug       = self.train_cfg['aug'],
                                                                            norm      = self.train_cfg['normalization'],
                                                                            n_repeat  = self.train_cfg['n_repeat'],
                                                                            **train_other_args,)
        self.train_data_loader = DataLoader(dataset=self.train_data_set, batch_size=self.train_cfg['batch_size'], shuffle=True, num_workers=self.cfg['thread'])

        if hasattr(self, 'add_data_set'):
            # custom function for adding more data set.
            self.add_data_set()

        # validation dataset loader
        if self.val_cfg['val']:
            val_other_args = self._set_other_args(self.val_cfg)
            self.val_data_set = get_dataset_object(self.val_cfg['dataset'])(crop_size = self.val_cfg['crop_size'], 
                                                                            add_noise = self.val_cfg['add_noise'],
                                                                            mask      = self.val_cfg['mask'],
                                                                            norm      = self.val_cfg['normalization'],
                                                                            **val_other_args,)
            self.val_data_loader   = DataLoader(dataset=self.val_data_set, batch_size=1, shuffle=False, num_workers=self.cfg['thread'])

        # other configuration
        self.max_epoch = self.train_cfg['max_epoch']
        self.epoch = self.start_epoch = 1
        self.max_iter = math.ceil(self.train_data_loader.dataset.__len__() / self.train_cfg['batch_size'])

        self.loss = Loss(self.train_cfg['loss'])
        self.loss_dict = self.loss.get_loss_dict_form()
        self.loss_log = []

        # set optimizer
        self.optimizer = self._set_optimizer()

        # resume
        if self.cfg["resume"]:
            # find last checkpoint
            load_epoch = self._find_last_epoch()

            # load last checkpoint
            self.load_checkpoint(load_epoch)
            self.epoch = load_epoch+1

            # logger initialization
            self.logger = Logger((self.max_epoch, self.max_iter), log_dir=self.get_dir(''), log_file_option='a')
        else:
            # logger initialization
            self.logger = Logger((self.max_epoch, self.max_iter), log_dir=self.get_dir(''), log_file_option='w')

        # cuda
        if self.cfg['gpu'] != 'None':
            # model to GPU
            self.model = {key: nn.DataParallel(self.module[key]).cuda() for key in self.module}
            # optimizer to GPU
            for optim in self.optimizer.values():
                for state in optim.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
            # loss module to GPU
            self.loss = self.loss.cuda()
        else:
            self.model = {key: nn.DataParallel(self.module[key]) for key in self.module}

        # start message
        self.logger.start((self.epoch-1, 0))
        self.logger.highlight(self.logger.get_start_msg())

    def _after_train(self):
        # finish message
        self.logger.highlight(self.logger.get_finish_msg())

    def _before_epoch(self):
        self.status = ('epoch %03d/%03d'%(self.epoch, self.max_epoch)).center(status_len)

        # dataloader iter
        self.train_data_loader_iter = iter(self.train_data_loader)

        # model training mode
        for m in self.model.values():
            m.train() 

        
    def _run_epoch(self):
        for self.iter in range(1, self.max_iter+1):
            self._before_step()
            self._run_step()
            self._after_step()

    def _after_epoch(self):
        # scheduler step
        self._adjust_lr()

        # save checkpoint
        if self.epoch >= self.ckpt_cfg['start_epoch']:
            if (self.epoch-self.ckpt_cfg['start_epoch'])%self.ckpt_cfg['interval_epoch'] == 0:
                self.save_checkpoint()

        # validation
        if self.val_cfg['val']:
            if self.epoch >= self.val_cfg['start_epoch'] and self.val_cfg['val']:
                if (self.epoch-self.val_cfg['start_epoch']) % self.val_cfg['interval_epoch'] == 0:
                    self.validation()

    def _before_step(self):
        pass

    def _run_step(self):
        # get data (data should be dictionary of Tensors)
        data = next(self.train_data_loader_iter)

        # to device
        if self.cfg['gpu'] != 'None':
            for key in data:
                data[key] = data[key].cuda()

        # step (forward, cal losses, backward)
        losses = self._step(data) # forward, losses, backward

        # save losses
        for key in losses:
            if key != 'count': self.loss_dict[key] += float(losses[key])
        self.loss_dict['count'] += 1

    def _after_step(self):
        # print loss
        if (self.iter%self.cfg['log']['interval_iter']==0 and self.iter!=0) or (self.iter == self.max_iter):
            self.print_loss()

        # print progress
        self.logger.print_prog_msg((self.epoch-1, self.iter-1))

    # ====================================================================== #

    def print_loss(self):
        temperal_loss = 0.
        for key in self.loss_dict:
            if key != 'count':
                    temperal_loss += self.loss_dict[key]/self.loss_dict['count']
        self.loss_log += [temperal_loss]

        loss_out_str = '[%s] %04d/%04d, lr:%s | '%(self.status, self.iter, self.max_iter, "{:.2e}".format(self._get_current_lr()))

        loss_out_str += 'avg_loss : %.4f | '%(np.mean(self.loss_log))

        for key in self.loss_dict:
            if key != 'count':
                loss_out_str += '%s : %.4f | '%(key, self.loss_dict[key]/self.loss_dict['count'])
                self.loss_dict[key] = 0.
        self.loss_dict['count'] = 0
        self.logger.info(loss_out_str)

    def save_checkpoint(self):
        checkpoint_name = self._checkpoint_name(self.epoch)
        torch.save({'epoch': self.epoch,
                    'model_weight': {key:self.model[key].module.state_dict() for key in self.model},
                    'optimizer_weight': {key:self.optimizer[key].state_dict() for key in self.optimizer}},
                    os.path.join(self.get_dir(self.checkpoint_folder), checkpoint_name))

    def load_checkpoint(self, load_epoch): 
        file_name = os.path.join(self.get_dir(self.checkpoint_folder), self._checkpoint_name(load_epoch))
        assert os.path.isfile(file_name), 'there is no checkpoint: %s'%file_name
        saved_checkpoint = torch.load(file_name)
        self.epoch = saved_checkpoint['epoch']
        for key in self.module:
            self.module[key].load_state_dict(saved_checkpoint['model_weight'][key])
        for key in self.optimizer:
            self.optimizer[key].load_state_dict(saved_checkpoint['optimizer_weight'][key])

    def _checkpoint_name(self, epoch):
        return self.session_name + '_%03d'%epoch + '.pth'

    def _find_last_epoch(self):
        checkpoint_list = os.listdir(self.get_dir(self.checkpoint_folder))
        epochs = [int(ckpt.replace('%s_'%self.session_name, '').replace('.pth', '')) for ckpt in checkpoint_list]
        assert len(epochs) > 0, 'There is no resumable checkpoint on session %s.'%self.session_name
        return max(epochs)

    def _get_current_lr(self):
        for first_optim in self.optimizer.values():
            for param_group in first_optim.param_groups:
                return param_group['lr']

    def _set_one_optimizer(self, parameters):
        opt = self.train_cfg['optimizer']
        lr = float(self.train_cfg['init_lr'])

        if opt['type'] == 'SGD':
            return optim.SGD(parameters, lr=lr, momentum=float(opt['SGD']['momentum']), weight_decay=float(opt['SGD']['weight_decay']))
        elif opt['type'] == 'Adam':
            return optim.Adam(parameters, lr=lr, betas=opt['Adam']['betas'])
        else:
            raise RuntimeError('ambiguious optimizer type: {}'.format(opt['type']))

    def _adjust_lr(self):
        sched = self.train_cfg['scheduler']

        if sched['type'] == 'step':
            if self.epoch % sched['step']['step_size'] == 0:
                for optimizer in self.optimizer.values():
                    lr_before = optimizer.param_groups[0]['lr']
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr_before * float(sched['step']['gamma'])
                lr_notice = '[%s] updated lr:%s | '%(self.status, "{:.2e}".format(self._get_current_lr()))
                self.logger.info(lr_notice)
        else:
            raise RuntimeError('ambiguious scheduler type: {}'.format(sched['type']))

    def _adjust_warmup_lr(self, warmup_iter):
        init_lr = float(self.train_cfg['init_lr'])
        warmup_lr = init_lr * self.iter / warmup_iter

        for optimizer in self.optimizer.values():
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr

    def _set_other_args(self, cfg):
        other_args = {}
        if 'power_cliping' in cfg:
            other_args['power_cliping'] = cfg['power_cliping']
        if 'keep_on_mem' in cfg:
            other_args['keep_on_mem'] = cfg['keep_on_mem']
        return other_args
