import os
import math
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from ..loss.loss import Loss
from ..datahandler import get_dataset_object
from ..util.file_manager import FileManager
from ..util.logger import Logger
from ..util.util import human_format, rot_hflip_img


status_len = 13

class BasicTrainer():
    def test(self):
        raise NotImplementedError('define this function for each trainer')
    def validation(self):
        raise NotImplementedError('define this function for each trainer')
    def _set_module(self):
        # return dict form with model name.
        raise NotImplementedError('define this function for each trainer')
    def _set_optimizer(self):
        # return dict form with each coresponding model name.
        raise NotImplementedError('define this function for each trainer')
    def _forward_fn(self, module, loss, data):
        # forward with model, loss function and data.
        # return output of loss function.
        raise NotImplementedError('define this function for each trainer')

    # =================================================================== #

    def __init__(self, cfg):
        self.session_name = cfg['session_name']

        self.checkpoint_folder = 'checkpoint'

        # this handles i/o include printing, file save
        self.file_manager = FileManager(self.session_name)
        self.logger = Logger()
        
        self.cfg = cfg
        self.train_cfg = cfg['training']
        self.val_cfg   = cfg['validation']
        self.test_cfg  = cfg['test']
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

        # make dataloader iterable.
        self.train_dataloader_iter = {}
        for key in self.train_dataloader:
            self.train_dataloader_iter[key] = iter(self.train_dataloader[key])

        warmup_iter = self.train_cfg['warmup_iter']
        if warmup_iter > self.max_iter:
            self.logger.info('currently warmup support 1 epoch as maximum. warmup iter is replaced to 1 epoch iteration. %d -> %d' \
                % (warmup_iter, self.max_iter))
            warmup_iter = self.max_iter

        for self.iter in range(1, warmup_iter+1):
            self._adjust_warmup_lr(warmup_iter)
            self._before_step()
            self._run_step()
            self._after_step()

    def _before_test(self):
        # initialing
        self.module = self._set_module()

        # load checkpoint file
        ckpt_epoch = self._find_last_epoch() if self.cfg['ckpt_epoch'] == -1 else self.cfg['ckpt_epoch']
        self.load_checkpoint(ckpt_epoch)
        self.epoch = self.cfg['ckpt_epoch'] # for print or saving file name.

        # test dataset loader
        self.test_dataloader = self._set_dataloader(self.test_cfg, batch_size=1, shuffle=False, num_workers=self.cfg['thread'])

        # wrapping and device setting
        if self.cfg['gpu'] != 'None':
            # model to GPU
            self.model = {key: nn.DataParallel(self.module[key]).cuda() for key in self.module}
        else:
            self.model = {key: nn.DataParallel(self.module[key]) for key in self.module}

        # start message
        self.logger.highlight(self.logger.get_start_msg())
            
    def _before_train(self):
        # cudnn
        torch.backends.cudnn.benchmark = False

        # initialing
        self.module = self._set_module()

        # training dataset loader
        self.train_dataloader = self._set_dataloader(self.train_cfg, batch_size=self.train_cfg['batch_size'], shuffle=True, num_workers=self.cfg['thread'])

        # validation dataset loader
        if self.val_cfg['val']:
            self.val_dataloader = self._set_dataloader(self.val_cfg, batch_size=1, shuffle=False, num_workers=self.cfg['thread'])

        # other configuration
        self.max_epoch = self.train_cfg['max_epoch']
        self.epoch = self.start_epoch = 1
        max_len = max([self.train_dataloader[key].dataset.__len__() for key in self.train_dataloader])
        self.max_iter = math.ceil(max_len / self.train_cfg['batch_size'])

        self.loss = Loss(self.train_cfg['loss'], self.train_cfg['tmp_info'])
        self.loss_dict = {'count':0}
        self.tmp_info = {}
        self.loss_log = []

        # set optimizer
        self.optimizer = self._set_optimizer()
        for opt in self.optimizer.values():
            opt.zero_grad(set_to_none=True)

        # resume
        if self.cfg["resume"]:
            # find last checkpoint
            load_epoch = self._find_last_epoch()

            # load last checkpoint
            self.load_checkpoint(load_epoch)
            self.epoch = load_epoch+1

            # logger initialization
            self.logger = Logger((self.max_epoch, self.max_iter), log_dir=self.file_manager.get_dir(''), log_file_option='a')
        else:
            # logger initialization
            self.logger = Logger((self.max_epoch, self.max_iter), log_dir=self.file_manager.get_dir(''), log_file_option='w')

        # wrapping and device setting
        if self.cfg['gpu'] != 'None':
            # model to GPU
            self.model = {key: nn.DataParallel(self.module[key]).cuda() for key in self.module}
            # optimizer to GPU
            for optim in self.optimizer.values():
                for state in optim.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
        else:
            self.model = {key: nn.DataParallel(self.module[key]) for key in self.module}

        # start message
        self.logger.info(self.summary())
        self.logger.start((self.epoch-1, 0))
        self.logger.highlight(self.logger.get_start_msg())

    def _after_train(self):
        # finish message
        self.logger.highlight(self.logger.get_finish_msg())

    def _before_epoch(self):
        self.status = ('epoch %03d/%03d'%(self.epoch, self.max_epoch)).center(status_len)

        # make dataloader iterable.
        self.train_dataloader_iter = {}
        for key in self.train_dataloader:
            self.train_dataloader_iter[key] = iter(self.train_dataloader[key])

        # model training mode
        for key in self.model:
            self.model[key].train()

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
        data = {}
        for key in self.train_dataloader_iter:
            data[key] = next(self.train_dataloader_iter[key])

        # to device
        if self.cfg['gpu'] != 'None':
            for dataset_key in data:
                for key in data[dataset_key]:
                    data[dataset_key][key] = data[dataset_key][key].cuda()

        # forward, cal losses, backward)
        losses, tmp_info = self._forward_fn(self.model, self.loss, data)
        losses   = {key: losses[key].mean()   for key in losses}
        tmp_info = {key: tmp_info[key].mean() for key in tmp_info}

        # backward
        total_loss = sum(v for v in losses.values()).mean()
        total_loss.backward()

        # optimizer step
        for opt in self.optimizer.values():
            opt.step()

        # zero grad
        for opt in self.optimizer.values():
            opt.zero_grad(set_to_none=True) 

        # save losses and tmp_info
        for key in losses:
            if key != 'count':
                if key in self.loss_dict:
                    self.loss_dict[key] += float(losses[key])
                else:
                    self.loss_dict[key] = float(losses[key])
        for key in tmp_info:
            if key in self.tmp_info:
                self.tmp_info[key] += float(tmp_info[key])
            else:
                self.tmp_info[key] = float(tmp_info[key])
        self.loss_dict['count'] += 1

    def _after_step(self):
        # print loss
        if (self.iter%self.cfg['log']['interval_iter']==0 and self.iter!=0) or (self.iter == self.max_iter):
            self.print_loss()

        # print progress
        self.logger.print_prog_msg((self.epoch-1, self.iter-1))

    @torch.no_grad()
    def self_ensemble(self, fn, x):
        '''
        Geomery self-ensemble function
        Note that in this function there is no gradient calculation.
        Args:
            fn : denoiser function
            x : input image
        Return:
            result : self-ensembled image
        '''
        result = torch.zeros_like(x)

        for i in range(8):
            tmp = fn(rot_hflip_img(x, rot_times=i%4, hflip=i//4))
            rot = 4-i%4 if i//4 == 0 else i%4
            result += rot_hflip_img(tmp, rot_times=rot, hflip=i//4)
        return result / 8

    # ====================================================================== #

    def print_loss(self):
        temporal_loss = 0.
        for key in self.loss_dict:
            if key != 'count':
                    temporal_loss += self.loss_dict[key]/self.loss_dict['count']
        self.loss_log += [temporal_loss]

        # print status and learning rate
        loss_out_str = '[%s] %04d/%04d, lr:%s ∣ '%(self.status, self.iter, self.max_iter, "{:.1e}".format(self._get_current_lr()))

        # print losses
        loss_out_str += 'avg_loss : %.3f ∣ '%(np.mean(self.loss_log))

        for key in self.loss_dict:
            if key != 'count':
                loss_out_str += '%s : %.3f ∣ '%(key, self.loss_dict[key]/self.loss_dict['count'])
                self.loss_dict[key] = 0.

        # print temporal information
        if len(self.tmp_info) > 0:
            loss_out_str += '\t['
            for key in self.tmp_info:
                loss_out_str += '  %s : %.2f'%(key, self.tmp_info[key]/self.loss_dict['count'])
                self.tmp_info[key] = 0.
            loss_out_str += ' ]'

        # reset
        self.loss_dict['count'] = 0
        self.logger.info(loss_out_str)

    def save_checkpoint(self):
        checkpoint_name = self._checkpoint_name(self.epoch)
        torch.save({'epoch': self.epoch,
                    'model_weight': {key:self.model[key].module.state_dict() for key in self.model},
                    'optimizer_weight': {key:self.optimizer[key].state_dict() for key in self.optimizer}},
                    os.path.join(self.file_manager.get_dir(self.checkpoint_folder), checkpoint_name))

    def load_checkpoint(self, load_epoch): 
        file_name = os.path.join(self.file_manager.get_dir(self.checkpoint_folder), self._checkpoint_name(load_epoch))
        assert os.path.isfile(file_name), 'there is no checkpoint: %s'%file_name
        saved_checkpoint = torch.load(file_name)
        self.epoch = saved_checkpoint['epoch']
        for key in self.module:
            self.module[key].load_state_dict(saved_checkpoint['model_weight'][key])
        if hasattr(self, 'optimizer'):
            for key in self.optimizer:
                self.optimizer[key].load_state_dict(saved_checkpoint['optimizer_weight'][key])

    def _checkpoint_name(self, epoch):
        return self.session_name + '_%03d'%epoch + '.pth'

    def _find_last_epoch(self):
        checkpoint_list = os.listdir(self.file_manager.get_dir(self.checkpoint_folder))
        epochs = [int(ckpt.replace('%s_'%self.session_name, '').replace('.pth', '')) for ckpt in checkpoint_list]
        assert len(epochs) > 0, 'There is no resumable checkpoint on session %s.'%self.session_name
        return max(epochs)

    def _get_current_lr(self):
        for first_optim in self.optimizer.values():
            for param_group in first_optim.param_groups:
                return param_group['lr']

    def _set_dataloader(self, dataset_cfg, batch_size, shuffle, num_workers):
        dataloader = {}
        dataset_dict = dataset_cfg['dataset']
        if not isinstance(dataset_dict, dict):
            dataset_dict = {'dataset': dataset_dict}

        other_args = self._set_other_args(dataset_cfg)
        for key in dataset_dict:
            dataset = get_dataset_object(dataset_dict[key])(crop_size = dataset_cfg['crop_size'], 
                                                            add_noise = dataset_cfg['add_noise'], 
                                                            mask      = dataset_cfg['mask'],
                                                            aug       = dataset_cfg['aug'] if 'aug' in dataset_cfg else None,
                                                            norm      = dataset_cfg['normalization'],
                                                            n_repeat  = dataset_cfg['n_repeat'] if 'n_repeat' in dataset_cfg else 1,
                                                            n_data    = dataset_cfg['n_data'] if 'n_data' in dataset_cfg else None,
                                                            **other_args,)
            dataloader[key] = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False)

        return dataloader

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
        if 'multiple_cliping' in cfg:
            other_args['multiple_cliping'] = cfg['multiple_cliping']
        if 'keep_on_mem' in cfg:
            other_args['keep_on_mem'] = cfg['keep_on_mem']
        if 'pd' in cfg:
            other_args['pd'] = cfg['pd']
        return other_args

    def summary(self):
        summary = ''

        summary += '-'*100 + '\n'
        # model
        for k, v in self.module.items():
            # get parameter number
            param_num = sum(p.numel() for p in v.parameters())

            # get information about architecture and parameter number
            summary += '[%s] paramters: %s -->'%(k, human_format(param_num)) + '\n'
            summary += str(v) + '\n\n'
        
        # optim

        # Hardware

        summary += '-'*100 + '\n'

        return summary
