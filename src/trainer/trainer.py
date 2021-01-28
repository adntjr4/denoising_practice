import os, math
from importlib import import_module

import cv2
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from .output import Output
from ..model import get_model_object
from ..loss.loss import Loss
from ..loss.metrics import psnr, ssim
from ..datahandler.denoise_dataset import get_dataset_object
from ..util.logger import Logger
from ..util.warmup_scheduler import WarmupLRScheduler
from ..util.util import tensor2np

status_len = 13

class Trainer(Output):
    def __init__(self, cfg):
        self.session_name = cfg['session_name']

        self.checkpoint_folder = 'checkpoint'

        dir_list = ['checkpoint', 'img', 'tboard']
        super().__init__(self.session_name, dir_list)
        
        self.cfg = cfg
        self.train_cfg = cfg['training']
        self.val_cfg   = cfg['validation']
        self.ckpt_cfg  = cfg['checkpoint']

    @torch.no_grad()
    def test(self):
        raise NotImplementedError('TODO')

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

    @torch.no_grad()
    def validation(self):
        # evaluation mode
        self.model.eval()

        # make directories for image saving
        if self.val_cfg['save_image']:
            img_save_path = self.get_dir('img/val_%03d'%self.epoch)
            os.makedirs(img_save_path, exist_ok=True)

        # validation
        psnr_sum = 0.
        psnr_count = 0
        for idx, data in enumerate(self.val_data_loader):
            # to device
            if self.cfg['gpu'] != 'None':
                for key in data:
                    data[key] = data[key].cuda()

            # forward
            denoised_image = self.model(data[self.cfg['model_input']])

            # inverse normalize dataset (if normalization is on)
            if self.val_cfg['normalization']:
                denoised_image = self.val_data_set.inverse_normalize(denoised_image, self.cfg['gpu'] != 'None')
                data = self.val_data_set.inverse_normalize_data(data, self.cfg['gpu'] != 'None')
            
            # evaluation
            if 'clean' in data:
                psnr_value = psnr(denoised_image, data['clean'])
                psnr_sum += psnr_value
                psnr_count += 1

            # image save
            if self.val_cfg['save_image']:
                # to cpu
                if 'clean' in data:
                    clean_img = data['clean'].squeeze().cpu()
                noisy_img = data[self.cfg['model_input']].squeeze().cpu()
                denoi_img = denoised_image.squeeze().cpu()

                # imwrite
                if 'clean' in data:
                    cv2.imwrite(os.path.join(img_save_path, '%04d_CL.png'%idx), tensor2np(clean_img))
                cv2.imwrite(os.path.join(img_save_path, '%04d_N.png'%idx), tensor2np(noisy_img))
                cv2.imwrite(os.path.join(img_save_path, '%04d_DN.png'%idx), tensor2np(denoi_img))

        # info 
        status = ('  val %03d '%self.epoch).ljust(status_len) #.center(status_len)
        if 'clean' in data:
            self.logger.val('[%s] Done! PSNR : %.2f dB'%(status, psnr_sum/psnr_count))
        else:
            self.logger.val('[%s] Done!')

    def _warmup(self):
        self.status = 'warmup'.ljust(status_len) #.center(status_len)

        self.train_data_loader_iter = iter(self.train_data_loader)
        warmup_iter = self.train_cfg['warmup_iter']
        if warmup_iter > self.train_data_loader.__len__():
            self.logger.info('currently warmup support 1 epoch as maximum. warmup iter is replaced to 1 epoch iteration. %d -> %d' \
                % (warmup_iter, self.train_data_loader.__len__()))
            warmup_iter = self.train_data_loader.__len__()
        self.warmup_scheduler = WarmupLRScheduler(self.optimizer, warmup_iter)

        for self.iter in range(1, warmup_iter+1):
            self._before_step()
            self._run_step()
            self._after_step()
            self.warmup_scheduler.step()

    def _before_train(self):
        # initialing
        self.module = get_model_object(self.cfg['model'])()

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
        
        # logger initialization
        self.logger = Logger((self.max_epoch, self.max_iter), log_dir=self.get_dir(''))

        # resume
        if self.cfg["resume"]:
            # find last checkpoint
            load_epoch = self._find_last_epoch()

            # load last checkpoint
            self.load_checkpoint(load_epoch)
            self.epoch = load_epoch+1

        # set optimizer and scheduler
        self.optimizer = self._set_optimizer(self.module.parameters())
        self.scheduler = self._set_scheduler()

        # cuda
        if self.cfg['gpu'] != 'None':
            self.model = nn.DataParallel(self.module).cuda()
            self.loss = self.loss.cuda()
        else:
            self.model = nn.DataParallel(self.module)

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
        self.model.train()

        
    def _run_epoch(self):
        for self.iter in range(1, self.max_iter+1):
            self._before_step()
            self._run_step()
            self._after_step()

    def _after_epoch(self):
        # scheduler step
        self.scheduler.step()

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

        # forward
        model_output = self.model(data[self.cfg['model_input']])

        # get losses
        losses = self.loss(model_output, data)

        # backward
        self.optimizer.zero_grad()
        total_loss = sum(v for v in losses.values())
        total_loss.backward()
        self.optimizer.step()

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
                    'model_weight': self.model.module.state_dict(),
                    'optimizer': self.optimizer,
                    'scheduler': self.scheduler},
                    os.path.join(self.get_dir(self.checkpoint_folder), checkpoint_name))

    def load_checkpoint(self, load_epoch):
        file_name = os.path.join(self.get_dir(self.checkpoint_folder), self._checkpoint_name(load_epoch))
        assert os.path.isfile(file_name), 'there is no checkpoint: %s'%file_name
        saved_checkpoint = torch.load(file_name)
        self.epoch = saved_checkpoint['epoch']
        self.module.load_state_dict(saved_checkpoint['model_weight'])
        self.optimizer = saved_checkpoint['optimizer']
        self.scheduler = saved_checkpoint['schedular']



    def _checkpoint_name(self, epoch):
        return self.session_name + '_%03d'%self.epoch + '.pth'

    def _find_last_epoch(self):
        checkpoint_list = os.listdir(self.get_dir(self.checkpoint_folder))
        epochs = [int(ckpt.replace('%s_'%self.session_name, '').replace('.pth', '')) for ckpt in checkpoint_list]
        assert len(epochs) > 0, 'There is no resumable checkpoint on session %s.'%self.session_name
        return max(epochs)

    def _get_current_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def _set_optimizer(self, parameters):
        opt = self.train_cfg['optimizer']
        lr = float(self.train_cfg['init_lr'])

        if opt['type'] == 'SGD':
            return optim.SGD(parameters, lr=lr, momentum=float(opt['SGD']['momentum']), weight_decay=float(opt['SGD']['weight_decay']))
        elif opt['type'] == 'Adam':
            return optim.Adam(parameters, lr=lr, betas=opt['Adam']['betas'])
        else:
            raise RuntimeError('ambiguious optimizer type: {}'.format(opt['type']))

    def _set_scheduler(self):
        sched = self.train_cfg['scheduler']

        if sched['type'] == 'step':
            return optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=sched['step']['step_size'], gamma=sched['step']['gamma'], last_epoch=self.epoch-2)
        else:
            raise RuntimeError('ambiguious scheduler type: {}'.format(sched['type']))

    def _set_other_args(self, cfg):
        other_args = {}
        if 'power_cliping' in cfg:
            other_args['power_cliping'] = cfg['power_cliping']
        if 'keep_on_mem' in cfg:
            other_args['keep_on_mem'] = cfg['keep_on_mem']
        return other_args
