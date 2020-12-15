import os, math
from importlib import import_module

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from .output import Output
from ..model import get_model_object
from ..loss.loss import Loss
from ..dataset.custom_dataset import CustomDataSet
from ..util.logger import Logger
from ..util.warmup_scheduler import WarmupLRScheduler

status_len = 13

class Trainer(Output):
    def __init__(self, cfg):
        self.session_name = cfg['session_name']

        self.checkpoint_folder = 'checkpoint'

        dir_list = ['checkpoint', 'img', 'tboard']
        super().__init__(self.session_name, dir_list)
        
        self.cfg = cfg

    def test(self):
        pass

    def train(self):
        # initializing
        self._before_train()

        # warmup
        if self.epoch == 0 and self.cfg['training']['warmup']:
            self._warmup()

        # training
        for self.epoch in range(self.epoch, self.max_epoch):
            self._before_epoch()
            self._run_epoch()
            self._after_epoch()
        
        self._after_train()

    def validation(self):
        error_count = 0
        for data in self.val_data_loader:
            # to device
            if self.cfg['gpu'] != '-1':
                for key in data:
                    data[key] = data[key].cuda()

            prediction = self.model(data['img'])
            error_count += (torch.argmax(prediction, dim=1)!=data['label']).sum().item()
        status = ('val %02d '%self.epoch).center(status_len)
        self.logger.val('[%s] Error : %.2f %%'%(status, error_count*100/self.val_data_set.__len__()))

    def _warmup(self):
        self.status = 'warmup'.center(status_len)
        self.train_data_loader_iter = iter(self.train_data_loader)

        warmup_iter = self.cfg['training']['warmup_iter']
        self.warmup_scheduler = WarmupLRScheduler(self.optimizer, warmup_iter)

        for self.iter in range(warmup_iter):
            self._before_step()
            self._run_step()
            self._after_step()
            self.warmup_scheduler.step()

    def _before_train(self):
        # initialing
        self.module = get_model_object(self.cfg['model']['name'])()

        self.train_data_set = CustomDataSet(name=self.cfg['training']['dataset'])
        self.val_data_set   = CustomDataSet(name=self.cfg['validation']['dataset'])
        self.train_data_loader = DataLoader(dataset=self.train_data_set, batch_size=self.cfg['training']['batch_size'], shuffle=True, num_workers=self.cfg['thread'])
        self.val_data_loader   = DataLoader(dataset=self.val_data_set, batch_size=self.cfg['validation']['batch_size'], shuffle=False, num_workers=self.cfg['thread'])

        self.max_epoch = self.cfg['training']['max_epoch']
        self.epoch = self.start_epoch = 0
        self.max_iter = math.ceil(self.train_data_loader.dataset.__len__() / self.cfg['training']['batch_size'])

        self.loss = Loss(self.cfg['training']['loss'])
        self.loss_dict = self.loss.get_loss_dict_form()
        
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
        if self.cfg['gpu'] != '-1':
            os.environ['CUDA_VISIBLE_DEVICES'] = self.cfg['gpu']
            self.model = nn.DataParallel(self.module).cuda()
        else:
            self.model = nn.DataParallel(self.module)

        # start message
        self.logger.start((self.epoch, 0))
        self.logger.info(self.logger.get_start_msg())

    def _after_train(self):
        # finish message
        self.logger.info(self.logger.get_finish_msg())

    def _before_epoch(self):
        self.status = ('epoch %02d/%02d'%(self.epoch, self.max_epoch)).center(status_len)

        self.train_data_loader_iter = iter(self.train_data_loader)
        
    def _run_epoch(self):
        for self.iter in range(self.max_iter):
            self._before_step()
            self._run_step()
            self._after_step()

    def _after_epoch(self):
        # scheduler step
        self.scheduler.step()

        # save checkpoint
        self.save_checkpoint()

        # validation
        if self.cfg['validation']['val']:
            if self.epoch % self.cfg['validation']['interval_epoch'] == 0:
                self.validation()

    def _before_step(self):
        pass

    def _run_step(self):
        # get data (data should be dictionary of Tensors)
        data = next(self.train_data_loader_iter)

        # to device
        if self.cfg['gpu'] != '-1':
            for key in data:
                data[key] = data[key].cuda()

        # forward
        model_output = self.model(data['img'])

        # get losses
        losses = self.loss(model_output, data)

        # backward
        self.optimizer.zero_grad()
        total_loss = sum(v for v in losses.values())
        total_loss.backward()
        self.optimizer.step()

        # save losses
        for key in losses:
            self.loss_dict[key] += float(losses[key])

    def _after_step(self):
        # print loss
        if self.iter % self.cfg['log']['interval_iter'] == 0:
            self.print_loss()

        # print progress
        self.logger.print_prog_msg((self.epoch, self.iter))

    def print_loss(self):
        loss_out_str = '[%s] %04d/%04d, lr:%s | '%(self.status, self.iter, self.max_iter, "{:.2e}".format(self._get_current_lr()))
        for loss_name in self.loss_dict:
            loss_out_str += '%s : %.4f | '%(loss_name, self.loss_dict[loss_name]/self.cfg['log']['interval_iter'])
            self.loss_dict[loss_name] = 0.
        self.logger.info(loss_out_str)

    def save_checkpoint(self):
        checkpoint_name = self._checkpoint_name(self.epoch)
        torch.save({'epoch': self.epoch,
                    'model_weight': self.model.module.state_dict(),
                    'optimizer': self.optimizer},
                    os.path.join(self.get_dir(self.checkpoint_folder), checkpoint_name))

    def load_checkpoint(self, load_epoch):
        file_name = os.path.join(self.get_dir(self.checkpoint_folder), self._checkpoint_name(load_epoch))
        assert os.path.isfile(file_name), 'there is no checkpoint: %s'%file_name
        saved_checkpoint = torch.load(file_name)
        self.epoch = saved_checkpoint['epoch']
        self.model.module.load_state_dict(saved_checkpoint['model_weight'])
        self.optimizer = saved_checkpoint['optimizer']

    def _checkpoint_name(self, epoch):
        return self.session_name + '_%03d'%self.epoch + '.pth'

    def _find_last_epoch(self):
        checkpoint_list = os.listdir(self.get_dir(self.checkpoint_folder))
        epochs = [int(ckpt.replace('%s'%self.session_name, '').replace('.pth', '')) for ckpt in checkpoint_list]
        assert len(epochs) > 0, 'There is no resumable checkpoint on session %s.'%self.session_name
        return max(epochs)

    def _get_current_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def _set_optimizer(self, parameters):
        opt = self.cfg['training']['optimizer']
        lr = float(self.cfg['training']['init_lr'])

        if opt['type'] == 'SGD':
            return optim.SGD(parameters, lr=lr, momentum=float(opt['SGD']['momentum']), weight_decay=float(opt['SGD']['weight_decay']))
        elif opt['type'] == 'Adam':
            return optim.Adam(parameters, lr=lr, betas=opt['Adam']['betas'])
        else:
            raise RuntimeError('ambiguious optimizer type: {}'.format(opt['type']))

    def _set_scheduler(self):
        sched = self.cfg['training']['scheduler']

        if sched['type'] == 'step':
            return optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=sched['step']['step_size'], gamma=sched['step']['gamma'], last_epoch=self.epoch-1)
        else:
            raise RuntimeError('ambiguious scheduler type: {}'.format(sched['type']))
