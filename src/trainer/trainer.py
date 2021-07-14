import os
import random

import cv2
import numpy as np
import torch
import torch.autograd as autograd

from ..util.dnd_submission.bundle_submissions import bundle_submissions_srgb
from ..util.dnd_submission.dnd_denoise import denoise_srgb
from ..util.dnd_submission.pytorch_wrapper import pytorch_denoiser

from . import regist_trainer
from .basic_trainer import BasicTrainer, status_len
from ..model import get_model_object
from ..util.util import tensor2np, psnr, ssim, pixel_shuffle_up_sampling


@regist_trainer
class Trainer(BasicTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    @torch.no_grad()
    def test(self):
        # initializing
        self._before_test()

        # evaluation mode
        for key in self.model:
            self.model[key].eval()

        # dnd benchmark evaluation is seperated.
        if self.cfg['test']['dataset'] == 'DND_benchmark':
            self.test_DND()
            exit()

        # make directories for image saving
        if self.test_cfg['save_image']:
            img_save_path = 'img/test_%03d' % self.epoch
            self.file_manager.make_dir(img_save_path)

        # validation
        psnr_sum = 0.
        ssim_sum = 0.
        count = 0
        for idx, data in enumerate(self.test_dataloader['dataset']):
            # to device
            if self.cfg['gpu'] != 'None':
                for key in data:
                    data[key] = data[key].cuda()

            # forward
            input_data = [data[arg] for arg in self.cfg['model_input']]
            if hasattr(self.model['denoiser'].module, 'denoise'):
                denoiser = self.model['denoiser'].module.denoise
            else:
                denoiser =  self.model['denoiser'].module

            # denoising w/ or w/o self ensemble
            if self.cfg['self_en']: denoiser = lambda x: self.self_ensemble(denoiser, x)

            # repeat
            denoised_image = 0.
            for i in range(self.cfg['repeat']):
                denoised_image += denoiser(*input_data)
            denoised_image /= self.cfg['repeat']
                
            # inverse normalize dataset (if normalization is on)
            if self.test_cfg['normalization']:
                denoised_image = self.test_dataloader['dataset'].dataset.inverse_normalize(denoised_image, self.cfg['gpu'] != 'None')
                data = self.test_dataloader['dataset'].dataset.inverse_normalize_data(data, self.cfg['gpu'] != 'None')

            # evaluation
            denoised_image += 0.5
            if 'clean' in data:
                psnr_value = psnr(denoised_image, data['clean'])
                ssim_value = ssim(denoised_image, data['clean'])

                psnr_sum += psnr_value
                ssim_sum += ssim_value

                count += 1

            # image save
            if self.test_cfg['save_image']:
                # to cpu
                if 'clean' in data:
                    clean_img = data['clean'].squeeze().cpu()
                noisy_img = data['real_noisy'] if 'real_noisy' in data else data['syn_noisy']
                noisy_img = noisy_img.squeeze().cpu()
                denoi_img = denoised_image.squeeze().cpu()

                # write psnr value on file name
                denoi_name = '%04d_DN_%.2f'%(idx, psnr_value) if 'clean' in data else '%04d_DN'%idx

                # imwrite
                if 'clean' in data:
                    self.file_manager.save_img_tensor(img_save_path, '%04d_CL'%idx, clean_img)
                self.file_manager.save_img_tensor(img_save_path, '%04d_N'%idx, noisy_img)
                self.file_manager.save_img_tensor(img_save_path, denoi_name, denoi_img)

            # logger msg
            status = (' test %03d '%self.epoch).ljust(status_len)
            if 'clean' in data:
                self.logger.info('[%s] testing... %04d/%04d. PSNR : %.2f dB'%(status, idx, self.test_dataloader['dataset'].__len__(), psnr_value))
            else:
                self.logger.info('[%s] testing... %04d/%04d.'%(status, idx, self.test_dataloader['dataset'].__len__()))

        # info 
        status = (' test %03d '%self.epoch).ljust(status_len) #.center(status_len)
        if 'clean' in data:
            self.logger.val('[%s] Done! PSNR : %.2f dB, SSIM : %.3f'%(status, psnr_sum/count, ssim_sum/count))
        else:
            self.logger.val('[%s] Done!'%status)

    @torch.no_grad()
    def test_DND(self):
        # make directories for image saving
        if self.test_cfg['save_image']:
            img_save_path = 'img/DND_%03d' % self.epoch
            self.file_manager.make_dir(img_save_path)

        # denoiser wrapping
        if hasattr(self.model['denoiser'].module, 'denoise'):
            denoiser = self.model['denoiser'].module.denoise
        else:
            denoiser = self.model['denoiser'].module

        def wrap_denoiser(Inoisy, nlf, idx):
            noisy = 255 * torch.from_numpy(Inoisy)

            # to device
            if self.cfg['gpu'] != 'None':
                noisy = noisy.cuda()

            noisy = autograd.Variable(noisy)

            # processing
            noisy = noisy.permute(2,0,1)
            noisy = self.test_dataloader['dataset'].dataset._pre_processing({'real_noisy': noisy})['real_noisy']

            noisy = noisy.view(1,noisy.shape[0], noisy.shape[1], noisy.shape[2])

            # denoising w/ or w/o self ensemble
            if self.cfg['self_en']:
                denoised = self.self_ensemble(denoiser, noisy)
            else:
                denoised = denoiser(noisy)
            
            # inverse normalize dataset (if normalization is on)
            if self.test_cfg['normalization']:
                denoised = self.test_dataloader['dataset'].dataset.inverse_normalize_data({'real_noisy': denoised}, self.cfg['gpu'] != 'None')['real_noisy']
            if 'pd' in self.test_cfg:
                denoised = pixel_shuffle_up_sampling(denoised, int(self.test_cfg['pd']))

            denoised += 0.5
            denoised = denoised[0,...].cpu().numpy()
            denoised = np.transpose(denoised, [1,2,0])

            # image save
            if self.test_cfg['save_image']:
                self.file_manager.save_img_numpy(img_save_path, '%04d_N'%idx,  255*Inoisy)
                self.file_manager.save_img_numpy(img_save_path, '%04d_DN'%idx, denoised)

            return denoised / 255

        denoise_srgb(wrap_denoiser, './dataset/DND/dnd_2017', img_save_path)

        bundle_submissions_srgb(img_save_path)

        # info 
        status = (' test %03d '%self.epoch).ljust(status_len) #.center(status_len)
        self.logger.val('[%s] Done!'%status)

    @torch.no_grad()
    def validation(self):
        # evaluation mode
        for key in self.model:
            self.model[key].eval()
        status = ('  val %03d '%self.epoch).ljust(status_len) #.center(status_len)

        # make directories for image saving
        if self.val_cfg['save_image']:
            img_save_path = 'img/val_%03d' % self.epoch
            self.file_manager.make_dir(img_save_path)

        # validation
        psnr_sum = 0.
        ssim_sum = 0.
        count = 0
        for idx, data in enumerate(self.val_dataloader['dataset']):
            # to device
            if self.cfg['gpu'] != 'None':
                for key in data:
                    data[key] = data[key].cuda()

            # forward
            input_data = [data[arg] for arg in self.cfg['model_input']]
            if hasattr(self.model['denoiser'].module, 'denoise'):
                denoised_image = self.model['denoiser'].module.denoise(*input_data)
            else:
                denoised_image = self.model['denoiser'].module(*input_data)

            # inverse normalize dataset (if normalization is on)
            if self.val_cfg['normalization']:
                denoised_image = self.val_dataloader['dataset'].dataset.inverse_normalize(denoised_image, self.cfg['gpu'] != 'None')
                data = self.val_dataloader['dataset'].dataset.inverse_normalize_data(data, self.cfg['gpu'] != 'None')
            
            denoised_image += 0.5

            # evaluation
            if 'clean' in data:
                psnr_value = psnr(denoised_image, data['clean'])
                ssim_value = ssim(denoised_image, data['clean'])

                psnr_sum += psnr_value
                ssim_sum += ssim_value

                count += 1

            # image save
            if self.val_cfg['save_image']:
                # to cpu
                if 'clean' in data:
                    clean_img = data['clean'].squeeze().cpu()
                noisy_img = data['real_noisy'] if 'real_noisy' in data else data['syn_noisy']
                noisy_img = noisy_img.squeeze().cpu()
                denoi_img = denoised_image.squeeze().cpu()

                # write psnr value on file name
                denoi_name = '%04d_DN_%.2f'%(idx, psnr_value) if 'clean' in data else '%04d_DN'%idx

                # imwrite
                if 'clean' in data:
                    self.file_manager.save_img_tensor(img_save_path, '%04d_CL'%idx, clean_img)
                self.file_manager.save_img_tensor(img_save_path, '%04d_N'%idx, noisy_img)
                self.file_manager.save_img_tensor(img_save_path, denoi_name, denoi_img)

            # print temporal msg
            self.logger.note('[%s] %04d/%04d evaluating...'%(status, idx, self.val_dataloader['dataset'].__len__()), end='\r')

        # info 
        if 'clean' in data:
            self.logger.val('[%s] Done! PSNR : %.2f dB, SSIM : %.3f'%(status, psnr_sum/count, ssim_sum/count))
        else:
            self.logger.val('[%s] Done!'%status)

        # return dict form with names in config.

    def _set_module(self):
        module = {}
        if self.cfg['model']['kwargs'] is None:
            module['denoiser'] = get_model_object(self.cfg['model']['type'])()
        else:   
            module['denoiser'] = get_model_object(self.cfg['model']['type'])(**self.cfg['model']['kwargs'])
        return module
        
    def _set_optimizer(self):
        optimizer = {}
        for key in self.module:
            optimizer[key] = self._set_one_optimizer(self.module[key].parameters())
        return optimizer

    def _forward_fn(self, module, loss, data):
        data = data['dataset']

        # forward
        input_data = [data[arg] for arg in self.cfg['model_input']]
        model_output = module['denoiser'](*input_data)

        # get losses
        losses, tmp_info = loss(input_data, model_output, data, module, \
                                    ratio=(self.epoch-1 + (self.iter-1)/self.max_iter)/self.max_epoch)

        # losses   = {key : losses[key].unsqueeze(-1) for key in losses}
        # tmp_info = {key : tmp_info[key].unsqueeze(-1) for key in tmp_info}

        return losses, tmp_info
