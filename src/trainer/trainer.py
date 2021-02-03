import os

import cv2
import torch

from .trainer_basic import BasicTrainer, status_len
from ..model import get_model_object
from ..loss.metrics import psnr, ssim
from ..util.util import tensor2np


class Trainer(BasicTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    @torch.no_grad()
    def test(self):
        raise NotImplementedError('TODO')

    @torch.no_grad()
    def validation(self):
        # evaluation mode
        for m in self.model.values():
            m.eval() 

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
            denoised_image = self.model['denoiser'](data[self.cfg['model_input']])

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

    def _set_module(self):
        module = {}
        module['denoiser'] = get_model_object(self.cfg['model'])()
        return module
        
    def _set_optimizer(self):
        optimizer = {}
        for key in self.module:
            optimizer[key] = self._set_one_optimizer(self.module[key].parameters())
        return optimizer

    def _step(self, data):
        # forward
        model_output = self.model['denoiser'](data[self.cfg['model_input']])

        # zero grad
        for opt in self.optimizer.values():
            opt.zero_grad() 

        # get losses
        losses = self.loss(model_output, data)

        # backward
        total_loss = sum(v for v in losses.values())
        total_loss.backward()

        # optimizer step
        for opt in self.optimizer.values():
            opt.step() 

        return losses

class Trainer_GAN(BasicTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.global_iter = 0

    @torch.no_grad()
    def test(self):
        raise NotImplementedError('TODO')

    @torch.no_grad()
    def validation(self):
        raise NotImplementedError('TODO')
        # evaluation mode
        for m in self.model.values():
            m.eval()

        # make directories for image saving
        if self.val_cfg['save_image']:
            img_save_path = self.get_dir('img/val_%03d'%self.epoch)
            os.makedirs(img_save_path, exist_ok=True)

        # validation
        KL_divergence_sum = 0.
        KL_divergence_count = 0
        for idx, data in enumerate(self.val_data_loader):
            # to device
            if self.cfg['gpu'] != 'None':
                for key in data:
                    data[key] = data[key].cuda()
            
            # forward (noisy image generation from clean image)
            generated_noisy_img = data['clean'] + self.model['model_G'](data['clean'])

            # inverse normalize dataset & result tensor
            if self.val_cfg['normalization']:
                generated_noisy_img = self.val_data_set.inverse_normalize(generated_noisy_img, self.cfg['gpu'] != 'None')
                data = self.val_data_set.inverse_normalize_data(data, self.cfg['gpu'] != 'None')

            # evaluation (KL-divergence)
            noisy_target_name = 'real_noisy' if 'real_noisy' in data else 'syn_noisy'
            
            # image save
            if self.val_cfg['save_image']:
                # to cpu
                clean_img = data['clean'].squeeze().cpu()
                noisy_img = data[noisy_target_name].squeeze().cpu()
                gener_img = generated_noisy_img.squeeze().cpu()

                # imwrite
                cv2.imwrite(os.path.join(img_save_path, '%04d_CL.png'%idx),    tensor2np(clean_img))
                cv2.imwrite(os.path.join(img_save_path, '%04d_N.png'%idx),     tensor2np(noisy_img))
                cv2.imwrite(os.path.join(img_save_path, '%04d_N_gen.png'%idx), tensor2np(denoi_img))

        # info
        status = ('  val %03d '%self.epoch).ljust(status_len) #.center(status_len)
        if KL_divergence_count > 0:
            self.logger.val('[%s] Done! KL-div : %.2f dB'%(status, KL_divergence_sum/KL_divergence_count))
        else:
            self.logger.val('[%s] Done!')

    def _set_module(self):
        module = {}
        module['model_G'] = get_model_object(self.cfg['model_G'])()
        module['model_D'] = get_model_object(self.cfg['model_D'])()
        return module

    def _set_optimizer(self):
        optimizer = {}
        for key in self.module:
            optimizer[key] = self._set_one_optimizer(self.module[key].parameters())
        return optimizer
        
    def _step(self, data):
        '''
        Currently here we implemented WGAN-GP only in original CLtoN code.
        TODO : implemete simple DCGAN and WGAN
        '''
        # WGAN_GP hyper-parameter setting
        n_critic = 5
        GP_weight = 10 # (lambda in paper)

        # step 1 (training model_D for critic iterations (here we set 5 as original setting))
        for _ in range(n_critic):
            # forward
            generated_noisy_img = data['clean'] + self.model['model_G'](data['clean'])
            generated_noisy_img = generated_noisy_img.detach()
            real_noisy_img = data['real_noisy']

            D_fake = self.model['model_D'](generated_noisy_img)
            D_real = self.model['model_D'](real_noisy_img)

            # zero grad
            for opt in self.optimizer.values():
                opt.zero_grad() 

            # get losses
            losse

            #

            


        # step 2 (training model_G)
        generated_noisy_img = data['clean'] + self.model['model_G'](data['clean'])

        # zero grad

        # get losses

        # backward

        return losses

class Trainer_w_GAN(BasicTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    @torch.no_grad()
    def test(self):
        raise NotImplementedError('TODO')

    def validation(self):
        raise NotImplementedError('TODO')

    def _set_module(self):
        raise NotImplementedError('TODO')

    def _set_optimizer(self):
        raise NotImplementedError('TODO')
    
    def _step(self, data):
        raise NotImplementedError('TODO')
    
