import os

import cv2
import torch
import torch.autograd as autograd

from .trainer_basic import BasicTrainer, status_len
from ..model import get_model_object
from ..loss.metrics import psnr, ssim
from ..util.util import tensor2np


class Trainer(BasicTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    @torch.no_grad()
    def test(self):
        # initializing
        self._before_test()

        # evaluation mode
        for m in self.model.values():
            m.eval() 

        # make directories for image saving
        if self.test_cfg['save_image']:
            img_save_path = self.get_dir('img/test_%03d'%self.epoch)
            os.makedirs(img_save_path, exist_ok=True)

        # validation
        psnr_sum = 0.
        psnr_count = 0
        for idx, data in enumerate(self.test_dataloader['dataset']):
            # to device
            if self.cfg['gpu'] != 'None':
                for key in data:
                    data[key] = data[key].cuda()

            # forward
            input_data = [data[arg] for arg in self.cfg['model_input']]
            if hasattr(self.model['denoiser'], 'denoise'):
                denoised_image = self.model['denoiser'].denoise(*input_data)
            else:
                denoised_image = self.model['denoiser'](*input_data)

            # inverse normalize dataset (if normalization is on)
            if self.test_cfg['normalization']:
                denoised_image = self.teset_data_set.inverse_normalize(denoised_image, self.cfg['gpu'] != 'None')
                data = self.test_data_set.inverse_normalize_data(data, self.cfg['gpu'] != 'None')

            # evaluation
            if 'clean' in data:
                psnr_value = psnr(denoised_image, data['clean'])
                psnr_sum += psnr_value
                psnr_count += 1

            # image save
            if self.test_cfg['save_image']:
                # to cpu
                if 'clean' in data:
                    clean_img = data['clean'].squeeze().cpu()
                noisy_img = data[self.cfg['model_input']].squeeze().cpu()
                denoi_img = denoised_image.squeeze().cpu()

                # write psnr value on file name
                denoi_name = '%04d_DN_%.2f.png'%(idx, psnr_value) if 'clean' in data else '%04d_DN.png'%idx

                # imwrite
                if 'clean' in data:
                    cv2.imwrite(os.path.join(img_save_path, '%04d_CL.png'%idx), tensor2np(clean_img))
                cv2.imwrite(os.path.join(img_save_path, '%04d_N.png'%idx), tensor2np(noisy_img))
                cv2.imwrite(os.path.join(img_save_path, denoi_name), tensor2np(denoi_img))

            # logger msg
            status = (' test %03d '%self.epoch).ljust(status_len)
            if 'clean' in data:
                self.logger.info('[%s] testing... %04d/%04d. PSNR : %.2f dB'%(status, idx, self.test_dataloader['dataset'].__len__(), psnr_value))
            else:
                self.logger.info('[%s] testing... %04d/%04d.'%(status, idx, self.test_dataloader['dataset'].__len__()))

        # info 
        status = (' test %03d '%self.epoch).ljust(status_len) #.center(status_len)
        if 'clean' in data:
            self.logger.val('[%s] Done! PSNR : %.2f dB'%(status, psnr_sum/psnr_count))
        else:
            self.logger.val('[%s] Done!'%status)

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
        for idx, data in enumerate(self.val_dataloader['dataset']):
            # to device
            if self.cfg['gpu'] != 'None':
                for key in data:
                    data[key] = data[key].cuda()

            # forward
            input_data = [data[arg] for arg in self.cfg['model_input']]
            model_output = self.model['denoiser'](*input_data)

            # get denoised image
            if 'likelihood' in self.train_cfg['loss']:
                c = 1 if model_output.shape[1] == 2 else 3
                means, variance = model_output[:,:c,:,:], model_output[:,c:,:,:]
                variance = torch.pow(variance, 2)
                denoised_image = 1/(1/variance + 1/625) * (1/variance*means+1/625*data[self.cfg['model_input']])
            else:
                denoised_image = model_output

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
                noisy_img = data['real_noisy'] if 'real_noisy' in data else data['syn_noisy']
                noisy_img = noisy_img.squeeze().cpu()
                denoi_img = denoised_image.squeeze().cpu()

                # write psnr value on file name
                denoi_name = '%04d_DN_%.2f.png'%(idx, psnr_value) if 'clean' in data else '%04d_DN.png'%idx

                # imwrite
                if 'clean' in data:
                    cv2.imwrite(os.path.join(img_save_path, '%04d_CL.png'%idx), tensor2np(clean_img))
                cv2.imwrite(os.path.join(img_save_path, '%04d_N.png'%idx), tensor2np(noisy_img))
                cv2.imwrite(os.path.join(img_save_path, denoi_name), tensor2np(denoi_img))

        # info 
        status = ('  val %03d '%self.epoch).ljust(status_len) #.center(status_len)
        if 'clean' in data:
            self.logger.val('[%s] Done! PSNR : %.2f dB'%(status, psnr_sum/psnr_count))
        else:
            self.logger.val('[%s] Done!'%status)

        # return dict form with names in config.

    def _set_module(self):
        module = {}
        module['denoiser'] = get_model_object(self.cfg['model'])()
        return module
        
    def _set_optimizer(self):
        optimizer = {}
        for key in self.module:
            optimizer[key] = self._set_one_optimizer(self.module[key].parameters())
        return optimizer

    def _step(self, data_dict):
        data = data_dict['dataset']

        # forward
        input_data = [data[arg] for arg in self.cfg['model_input']]
        model_output = self.model['denoiser'](*input_data)

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
        # evaluation mode
        for m in self.model.values():
            m.eval()
        status = ('  val %03d '%self.epoch).ljust(status_len) #.center(status_len)

        # make directories for image saving
        if self.val_cfg['save_image']:
            img_save_path = self.get_dir('img/val_%03d'%self.epoch)
            os.makedirs(img_save_path, exist_ok=True)

        # validation
        KL_divergence_sum = 0.
        KL_divergence_count = 0
        for idx, data in enumerate(self.val_dataloader['dataset']):
            # to device
            if self.cfg['gpu'] != 'None':
                for key in data:
                    data[key] = data[key].cuda()
            
            # forward (noisy image generation from clean image)
            generated_noise_map = self.model['model_G'](data['clean'])
            generated_noisy_img = data['clean'] + generated_noise_map

            # inverse normalize dataset & result tensor
            if self.val_cfg['normalization']:
                generated_noise_map = self.val_data_set.inverse_normalize(generated_noise_map, self.cfg['gpu'] != 'None')
                generated_noisy_img = self.val_data_set.inverse_normalize(generated_noisy_img, self.cfg['gpu'] != 'None')
                data = self.val_data_set.inverse_normalize_data(data, self.cfg['gpu'] != 'None')

            # evaluation (KL-divergence)
            '''
            implementation point.
            #1 : do we quantize value of image?
            #2 : can we just sum KL values?
            #3 : what value range do we evaluate KL-div in? [0,1]? [0,255]?  
            '''
            noisy_img = data['real_noisy'] if 'real_noisy' in data else data['syn_noisy']
            clean_img = data['clean']
            real_noise_map = noisy_img - clean_img

            # evaluate KL-div using 'real_noise_map' & generated_noise_map

            
            # image save
            if self.val_cfg['save_image']:
                # to cpu
                clean_img = clean_img.squeeze().cpu()
                noisy_img = noisy_img.squeeze().cpu()
                gener_img = generated_noisy_img.squeeze().cpu()

                # imwrite
                cv2.imwrite(os.path.join(img_save_path, '%04d_CL.png'%idx),    tensor2np(clean_img))
                cv2.imwrite(os.path.join(img_save_path, '%04d_N.png'%idx),     tensor2np(noisy_img))
                cv2.imwrite(os.path.join(img_save_path, '%04d_N_gen.png'%idx), tensor2np(gener_img))

            # print temporal msg
            print('[%s] %04d/%04d evaluating...'%(status, idx, self.val_dataloader['dataset'].__len__()), end='\r')

        # info
        if KL_divergence_count > 0:
            self.logger.val('[%s] Done! KL-div : %.2f dB'%(status, KL_divergence_sum/KL_divergence_count))
        else:
            self.logger.val('[%s] Done!'%status)

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
        TODO : implemete simple DCGAN
        '''
        # WGAN_GP hyper-parameter setting
        n_critic = 5

        data_CL = data['dataset_CL']
        data_N = data['dataset_N']

        losses_for_print = {}

        '''
        STEP 1 (training model_D for critic iterations (here we set 5 as original setting))
        '''
        for _ in range(n_critic):
            # forward
            generated_noisy_img = data_CL['clean'] + (self.model['model_G'](data_CL['clean'])).detach()
            real_noisy_img = data_N['real_noisy']

            D_fake = self.model['model_D'](generated_noisy_img)
            D_real = self.model['model_D'](real_noisy_img)

            # interpolation
            # TODO : implement alpha variable can operate in CPU.
            (N, C, H, W) = real_noisy_img.size()
            alpha = torch.rand(N, 1).cuda()
            map_alpha = alpha.expand(N, int(real_noisy_img.nelement()/N)).contiguous().view(N, C, H, W)
            img_inter = map_alpha*real_noisy_img + (1-map_alpha)*generated_noisy_img
            img_inter = autograd.Variable(img_inter, requires_grad=True)

            D_inter = self.model['model_D'](img_inter)

            # get losses for D
            # (if there is no loss name in configuration, loss returns empty dict.)
            losses = {}
            losses.update(self.loss((D_fake, D_real), None, 'WGAN_D'))
            losses.update(self.loss((D_inter, img_inter), None, 'GP'))

            # zero grad for D optimizer
            self.optimizer['model_D'].zero_grad()

            # backward
            total_loss = sum(v for v in losses.values())
            total_loss.backward()

            # optimizer step
            self.optimizer['model_D'].step()

            # save losses for print
            for key in losses:
                if key in losses_for_print:
                    losses_for_print[key] += losses[key]
                else:
                    losses_for_print[key] = losses[key]
        for key in losses:
            losses_for_print[key] /= n_critic
            
        '''
        STEP 2 (training model_G)
        '''
        generated_noise_map = self.model['model_G'](data_CL['clean'])
        generated_noisy_img = data_CL['clean'] + generated_noise_map
        D_fake_for_G = self.model['model_D'](generated_noisy_img)
        
        # get losses for G
        losses = {}
        losses.update(self.loss(D_fake_for_G, None, 'WGAN_G'))
        losses.update(self.loss(generated_noise_map, None, 'batch_zero_mean'))

        # zero grad
        self.optimizer['model_G'].zero_grad()

        # backward
        total_loss = sum(v for v in losses.values())
        total_loss.backward()

        # optimizer step
        self.optimizer['model_G'].step()

        # save losses for print
        for key in losses:
            losses_for_print[key] = losses[key]

        return losses_for_print
