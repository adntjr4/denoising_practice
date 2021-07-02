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
class Trainer_GAN(BasicTrainer):
    def __init__(self, cfg):
        raise NotImplementedError('This trainer should be modified: _step() -> _forward_fn()')

        super().__init__(cfg)

        seed = random.randint(1, 10000)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # WGAN_GP hyper-parameter setting
        self.mode = 'WGAN' if 'WGAN' in cfg['training']['loss'] else None
        self.n_critic = 1 if self.mode == 'WGAN' else 1

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
        stds = []
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
                generated_noise_map = self.val_dataloader['dataset'].dataset.inverse_normalize(generated_noise_map, self.cfg['gpu'] != 'None')
                generated_noisy_img = self.val_dataloader['dataset'].dataset.inverse_normalize(generated_noisy_img, self.cfg['gpu'] != 'None')
                data = self.val_dataloader['dataset'].dataset.inverse_normalize_data(data, self.cfg['gpu'] != 'None')

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
            std = float(torch.std(generated_noise_map))
            stds.append(std)

            
            # image save
            if self.val_cfg['save_image']:
                # to cpu
                clean_img = clean_img.squeeze().cpu()
                noisy_img = noisy_img.squeeze().cpu()
                gener_img = generated_noisy_img.squeeze().cpu()

                # imwrite
                clean_name     = '%04d_CL.png'%idx
                noisy_name     = '%04d_N_(%.2f).png'%(idx, data['nlf']) if 'nlf' in data else '%04d_N.png'%idx
                gen_noisy_name = '%04d_N_gen_(%.2f).png'%(idx, std)

                cv2.imwrite(os.path.join(img_save_path, clean_name),    tensor2np(clean_img))
                cv2.imwrite(os.path.join(img_save_path, noisy_name),     tensor2np(noisy_img))
                cv2.imwrite(os.path.join(img_save_path, gen_noisy_name), tensor2np(gener_img))

            # print temporal msg
            print('[%s] %04d/%04d evaluating...'%(status, idx, self.val_dataloader['dataset'].__len__()), end='\r')

        # info
        info = '[%s] Done! '%status
        if KL_divergence_count > 0:
            info += 'KL-div : %.2f dB, '%KL_divergence_sum/KL_divergence_count
        info += 'mean of std : %.2f, '%np.mean(stds)
        info += 'std of std : %.2f, '%np.std(stds)
        self.logger.val(info)

    def _set_module(self):
        module = {}
        module['model_G'] = get_model_object(self.cfg['model_G']['type'])(**self.cfg['model_G']['kwargs'])
        module['model_D'] = get_model_object(self.cfg['model_D']['type'])(**self.cfg['model_D']['kwargs'])
        return module

    def _set_optimizer(self):
        optimizer = {}
        for key in self.module:
            optimizer[key] = self._set_one_optimizer(self.module[key].parameters())
        return optimizer
        
    def _step(self, data):
        '''
        Currently here we implemented WGAN-GP only in original CLtoN code.
        '''
        
        data_CL = data['dataset_CL']
        data_N = data['dataset_N']

        losses_for_print = {}

        '''
        STEP 1 (training model_D for critic iterations (here we set 5 as original setting))
        '''
        for _ in range(self.n_critic):
            # forward
            generated_noisy_img = data_CL['clean'] + (self.model['model_G'](data_CL['clean'])).detach()
            real_noisy_img = data_N['real_noisy'] if 'real_noisy' in data_N else data_N['syn_noisy']

            D_fake = self.model['model_D'](generated_noisy_img)
            D_real = self.model['model_D'](real_noisy_img)

            # get losses for D
            losses = {}

            # interpolation
            if self.mode == 'WGAN':
                (N, C, H, W) = real_noisy_img.size()
                alpha = torch.rand(N, 1).cuda() if real_noisy_img.is_cuda else torch.rand(N, 1)
                map_alpha = alpha.expand(N, int(real_noisy_img.nelement()/N)).contiguous().view(N, C, H, W)
                img_inter = map_alpha*real_noisy_img + (1-map_alpha)*generated_noisy_img
                img_inter = autograd.Variable(img_inter, requires_grad=True)

                D_inter = self.model['model_D'](img_inter)

            if self.mode == 'WGAN':
                losses.update(self.loss(None, (D_fake, D_real), None, None, 'WGAN_D'))
                losses.update(self.loss(None, (D_inter, img_inter), None, None, 'GP'))
            else:
                losses.update(self.loss(None, (D_fake, D_real), None, None, 'DCGAN_D'))
                losses.update(self.loss(None, (D_fake, D_real), None, None, 'LSGAN_D'))

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
            losses_for_print[key] /= self.n_critic
             
        '''
        STEP 2 (training model_G)
        '''
        generated_noise_map = self.model['model_G'](data_CL['clean'])
        generated_noisy_img = data_CL['clean'] + generated_noise_map
        D_fake_for_G = self.model['model_D'](generated_noisy_img)
        
        # get losses for G
        losses = {}
        if self.mode == 'WGAN':
            losses.update(self.loss(None, D_fake_for_G, None, None, 'WGAN_G'))
        else:
            losses.update(self.loss(None, D_fake_for_G, None, None, 'DCGAN_G'))
            losses.update(self.loss(None, D_fake_for_G, None, None, 'LSGAN_G'))
        losses.update(self.loss(None, generated_noise_map, None, None, 'batch_zero_mean'))
        losses.update(self.loss(None, generated_noise_map, None, None, 'zero_mean'))

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

@regist_trainer
class Trainer_DN_GAN(BasicTrainer):
    def __init__(self, cfg):
        raise NotImplementedError('This trainer should be modified: _step() -> _forward_fn()')
        super().__init__(cfg)

        seed = random.randint(1, 10000)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    @torch.no_grad()
    def test(self):
        raise NotImplementedError

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
                means, covariance, nlf = model_output
                b,c,w,h = means.shape

                means = means.unsqueeze(2).permute(0,3,4,1,2) # (b,w,h,c,1)
                covariance = covariance.permute(0,3,4,1,2) # (b,w,h,c,c)
                nlf = nlf.permute(0,3,4,1,2) # (b,w,h,c,c)

                value_sum = torch.matmul(covariance, input_data[0].unsqueeze(2).permute(0,3,4,1,2)) + torch.matmul(nlf, means) # (b,w,h,c,1)
                denoised_image = torch.matmul(torch.inverse(covariance+nlf), value_sum) # (b,w,h,c,1)
                denoised_image = denoised_image.squeeze(4).permute(0,3,1,2)       
            else:
                denoised_image = model_output

            # inverse normalize dataset (if normalization is on)
            if self.val_cfg['normalization']:
                denoised_image = self.val_dataloader['dataset'].dataset.inverse_normalize(denoised_image, self.cfg['gpu'] != 'None')
                data = self.val_dataloader['dataset'].dataset.inverse_normalize_data(data, self.cfg['gpu'] != 'None')
            
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

            # print temporal msg
            print('[%s] %04d/%04d evaluating...'%(status, idx, self.val_dataloader['dataset'].__len__()), end='\r')

        # info 
        if 'clean' in data:
            self.logger.val('[%s] Done! PSNR : %.2f dB'%(status, psnr_sum/psnr_count))
        else:
            self.logger.val('[%s] Done!'%status)

        # return dict form with names in config.

    def _set_module(self):
        module = {}
        module['model_G']  = get_model_object(self.cfg['model_G'])()
        module['denoiser'] = get_model_object(self.cfg['denoiser'])()

        # load model_G
        file_name = os.path.join('output', self.cfg['other'])
        assert os.path.isfile(file_name), 'there is no checkpoint: %s'%file_name
        saved_checkpoint = torch.load(file_name)
        module['model_G'].load_state_dict(saved_checkpoint['model_weight']['model_G'])

        return module

    def _set_optimizer(self):
        optimizer = {}
        optimizer['denoiser'] = self._set_one_optimizer(self.module['denoiser'].parameters())
        return optimizer

    def _step(self, data_dict):
        data = data_dict['dataset']

        # model_G forward for get paired noisy image.
        noisy_image = data['clean'] + self.model['model_G'](data['clean'])

        # denoiser forward
        model_output = self.model['denoiser'](noisy_image)

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

@regist_trainer
class Trainer_GAN_E2E(BasicTrainer):
    def __init__(self, cfg):
        raise NotImplementedError('This trainer should be modified: _step() -> _forward_fn()')
        super().__init__(cfg)

        # WGAN_GP hyper-parameter setting
        self.mode = 'WGAN' if 'WGAN' in cfg['training']['loss'] else 1
        self.n_critic = 1 if self.mode == 'WGAN' else 1

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
        # TODO code for measure KL-divergence

        psnr_sum = 0.
        psnr_count = 0
        stds = []
        for idx, data in enumerate(self.val_dataloader['dataset']):
            # to device
            if self.cfg['gpu'] != 'None':
                for key in data:
                    data[key] = data[key].cuda()
            
            noisy_image_name = 'real_noisy' if 'real_noisy' in data else 'syn_noisy'

            # forward (noisy image generation from clean image)
            gened_noise_map = self.model['model_G'](data['clean'])
            gened_noisy_img = data['clean'] + gened_noise_map
            denoised_image = self.model['denoiser'](data[noisy_image_name])

            # inverse normalize dataset & result tensor
            if self.val_cfg['normalization']:
                gened_noise_map = self.val_dataloader['dataset'].dataset.inverse_normalize(gened_noise_map, self.cfg['gpu'] != 'None')
                gened_noisy_img = self.val_dataloader['dataset'].dataset.inverse_normalize(gened_noisy_img, self.cfg['gpu'] != 'None')
                denoised_image  = self.val_dataloader['dataset'].dataset.inverse_normalize(denoised_image, self.cfg['gpu'] != 'None')
                data            = self.val_dataloader['dataset'].dataset.inverse_normalize_data(data, self.cfg['gpu'] != 'None')

            # evaluation
            if 'clean' in data:
                psnr_value = psnr(denoised_image, data['clean'])
                psnr_sum += psnr_value
                psnr_count += 1

            std = float(torch.std(gened_noise_map))
            stds.append(std)

            # image save
            if self.val_cfg['save_image']:
                # to cpu
                if 'clean' in data:
                    clean_img = data['clean'].squeeze().cpu()
                noisy_img = data['real_noisy'] if 'real_noisy' in data else data['syn_noisy']
                noisy_img = noisy_img.squeeze().cpu()
                denoi_img = denoised_image.squeeze().cpu()
                gened_img = gened_noisy_img.squeeze().cpu()

                # write psnr value on file name
                denoi_name = '%04d_DN_%.2f.png'%(idx, psnr_value) if 'clean' in data else '%04d_DN.png'%idx

                # imwrite
                if 'clean' in data:
                    cv2.imwrite(os.path.join(img_save_path, '%04d_CL.png'%idx), tensor2np(clean_img))
                cv2.imwrite(os.path.join(img_save_path, '%04d_N.png'%idx), tensor2np(noisy_img))
                cv2.imwrite(os.path.join(img_save_path, '%04d_N_gen_(%.2f).png'%(idx, std)), tensor2np(gened_img))
                cv2.imwrite(os.path.join(img_save_path, denoi_name), tensor2np(denoi_img))

            # print temporal msg
            print('[%s] %04d/%04d evaluating...'%(status, idx, self.val_dataloader['dataset'].__len__()), end='\r')

        # info
        info = '[%s] Done! '%status

        if 'clean' in data:
            info += 'PSNR : %.2f dB, '%(psnr_sum/psnr_count)

        info += 'mean of std : %.2f, '%np.mean(stds)
        info += 'std of std : %.2f, '%np.std(stds)

        self.logger.val(info)

    def _set_module(self):
        module = {}
        module['model_G']  = get_model_object(self.cfg['model_G']['type'])(**self.cfg['model_G']['kwargs'])
        module['model_DN'] = get_model_object(self.cfg['model_DN']['type'])(**self.cfg['model_DN']['kwargs'])
        module['model_DC'] = get_model_object(self.cfg['model_DC']['type'])(**self.cfg['model_DC']['kwargs'])
        module['denoiser'] = get_model_object(self.cfg['denoiser']['type'])(**self.cfg['denoiser']['kwargs'])
        return module

    def _set_optimizer(self):
        optimizer = {}
        for key in self.module:
            optimizer[key] = self._set_one_optimizer(self.module[key].parameters())
        return optimizer
        
    def _step(self, data):
        '''
        end-to-end learning combining with GAN and denoiser.
        Currently here we implemented WGAN-GP only in original CLtoN code.
        '''
        data_CL = data['dataset_CL']
        data_N = data['dataset_N']
        noisy_data_name = 'real_noisy' if 'real_noisy' in data_N else 'syn_noisy'

        losses_for_print = {}

        '''
        STEP 1 (training model_DN and model_DC for critic iterations (here we set 5 as original setting))
        '''
        for _ in range(self.n_critic):
            # forward
            clean_img = data_CL['clean']
            #denoised_img = self.model['denoiser'](data_N[noisy_data_name]).detach()

            generated_noisy_img = clean_img + (self.model['model_G'](clean_img)).detach()
            real_noisy_img = data_N[noisy_data_name]

            DN_fake = self.model['model_DN'](generated_noisy_img)
            DN_real = self.model['model_DN'](real_noisy_img)

            #DC_fake = self.model['model_DC'](denoised_img)
            #DC_real = self.model['model_DC'](clean_img)

            # get losses for D
            losses = {}

            # interpolation for gp
            if self.mode == 'WGAN':
                (N, C, H, W) = real_noisy_img.size()
                alpha = torch.rand(N, 1).cuda() if real_noisy_img.is_cuda else torch.rand(N, 1)
                map_alpha = alpha.expand(N, int(real_noisy_img.nelement()/N)).contiguous().view(N, C, H, W)

                N_img_inter = map_alpha*real_noisy_img + (1-map_alpha)*generated_noisy_img
                N_img_inter = autograd.Variable(N_img_inter, requires_grad=True)

                C_img_inter = map_alpha*clean_img + (1-map_alpha)*denoised_img
                C_img_inter = autograd.Variable(C_img_inter, requires_grad=True)

                DN_inter = self.model['model_DN'](N_img_inter)
                DC_inter = self.model['model_DC'](C_img_inter)

            if self.mode == 'WGAN':
                losses.update(self.loss(None, (DN_fake, DN_real), None, None, loss_name='WGAN_D', change_name='WGAN_DN'))
                #losses.update(self.loss(None, (DC_fake, DC_real), None, None, loss_name='WGAN_D', change_name='WGAN_DC'))
                losses.update(self.loss(None, (DN_inter, N_img_inter), None, None, loss_name='GP', change_name='GP_N'))
                #losses.update(self.loss(None, (DC_inter, C_img_inter), None, None, loss_name='GP', change_name='GP_C'))
            else:
                losses.update(self.loss(None, (DN_fake, DN_real), None, None, loss_name='DCGAN_D', change_name='DCGAN_DN'))
                #losses.update(self.loss(None, (DC_fake, DC_real), None, None, loss_name='DCGAN_D', change_name='DCGAN_DC'))

            # zero grad for D optimizer
            self.optimizer['model_DN'].zero_grad()
            self.optimizer['model_DC'].zero_grad()

            # backward
            total_loss = sum(v for v in losses.values())
            total_loss.backward()

            # optimizer step
            self.optimizer['model_DN'].step()
            #self.optimizer['model_DC'].step()

            # save losses for print
            for key in losses:
                if key in losses_for_print:
                    losses_for_print[key] += losses[key]
                else:
                    losses_for_print[key] = losses[key]
        for key in losses:
            losses_for_print[key] /= self.n_critic
            
        '''
        STEP 2 (training model_G and denoiser with adverarial loss and cyclic loss)
        '''
        generated_noise_map = self.model['model_G'](data_CL['clean'])
        generated_noisy_img = data_CL['clean'] + generated_noise_map
        DN_fake_for_G = self.model['model_DN'](generated_noisy_img)
        denoised_generated_noisy_img = self.model['denoiser'](generated_noisy_img)

        # denoised_img = self.model['denoiser'](data_N[noisy_data_name])
        # denoised_map = data_N[noisy_data_name] - denoised_img
        # DC_fake_for_denoiser = self.model['model_DC'](denoised_img)

        # get losses for G
        losses = {}
        if self.mode == 'WGAN':
            losses.update(self.loss(None, DN_fake_for_G,        None, None, loss_name='WGAN_G', change_name='WGAN_GN'))
            # losses.update(self.loss(None, DC_fake_for_denoiser, None, None, loss_name='WGAN_G', change_name='WGAN_GC'))
        else:
            losses.update(self.loss(None, DN_fake_for_G,        None, None, loss_name='DCGAN_G', change_name='DCGAN_GN'))
            # losses.update(self.loss(None, DC_fake_for_denoiser, None, None, loss_name='DCGAN_G', change_name='DCGAN_GC'))
            
        losses.update(self.loss(None, denoised_generated_noisy_img, data_CL, None, loss_name='L1',   change_name='L1_cyclic'))
        losses.update(self.loss(None, denoised_generated_noisy_img, data_CL, None, loss_name='L2',   change_name='L2_cyclic'))
        losses.update(self.loss(None, generated_noise_map,  None, None, loss_name='batch_zero_mean', change_name='BZM_N'))
        # losses.update(self.loss(None, denoised_map,         None, None, loss_name='batch_zero_mean', change_name='BZM_C'))

        # zero grad
        self.optimizer['model_G'].zero_grad()
        self.optimizer['denoiser'].zero_grad()

        # backward
        total_loss = sum(v for v in losses.values())
        total_loss.backward()

        # optimizer step
        self.optimizer['model_G'].step()
        self.optimizer['denoiser'].step()

        # save losses for print
        for key in losses:
            losses_for_print[key] = losses[key]

        return losses_for_print
