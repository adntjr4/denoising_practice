import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from . import regist_loss

@regist_loss
class WGAN_D():
    def __call__(self, input_data, model_output, data, module):
        D_fake, D_real = model_output
        return torch.mean(D_fake)-torch.mean(D_real)

@regist_loss
class WGAN_G():
    def __call__(self, input_data, model_output, data, module):
        D_fake_for_G = model_output
        return -torch.mean(D_fake_for_G)

@regist_loss
class DCGAN_D():
    def __call__(self, input_data, model_output, data, module):
        D_fake, D_real = model_output
        return  F.binary_cross_entropy(torch.sigmoid(D_fake), torch.zeros_like(D_fake)) + \
                F.binary_cross_entropy(torch.sigmoid(D_real), torch.ones_like(D_real))

@regist_loss
class DCGAN_G():
    def __call__(self, input_data, model_output, data, module):
        D_fake_for_G = model_output
        return F.binary_cross_entropy(torch.sigmoid(D_fake_for_G), torch.ones_like(D_fake_for_G))

@regist_loss
class LSGAN_D():
    def __call__(self, input_data, model_output, data, module):
        D_fake, D_real = model_output
        return  F.mse_loss(D_fake, torch.zeros_like(D_fake)) + \
                F.mse_loss(D_real, torch.ones_like(D_real))

@regist_loss
class LSGAN_G():
    def __call__(self, input_data, model_output, data, module):
        D_fake_for_G = model_output
        return F.mse_loss(D_fake_for_G, torch.ones_like(D_fake_for_G))

@regist_loss
class GP():
    def __call__(self, input_data, model_output, data, module):
        '''
        return (||grad(D_inter)||_2 - 1)^2
        Args
            D_inter   : results which input is interpolated image.
            img_inter : interpolation between fake image and real image.
        '''
        D_inter, img_inter = model_output

        gradients = autograd.grad(outputs=D_inter, inputs=img_inter, grad_outputs=torch.ones_like(D_inter), create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        grad_dists = ((gradients.norm(2, dim=1) - 1)**2)

        return grad_dists.mean()

@regist_loss
class batch_zero_mean():
    def __call__(self, input_data, model_output, data, module):
        generated_noise_maps = model_output
        batch_mean = torch.mean(generated_noise_maps, dim=(0,2,3), keepdim=False)
        return F.l1_loss(batch_mean, torch.zeros_like(batch_mean))

@regist_loss
class zero_mean():
    def __call__(self, input_data, model_output, data, module):
        generated_noise_maps = model_output
        instance_mean = torch.mean(generated_noise_maps, dim=(2,3), keepdim=False)
        return F.l1_loss(instance_mean, torch.zeros_like(instance_mean))
