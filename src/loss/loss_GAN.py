import torch
import torch.nn as nn
import torch.nn.functional as F

from .single_loss import regist_loss

@regist_loss
class WGAN_D():
    def __call__(self, input_data, model_output, data, model):
        D_fake, D_real = model_output
        return torch.mean(D_fake)-torch.mean(D_real)

@regist_loss
class WGAN_G():
    def __call__(self, input_data, model_output, data, model):
        D_fake_for_G = model_output
        return -torch.mean(D_fake_for_G)

@regist_loss
class DCGAN_D():
    def __call__(self, input_data, model_output, data, model):
        D_fake, D_real = model_output
        return  F.binary_cross_entropy(torch.sigmoid(D_fake), torch.zeros_like(D_fake)) + \
                F.binary_cross_entropy(torch.sigmoid(D_real), torch.ones_like(D_real))

@regist_loss
class DCGAN_G():
    def __call__(self, input_data, model_output, data, model):
        D_fake_for_G = model_output
        return F.binary_cross_entropy(torch.sigmoid(D_fake_for_G), torch.ones_like(D_fake_for_G))

@regist_loss
class GP():
    def __call__(self, input_data, model_output, data, model):
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
