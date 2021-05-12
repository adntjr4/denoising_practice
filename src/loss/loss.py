import sys
import inspect

import torch
import torch.nn as nn
import torch.autograd as autograd

from . import loss_L
from . import loss_GAN
from . import loss_self
from .single_loss import loss_types


class Loss(nn.Module):
    def __init__(self, loss_string):
        super().__init__()
        loss_string = loss_string.replace(' ', '')

        self.loss_list = []
        for single_loss in loss_string.split('+'):
            weight, name = single_loss.split('*')
            weight = float(weight)
            name = name.lower()

            if name in loss_types:
                self.loss_list.append({ 'name': name,
                                        'weight': float(weight),
                                        'func': loss_types[name]()})
            else:
                raise RuntimeError('undefined loss term: {}'.format(name))
            
    def forward(self, input_data, model_output, data, model, loss_name=None):
        loss_arg = (input_data, model_output, data, model)

        losses = {}
        for single_loss in self.loss_list:
            name = single_loss['name']
            
            # this makes calculate only specific loss.
            if loss_name is not None:
                if loss_name.lower() != name:
                    continue

            if name in loss_types:
                losses[name] = single_loss['weight'] * single_loss['func'](*loss_arg)

        return losses

    def get_loss_dict_form(self):
        loss_dict = {}
        loss_dict['count'] = 0
        for single_loss in self.loss_list:
            loss_dict[single_loss['name']] = 0.
        return loss_dict

def gradient_penalty(D_inter, img_inter):
    '''
    return (||grad(D_inter)||_2 - 1)^2
    Args
        D_inter   : results which input is interpolated image.
        img_inter : interpolation between fake image and real image.
    '''
    gradients = autograd.grad(outputs=D_inter, inputs=img_inter, grad_outputs=torch.ones_like(D_inter), create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    grad_dists = ((gradients.norm(2, dim=1) - 1)**2)
    return grad_dists.mean()
    