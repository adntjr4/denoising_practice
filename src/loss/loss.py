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
            ratio = True if 'r' in weight else False
            weight = float(weight.replace('r', ''))
            name = name.lower()

            if name in loss_types:
                self.loss_list.append({ 'name': name,
                                        'weight': float(weight),
                                        'func': loss_types[name](),
                                        'ratio': ratio})
            else:
                raise RuntimeError('undefined loss term: {}'.format(name))
            
    def forward(self, input_data, model_output, data, model, loss_name=None, change_name=None, ratio=1.0):
        '''
        forward all loss and return as dict format.
        Args
            input_data   : input of the network (also in the data)
            model_output : output of the network
            data         : entire batch of data
            model        : model (for another network forward)
            loss_name    : (optional) choose specific loss with name
            ratio        : percentage of learning procedure for increase weight during training
        Return
            losses       : dictionary of loss
        '''
        loss_arg = (input_data, model_output, data, model)

        losses = {}
        for single_loss in self.loss_list:
            name = single_loss['name']
            
            # this makes calculate only specific loss.
            if loss_name is not None:
                if loss_name.lower() != name:
                    continue
                else:
                    if change_name is not None:
                        losses[change_name] = single_loss['weight'] * single_loss['func'](*loss_arg)
                        if single_loss['ratio']: losses[change_name] *= ratio 
                        return losses

            losses[name] = single_loss['weight'] * single_loss['func'](*loss_arg)
            if single_loss['ratio']: losses[name] *= ratio 

        return losses

    # def get_loss_dict_form(self):
    #     '''
    #     return dict{name: 0} for log or whatever
    #     '''
    #     loss_dict = {}
    #     loss_dict['count'] = 0
    #     for single_loss in self.loss_list:
    #         loss_dict[single_loss['name']] = 0.
    #     return loss_dict
