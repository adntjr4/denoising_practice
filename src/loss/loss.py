
import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, loss_string):
        super().__init__()

        self.loss_list = []
        for single_loss in loss_string.split('+'):
            weight, name = single_loss.split('*')

            if name == 'L1' or name == 'self_L1':
                loss_function = nn.L1Loss(reduction='mean')
            elif name == 'L2' or name == 'self_L2':
                loss_function = nn.MSELoss(reduction='mean')
            else:
                raise RuntimeError('ambiguious loss term: {}'.format(name))

            self.loss_list.append({'name': name,
                                   'weight': float(weight),
                                   'func': loss_function})
            
    def forward(self, model_output, data):
        losses = {}
        for single_loss in self.loss_list:
            name = single_loss['name']
            if name == 'L1' or name == 'L2':
                losses[name] = single_loss['weight'] * single_loss['func'](model_output, data['clean'])
            elif name == 'self_L1' or name == 'self_L2':
                self_noisy = data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']
                losses[name] = single_loss['weight'] * single_loss['func'](model_output * data['mask'], self_noisy * data['mask'])
        return losses

    def get_loss_dict_form(self):
        loss_dict = {}
        loss_dict['count'] = 0
        for single_loss in self.loss_list:
            loss_dict[single_loss['name']] = 0.
        return loss_dict

