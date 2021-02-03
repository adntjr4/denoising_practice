
import torch
import torch.nn as nn
import torch.autograd as autograd


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
            elif name == 'WGAN_D' or name == 'WGAN_G':
                loss_function = None
            elif name == 'GP':
                loss_function = gradient_penalty
            else:
                raise RuntimeError('ambiguious loss term: {}'.format(name))

            self.loss_list.append({'name': name,
                                   'weight': float(weight),
                                   'func': loss_function})
            
    def forward(self, model_output, data, loss_name=None):
        losses = {}
        for single_loss in self.loss_list:
            name = single_loss['name'] if loss_name is None else loss_name
            if name == 'L1' or name == 'L2':
                losses[name] = single_loss['weight'] * single_loss['func'](model_output, data['clean'])
            elif name == 'self_L1' or name == 'self_L2':
                self_noisy = data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']
                losses[name] = single_loss['weight'] * single_loss['func'](model_output * data['mask'], self_noisy * data['mask'])
            elif name == 'WGAN_D':
                D_fake, D_real = model_output
                losses[name] = single_loss['weight'] * (torch.mean(D_fake)-torch.mean(D_real))
            elif name == 'WGAN_G':
                losses[name] = single_loss['weight'] * -torch.mean(model_output)
            elif name == 'GP':
                D_inter, img_inter = model_output
                losses[name] = single_loss['weight'] * single_loss['func'](D_inter, img_inter)
            
            if loss_name is not None:
                return losses
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
    