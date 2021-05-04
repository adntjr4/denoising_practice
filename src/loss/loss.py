
import torch
import torch.nn as nn
import torch.autograd as autograd

eps = 1e-6

class Loss(nn.Module):
    def __init__(self, loss_string):
        super().__init__()
        loss_string = loss_string.replace(' ', '')

        self.loss_list = []
        for single_loss in loss_string.split('+'):
            weight, name = single_loss.split('*')

            if name in ['L1', 'self_L1', 'nlf_L1', 'batch_zero_mean']:
                loss_function = nn.L1Loss(reduction='mean')
            elif name in ['L2', 'self_L2', 'nlf_L2']:
                loss_function = nn.MSELoss(reduction='mean')
            elif name in ['self_Gau_likelihood', 'self_Lap_likelihood', 'WGAN_D', 'WGAN_G', 'DCGAN_D', 'DCGAN_G']:
                loss_function = None
            elif name == 'GP':
                loss_function = gradient_penalty
            elif name in ['self_Gau_likelihood', 'self_Gau_likelihood_DBSN', 'self_Lap_likelihood', 'neg_nlf_mean',
                            'WGAN_D', 'WGAN_G']:
                loss_function = None
            else:
                raise RuntimeError('undefined loss term: {}'.format(name))

            self.loss_list.append({'name': name,
                                   'weight': float(weight),
                                   'func': loss_function})
            
    def forward(self, model_output, data, loss_name=None):
        losses = {}
        for single_loss in self.loss_list:
            name = single_loss['name']
            
            # this makes calculate only specific loss.
            if loss_name is not None:
                if loss_name != name:
                    continue

            if name in ['L1', 'L2']:
                if type(model_output) is tuple: output = model_output[0]
                else: output = model_output
                losses[name] = single_loss['weight'] * single_loss['func'](output, data['clean'])

            elif name in ['self_L1', 'self_L2']:
                if type(model_output) is tuple: output = model_output[0]
                else: output = model_output
                target_noisy = data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']
                losses[name] = single_loss['weight'] * single_loss['func'](output * data['mask'], target_noisy * data['mask'])

            elif name in ['nlf_L1', 'nlf_L2']:
                raise NotImplementedError

            elif name == 'self_Gau_likelihood':
                '''
                MAP loss for gaussian likelihood.
                model output should be as following shape.
                    x_mean (Tensor[b,c,w,h])  : mean of prediction.
                    x_var (Tensor[b,c,c,w,h]) : covariance matrix of prediction.
                    n_sigma (Tensor[b,1,1,w,h])   : estimation of noise level. (in Laine19's paper, noise level is scalar)
                '''
                target_noisy = data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']
                x_mean, x_var, n_sigma = model_output

                # transform the shape of tensors
                b, c, w, h = x_mean.shape
                n_var = torch.eye(c).view(-1)
                if x_mean.is_cuda: n_var = n_var.cuda()
                n_var = torch.pow(n_sigma, 2).permute(0,3,4,1,2).squeeze(-1) * n_var.repeat(b,w,h,1) #b,w,h,c**2
                n_var = n_var.view(b,w,h,c,c)
                x_var = x_var.permute(0,3,4,1,2) # b,w,h,c,c

                y_var = x_var + n_var # b,w,h,c,c
                y_muy = (target_noisy - x_mean).permute(0,2,3,1).unsqueeze(-1) # b,w,h,c,1
                y_var_inv = torch.inverse(y_var) # b,w,h,c,c

                # first term in paper
                loss = torch.matmul(torch.transpose(y_muy,3,4), y_var_inv)
                loss = torch.matmul(loss, y_muy) # b,w,h,1,1
                loss = loss.squeeze(-1).squeeze(-1) # b,w,h

                # second term in paper
                #loss += torch.log(torch.clamp(torch.det(y_var), eps)) # b,w,h
                loss += torch.log(torch.det(y_var)) # b,w,h
                
                # divide with 2
                loss /= 2

                losses[name] = single_loss['weight'] * loss.mean()

            elif name == 'self_Gau_likelihood_DBSN':
                '''
                MAP loss for gaussian likelihood in DBSN paper
                model output should be as following shape.
                    x_mean (Tensor[b,c,w,h])  : mean of prediction.
                    mu_var (Tensor[b,c,c,w,h]) : covariance matrix of difference between prediction and original signal.
                    n_var (Tensor[b,c,c,w,h]) : covariance matrix of noise.
                '''
                target_noisy = data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']
                x_mean, mu_var, n_var = model_output

                mu_var = mu_var.permute(0,3,4,1,2) # b,w,h,c,c
                n_var = n_var.permute(0,3,4,1,2)   # b,w,h,c,c

                y_var = n_var + mu_var # b,w,h,c,c

                y_muy = (target_noisy - x_mean).permute(0,2,3,1).unsqueeze(-1) # b,w,h,c,1
                y_var_inv = torch.inverse(y_var) # b,w,h,c,c

                # first term in paper
                loss = torch.matmul(torch.transpose(y_muy,3,4), y_var_inv)
                loss = torch.matmul(loss, y_muy) # b,w,h,1,1
                loss = loss.squeeze(-1).squeeze(-1) # b,w,h

                # second term in paper (log)
                loss += torch.log(torch.clamp(torch.det(n_var), eps)) # b,w,h

                # third term in paper (trace)
                n_var_inv = torch.inverse(n_var) # b,w,h,c,c
                loss += torch.diagonal(torch.matmul(n_var_inv, mu_var), dim1=-2, dim2=-1).sum(-1)
                
                # divide with 2
                loss /= 2

                losses[name] = single_loss['weight'] * loss.mean()

            elif name == 'self_Lap_likelihood':
                raise NotImplementedError

            elif name == 'WGAN_D':
                D_fake, D_real = model_output
                losses[name] = single_loss['weight'] * (torch.mean(D_fake)-torch.mean(D_real))

            elif name == 'WGAN_G':
                D_fake_for_G = model_output
                losses[name] = single_loss['weight'] * -torch.mean(D_fake_for_G)

            elif name == 'DCGAN_D':
                D_fake, D_real = model_output
                losses[name] = single_loss['weight'] * (F.binary_cross_entropy(torch.sigmoid(D_fake), torch.zeros_like(D_fake)) + \
                                                        F.binary_cross_entropy(torch.sigmoid(D_real), torch.ones_like(D_real)))

            elif name == 'DCGAN_G':
                D_fake_for_G = model_output
                losses[name] = single_loss['weight'] * F.binary_cross_entropy(torch.sigmoid(D_fake_for_G), torch.ones_like(D_fake_for_G))
            
            elif name == 'batch_zero_mean':
                generated_noise_maps = model_output
                batch_mean = torch.mean(generated_noise_maps, dim=(0,2,3), keepdim=False)
                losses[name] = single_loss['weight'] * single_loss['func'](batch_mean, torch.zeros_like(batch_mean))

            elif name == 'neg_nlf_mean':
                losses[name] = single_loss['weight'] * -model_output[2].mean()

            elif name == 'GP':
                D_inter, img_inter = model_output
                losses[name] = single_loss['weight'] * single_loss['func'](D_inter, img_inter)

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
    