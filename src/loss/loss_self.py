import torch
import torch.nn as nn
import torch.nn.functional as F

from . import regist_loss
from ..util.util import mean_conv2d, variance_conv2d, get_gaussian_2d_filter

eps = 1e-6

# =================== #
#      Recon loss     #
# =================== #

@regist_loss
class self_L1():
    def __call__(self, input_data, model_output, data, module):
        if type(model_output) is tuple: 
            output = model_output[0]
        else: 
            output = model_output

        target_noisy = data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']

        return F.l1_loss(output, target_noisy)

@regist_loss
class self_L1_1():
    def __call__(self, input_data, model_output, data, module):
        if type(model_output) is tuple: 
            output = model_output[1]
        else: 
            output = model_output

        target_noisy = data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']

        return F.l1_loss(output, target_noisy)       

@regist_loss
class self_L2():
    def __call__(self, input_data, model_output, data, module):
        if type(model_output) is tuple: 
            output = model_output[0]
        else: 
            output = model_output

        target_noisy = data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']

        return F.mse_loss(output, target_noisy)

@regist_loss
class self_huber():
    def __call__(self, input_data, model_output, data, module):
        if type(model_output) is tuple: 
            output = model_output[0]
        else: 
            output = model_output

        target_noisy = data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']

        return F.smooth_l1_loss(output, target_noisy, beta=1)

@regist_loss
class self_ssim():
    def __call__(self, input_data, model_output, data, module):
        '''
        denoised = model_output
        '''
        # hyper-parameters
        C1 = (0.01*255) ** 2
        C2 = (0.03*255) ** 2
        window_size = 11
        sigma = 3.0

        # extract needed data
        denoised = model_output
        real_noisy = data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']
        nlf_map = nlf_est(real_noisy, window_size=window_size, filter_type='gau', sigma=sigma)
        noise_map = real_noisy-denoised

        # get gaussian kernel
        window = get_gaussian_2d_filter(window_size, sigma, channel=real_noisy.shape[1]).to(real_noisy.device)

        # calculate mean and variance of each image
        mu_y = mean_conv2d(real_noisy, window=window, padd=False)
        mu_x = mean_conv2d(denoised, window=window, padd=False)
        mu_n = mean_conv2d(noise_map, window=window, padd=False)
        mu_y_mu_x = mu_y*mu_x

        sigma_y_sq = variance_conv2d(real_noisy, window=window, padd=False)
        sigma_x_sq = variance_conv2d(denoised, window=window, padd=False)

        nlf_map = nlf_map.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(sigma_y_sq.shape)
        sigma_y_n_sq = F.relu(sigma_y_sq - nlf_map.pow(2), inplace=True)

        sigma_yx = mean_conv2d(real_noisy * denoised, window=window, padd=False) - mu_y_mu_x
        sigma_nx = mean_conv2d(noise_map * denoised, window=window, padd=False) - mu_n*mu_x

        # calculate SSIM
        v1 = 2 * sigma_yx + C2
        v2 = sigma_y_n_sq + sigma_x_sq + C2
        ssim_map = ((2 * mu_y_mu_x + C1) * v1) / ((mu_y.pow(2) + mu_x.pow(2) + C1) * v2)

        # return SSIM + sigma_nx
        return (1-ssim_map.mean((1,2,3)) + sigma_nx.pow(2).mean((1,2,3))).mean()

# =================== #
#       MAP loss      #
# =================== #

@regist_loss
class MAP_scalar():
    '''
    MAP loss for gaussian likelihood from Laine et al.
    model output should be as following shape.
        x_mean (Tensor[b,c,w,h])  : mean of prediction.
        x_var (Tensor[b,c,c,w,h]) : covariance matrix of prediction.
        n_sigma (Tensor[b])   : estimation of noise level.
    '''
    def __call__(self, input_data, model_output, data, module):
        target_noisy = data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']
        x_mean, x_var, n_sigma = model_output

        # transform the shape of tensors
        b, c, w, h = x_mean.shape
        n_var = torch.eye(c).view(-1)
        if x_mean.is_cuda: n_var = n_var.cuda()
        n_var = torch.pow(n_sigma, 2) * n_var.repeat(b,w,h,1).permute(1,2,3,0) # w,h,c**2,b
        n_var = n_var.permute(3,0,1,2) # b,w,h,c**2
        n_var = n_var.view(b,w,h,c,c)
        x_var = x_var.permute(0,3,4,1,2) # b,w,h,c,c
        epsI = eps * torch.eye(c, device=x_mean.device).repeat(b,w,h,1,1)

        y_muy = (target_noisy - x_mean).permute(0,2,3,1).unsqueeze(-1) # b,w,h,c,1

        # first term in paper
        loss = torch.matmul(torch.transpose(y_muy,3,4), torch.inverse(x_var + n_var + epsI))
        loss = torch.matmul(loss, y_muy) # b,w,h,1,1
        loss = loss.squeeze(-1).squeeze(-1) # b,w,h

        # second term in paper
        loss += torch.log(torch.clamp(torch.det(x_var + n_var), eps)) # b,w,h
        # loss += torch.log(torch.det(x_var + n_var)) # b,w,h
        
        # divide with 2
        loss /= 2

        return loss.mean()

@regist_loss
class MAP():
    '''
    MAP loss for gaussian likelihood from Laine et al.
    model output should be as following shape.
        x_mean (Tensor[b,c,w,h])   : mean of prediction.
        mu_var (Tensor[b,c,c,w,h]) : covariance matrix of prediction.
        n_var (Tensor[b,c,c,w,h])  : estimation of noise level.
    '''
    def __call__(self, input_data, model_output, data, module):
        target_noisy = data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']
        x_mean, mu_var, n_var = model_output[0], model_output[1], model_output[2]
        x_mean = x_mean.detach()

        # transform the shape of tensors
        b, c, w, h = x_mean.shape
        mu_var = mu_var.permute(0,3,4,1,2) # b,w,h,c,c
        n_var = n_var.permute(0,3,4,1,2)   # b,w,h,c,c
        epsI = eps * torch.eye(c, device=x_mean.device).repeat(b,w,h,1,1)

        y_muy = (target_noisy - x_mean).permute(0,2,3,1).unsqueeze(-1) # b,w,h,c,1

        # first term in paper
        loss = torch.matmul(torch.transpose(y_muy,3,4), torch.inverse(mu_var + n_var + epsI))
        loss = torch.matmul(loss, y_muy) # b,w,h,1,1
        loss = loss.squeeze(-1).squeeze(-1) # b,w,h

        # second term in paper
        loss += torch.log(torch.clamp(torch.det(mu_var + n_var), eps)) # b,w,h
        # loss += torch.log(torch.det(mu_var + n_var)) # b,w,h

        # divide 2
        loss[loss>1e+5] = 0
        loss = loss.mean() /2

        return loss

@regist_loss
class MAP_DBSN():
    '''
    MAP loss for gaussian likelihood in DBSN paper
    model output should be as following shape.
        x_mean (Tensor[b,c,w,h])  : mean of prediction.
        mu_var (Tensor[b,c,c,w,h]) : covariance matrix of difference between prediction and original signal.
        n_var (Tensor[b,c,c,w,h]) : covariance matrix of noise.
    '''
    def __call__(self, input_data, model_output, data, module):
        target_noisy = data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']
        x_mean, mu_var, n_var = model_output
        x_mean = x_mean.detach()

        # transform the shape of tensors
        b,c,w,h = x_mean.shape
        mu_var = mu_var.permute(0,3,4,1,2) # b,w,h,c,c
        n_var = n_var.permute(0,3,4,1,2)   # b,w,h,c,c
        epsI = eps * torch.eye(c, device=x_mean.device).repeat(b,w,h,1,1)

        y_muy = (target_noisy - x_mean).permute(0,2,3,1).unsqueeze(-1) # b,w,h,c,1

        # first term in paper
        loss = torch.matmul(torch.transpose(y_muy,3,4), torch.inverse(n_var + mu_var + epsI))
        loss = torch.matmul(loss, y_muy) # b,w,h,1,1
        loss = loss.squeeze(-1).squeeze(-1) # b,w,h
        
        # second term in paper (log)
        loss += torch.log(torch.clamp(torch.det(n_var), eps)) # b,w,h

        # third term in paper (trace)
        loss += torch.diagonal(torch.matmul(torch.inverse(n_var + epsI), mu_var), dim1=-2, dim2=-1).sum(-1)
        
        # divide with 2
        loss = loss.mean() / 2

        return loss

# =================== #
#     Fusion loss     #
# =================== #

@regist_loss
class TV():
    def __call__(self, input_data, model_output, data, module):
        if type(model_output) is tuple: 
            output = model_output[4]
        else: 
            output = model_output

        return torch.sum(torch.abs(output[:,:,:,:-1] - output[:,:,:,1:])) + \
                torch.sum(torch.abs(output[:,:,:-1,:] - output[:,:,1:,:]))

@regist_loss
class self_L2_fusion():
    def __call__(self, input_data, model_output, data, module):
        if type(model_output) is tuple: 
            output = model_output[4]
        else: 
            output = model_output

        target_noisy = data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']

        return F.mse_loss(output * data['mask'], target_noisy * data['mask'])

# =================== #
#       Mu loss       #
# =================== #

@regist_loss
class mu_log_det():
    def __call__(self, input_data, model_output, data, module):
        return torch.log(torch.clamp(torch.det(model_output[1].permute(0,3,4,1,2)), eps)).mean()

@regist_loss
class mu_det():
    def __call__(self, input_data, model_output, data, module):
        return torch.det(model_output[1].permute(0,3,4,1,2)).mean()

@regist_loss
class top_singular_mean():
    def __call__(self, input_data, model_output, data, module):
        mu_var = model_output[1].permute(0,3,4,1,2) # b,w,h,c,c
        b,w,h,c,c = mu_var.shape
        _, s, _ = torch.svd(mu_var)

        return s[:,:,:,0].mean()

@regist_loss
class zero_singular():
    def __call__(self, input_data, model_output, data, module):
        mu_var = model_output[1].permute(0,3,4,1,2) # b,w,h,c,c
        b,w,h,c,c = mu_var.shape
        _, s, _ = torch.svd(mu_var)

        return F.mse_loss(s[:,:,:,0], torch.zeros_like(s[:,:,:,0]))

@regist_loss
class one_singular():
    def __call__(self, input_data, model_output, data, module):
        mu_var = model_output[1].permute(0,3,4,1,2) # b,w,h,c,c
        # b,w,h,c,c = mu_var.shape
        _, s, _ = torch.svd(mu_var)

        return F.mse_loss(s[:,:,:,1]/s[:,:,:,0], torch.zeros_like(s[:,:,:,0]))

@regist_loss
class two_singular():
    def __call__(self, input_data, model_output, data, module):
        mu_var = model_output[1].permute(0,3,4,1,2) # b,w,h,c,c
        b,w,h,c,c = mu_var.shape
        _, s, _ = torch.svd(mu_var)

        return F.mse_loss(s[:,:,:,2], torch.zeros_like(s[:,:,:,0]))

# =================== #
#      Noise loss     #
# =================== #

@regist_loss
class nlf_tv():
    def __call__(self, input_data, model_output, data, module):
        n_var = torch.diagonal(model_output[2], dim1=1, dim2=2)

        return 0.5 * torch.mean(torch.abs(n_var[:,:,:,:-1] - n_var[:,:,:,1:])) + \
                0.5 * torch.mean(torch.abs(n_var[:,:,:-1,:] - n_var[:,:,1:,:]))

@regist_loss
class nlfc_l2():
    def __call__(self, input_data, model_output, data, module):
        n_var = model_output[2] # b,c,c,w,h
        b,c,c,w,h = n_var.shape

        n_var_input = torch.diagonal(n_var, dim1=1, dim2=2)

        n_var_target = torch.square(module['denoiser'].nlf_est(input_data[0]))
        n_var_target = n_var_target.view(b,1,1,1).expand(b,w,h,c)

        return F.mse_loss(n_var_input, n_var_target)

@regist_loss
class nlfc_l1():
    def __call__(self, input_data, model_output, data, module):
        n_var = model_output[2] # b,c,c,w,h
        b,c,c,w,h = n_var.shape

        n_var_input = torch.diagonal(n_var, dim1=1, dim2=2).mean(-1)

        n_var_target = torch.square(module['denoiser'].nlf_est(input_data[0]))
        n_var_target = n_var_target.view(b,1,1).expand(b,w,h)

        return F.l1_loss(n_var_input, n_var_target)

@regist_loss
class nlf_l2():
    def __call__(self, input_data, model_output, data, module):
        n_var = model_output[2] # b,c,c,w,h
        b,c,c,w,h = n_var.shape

        n_var_input = torch.diagonal(n_var, dim1=1, dim2=2).mean(-1)

        n_var_target = torch.square(module['denoiser'].nlf_est(input_data[0]))
        n_var_target = n_var_target.view(b,1,1).expand(b,w,h)

        return F.mse_loss(n_var_input, n_var_target)

@regist_loss
class nlf_l1():
    def __call__(self, input_data, model_output, data, module):
        n_var = model_output[2] # b,c,c,w,h
        b,c,c,w,h = n_var.shape

        n_var_input = torch.diagonal(n_var, dim1=1, dim2=2).mean(-1)

        n_var_target = torch.square(module['denoiser'].nlf_est(input_data[0]))
        n_var_target = n_var_target.view(b,1,1).expand(b,w,h)

        return F.l1_loss(n_var_input, n_var_target)

@regist_loss
class nlf_mean_l2():
    def __call__(self, input_data, model_output, data, module):
        n_var = model_output[2] # b,c,c,w,h
        
        # each variance of channel are calculated for total variance of noise.
        n_var_input = torch.diagonal(n_var.mean((-1,-2)), dim1=1, dim2=2).mean(-1)

        # estimate noise variance as target
        n_var_target = torch.square(module['denoiser'].nlf_est(input_data[0]))

        return F.mse_loss(n_var_input, n_var_target)

@regist_loss
class nlf_mean_l1():
    def __call__(self, input_data, model_output, data, module):
        n_var = model_output[2] # b,c,c,w,h
        
        # each variance of channel are calculated for total variance of noise.
        n_var_input = torch.diagonal(n_var.mean((-1,-2)), dim1=1, dim2=2).mean(-1)

        # estimate noise variance as target
        n_var_target = torch.square(module['denoiser'].nlf_est(input_data[0]))

        return F.l1_loss(n_var_input, n_var_target)

@regist_loss
class neg_nlf_det():
    def __call__(self, input_data, model_output, data, module):
        return -torch.log(torch.clamp(torch.det(model_output[2].permute(0,3,4,1,2)), eps)).mean()

@regist_loss
class neg_nlf_log_dig():
    def __call__(self, input_data, model_output, data, module):
        return -torch.log(torch.diagonal(model_output[2], dim1=-4, dim2=-3).mean())

@regist_loss
class neg_nlf_dig():
    def __call__(self, input_data, model_output, data, module):
        return -torch.sqrt(torch.diagonal(model_output[2], dim1=-4, dim2=-3)).mean()

@regist_loss
class nlf_si_syn():
    def __call__(self, input_data, model_output, data, module):
        target_noisy = data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']

        alphas = torch.var(target_noisy, dim=(2,3)).sum(1)/3
        betas  = torch.var(target_noisy.sum(1)/3, dim=(1,2))

        target_vars = 3/2*(alphas-betas)

        return F.mse_loss(torch.diagonal(model_output[2], dim1=-4, dim2=-3).mean((1,2,3)), target_vars)

# =================== #
#     Display loss    #
# =================== #

@regist_loss
class n_var():
    def __call__(self, input_data, model_output, data, module):
        return torch.diagonal(model_output[2], dim1=-4, dim2=-3).mean()

@regist_loss
class mu_var():
    def __call__(self, input_data, model_output, data, module):
        return torch.diagonal(model_output[1], dim1=-4, dim2=-3).mean()

@regist_loss
class n_singular():
    def __call__(self, input_data, model_output, data, module):
        n_var = model_output[2].permute(0,3,4,1,2) # b,w,h,c,c
        _, s, _ = torch.svd(n_var)

        return s.mean()

@regist_loss
class mu_singular():
    def __call__(self, input_data, model_output, data, module):
        mu_var = model_output[1].permute(0,3,4,1,2) # b,w,h,c,c
        _, s, _ = torch.svd(mu_var)

        return s[:,:,:,0].mean()



# =================== #
#    util functions   #
# =================== #

def nlf_est(x, real=True, window_size=11, filter_type='gau', sigma=3., gamma=0.5):
    b,c,w,h = x.shape

    def do_var(x):
        return variance_conv2d(x, window_size, filter_type=filter_type, sigma=sigma, padd=False)

    alpha_map = do_var(x).sum(1, keepdim=True)/3
    beta_map = do_var(x.sum(1, keepdim=True)/3) 

    weight = do_var(x-x[:, [2,1,0]]).sum(1, keepdim=True)/3
    w_sum = torch.clamp(weight.sum((-1,-2,-3)), eps)
    weight = torch.exp(-gamma * w*h * weight / w_sum.view(b,1,1,1))
    w_sum = torch.clamp(weight.sum((-1,-2,-3)), eps)

    if real:
        nlf = 9/4*(weight*(alpha_map-beta_map)).sum((-1,-2,-3)) / w_sum
        nlf = torch.nan_to_num(nlf, nan=eps)
        return torch.sqrt(torch.clamp(nlf, eps))
    else:
        nlf = 3/2*(weight*(alpha_map-beta_map)).sum((-1,-2,-3)) / w_sum
        nlf = torch.nan_to_num(nlf, nan=eps)
        return torch.sqrt(torch.clamp(nlf, eps))