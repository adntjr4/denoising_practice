import torch
import torch.nn as nn
import torch.nn.functional as F

from .single_loss import regist_loss

eps = 1e-6

@regist_loss
class self_L1():
    def __call__(self, input_data, model_output, data, model):
        if type(model_output) is tuple: 
            output = model_output[0]
        else: 
            output = model_output

        target_noisy = data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']

        return F.l1_loss(output * data['mask'], target_noisy * data['mask'])

@regist_loss
class self_L2():
    def __call__(self, input_data, model_output, data, model):
        if type(model_output) is tuple: 
            output = model_output[0]
        else: 
            output = model_output

        target_noisy = data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']

        return F.mse_loss(output * data['mask'], target_noisy * data['mask'])

@regist_loss
class self_huber():
    def __call__(self, input_data, model_output, data, model):
        if type(model_output) is tuple: 
            output = model_output[0]
        else: 
            output = model_output

        target_noisy = data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']

        return F.smooth_l1_loss(output * data['mask'], target_noisy * data['mask'], beta=1)
        
@regist_loss
class self_Gau_likelihood_scalar():
    '''
    MAP loss for gaussian likelihood from Laine et al.
    model output should be as following shape.
        x_mean (Tensor[b,c,w,h])  : mean of prediction.
        x_var (Tensor[b,c,c,w,h]) : covariance matrix of prediction.
        n_sigma (Tensor[b])   : estimation of noise level.
    '''
    def __call__(self, input_data, model_output, data, model):
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
class self_Gau_likelihood():
    '''
    MAP loss for gaussian likelihood from Laine et al.
    model output should be as following shape.
        x_mean (Tensor[b,c,w,h])   : mean of prediction.
        mu_var (Tensor[b,c,c,w,h]) : covariance matrix of prediction.
        n_var (Tensor[b,c,c,w,h])  : estimation of noise level.
    '''
    def __call__(self, input_data, model_output, data, model):
        target_noisy = data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']
        x_mean, mu_var, n_var = model_output
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
        
        # divide with 2
        loss = loss.mean() / 2

        return loss

@regist_loss
class self_Gau_likelihood_DBSN():
    '''
    MAP loss for gaussian likelihood in DBSN paper
    model output should be as following shape.
        x_mean (Tensor[b,c,w,h])  : mean of prediction.
        mu_var (Tensor[b,c,c,w,h]) : covariance matrix of difference between prediction and original signal.
        n_var (Tensor[b,c,c,w,h]) : covariance matrix of noise.
    '''
    def __call__(self, input_data, model_output, data, model):
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

@regist_loss
class neg_nlf_det():
    def __call__(self, input_data, model_output, data, model):
        return -torch.log(torch.clamp(torch.det(model_output[2].permute(0,3,4,1,2)), eps)).mean()

@regist_loss
class neg_nlf_mean():
    def __call__(self, input_data, model_output, data, model):
        return -torch.diagonal(model_output[2], dim1=-4, dim2=-3).mean()

@regist_loss
class nlf_si_syn():
    def __call__(self, input_data, model_output, data, model):
        target_noisy = data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']

        alphas = torch.var(target_noisy, dim=(2,3)).sum(1)/3
        betas  = torch.var(target_noisy.sum(1)/3, dim=(1,2))

        target_vars = 3/2*(alphas-betas)

        return F.mse_loss(torch.diagonal(model_output[2], dim1=-4, dim2=-3).mean((1,2,3)), target_vars)

@regist_loss
class n_var():
    def __call__(self, input_data, model_output, data, model):
        return torch.diagonal(model_output[2].detach().clone(), dim1=-4, dim2=-3).mean()

@regist_loss
class mu_var():
    def __call__(self, input_data, model_output, data, model):
        return torch.diagonal(model_output[1].detach().clone(), dim1=-4, dim2=-3).mean()

