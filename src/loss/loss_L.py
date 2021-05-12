import torch
import torch.nn as nn
import torch.nn.functional as F

from .single_loss import regist_loss

@regist_loss
class L1():
    def __call__(self, input_data, model_output, data, model):
        if type(model_output) is tuple: output = model_output[0]
        else: output = model_output
        return F.l1_loss(output, data['clean'])

@regist_loss
class L2():
    def __call__(self, input_data, model_output, data, model):
        if type(model_output) is tuple: output = model_output[0]
        else: output = model_output
        return F.l2_loss(output, data['clean'])
        

