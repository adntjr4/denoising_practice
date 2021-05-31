import os, sys, argparse, cv2
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from src.datahandler import get_dataset_object
from src.util.util import pixel_shuffle_up_sampling, tensor2np

def nlf_est():
    


dataset = get_dataset_object('Synthesized_CBSD432_25')(crop_size=(128,128))

# noisy_image = torch.randn((3,10,10))

data = dataset.__getitem__(5)

noisy_image = data['real_noisy']
#print('get var: ', torch.var(noisy_image-clean_image))
#print('get var: ', [torch.var(noisy_image[i]-clean_image[i]) for i in range(3)])

alpha = sum([torch.var(noisy_image[i]) for i in range(3)])/3
beta  = torch.var(noisy_image.sum(0)/3)

print(3*(alpha-beta)/2)
