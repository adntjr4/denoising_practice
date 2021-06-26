import os, sys, argparse, cv2
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

a = torch.Tensor([float('nan')])
print(a)
a = torch.nan_to_num(a, nan=1e-6)
print(a)
print(torch.sqrt(a))