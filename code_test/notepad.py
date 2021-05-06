import torch
import torch.nn as nn
import torch.nn.functional as F


t = torch.randn(1,1,4,4)
t = F.pad(t, (0,0,2,0))
print(t)
print(t[:,:,:-2,:])
