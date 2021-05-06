import torch
import torch.nn as nn
import torch.nn.functional as F


eye = torch.eye(3).view(-1)
t = torch.Tensor([1,2,3,4])

print(t.shape)
print(eye.shape)
print(eye.repeat(4,16,16,1).shape)

tt = t * eye.repeat(4,16,16,1).permute(1,2,3,0)
tt = tt.permute(3,0,1,2)

print(tt.shape)
print(tt)
