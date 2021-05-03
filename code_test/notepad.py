import torch
import torch.nn as nn
import torch.nn.functional as F


t = torch.Tensor(list(range(25))).view(1,1,5,5)

tp = F.pixel_unshuffle(t, 2)

print(t.shape)

print(t)
print(tp.view(1,1,3,3,2,2))
print(tp.view(1,1,3,3,2,2).permute(0,1,2,4,3,5).reshape(1,1,6,6))
