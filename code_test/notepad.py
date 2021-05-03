import torch
import torch.nn as nn
import torch.nn.functional as F


t = torch.randn(2,5)

print(t)
print(t.mean(dim=1))

t[:] = t.mean(dim=1)

print(t)
