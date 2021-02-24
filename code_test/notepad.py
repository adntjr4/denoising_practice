import torch

t = torch.rand((5,2))

print(torch.mean(t, dim=1, keepdim=True)