import torch

t = torch.randn((2,4))

a, b = torch.split(t, 2, dim=1)

print(a.shape)
print(b.shape)