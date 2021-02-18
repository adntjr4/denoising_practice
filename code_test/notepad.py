import torch

t = torch.randn((2,4))

print(t)
print(t[:,1:])
print(t[:,:-1])