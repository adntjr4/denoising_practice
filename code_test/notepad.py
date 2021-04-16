import torch


a = torch.randn(16,3,4,5,5)
b = torch.randn(16,5,1,3,4)

print(torch.diagonal(a, dim1=-2, dim2=-1).sum(-1).shape)