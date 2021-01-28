import torch.nn as nn

m = nn.Conv2d(16, 33, 3, stride=2)

print(type(m.kernel_size[0]))