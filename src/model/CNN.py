import torch
import torch.nn as nn

class simple_CNN(nn.Module):
    '''
    simple CNN for MNIST
    '''
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(3136, 128)
        self.fc2 = nn.Linear(128, 10)
        self.max_pool = nn.MaxPool2d(2, 2)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        t = self.act(self.layer1(x))
        t = self.max_pool(t)
        t = self.act(self.layer2(t))
        t = self.max_pool(t)
        t = self.act(self.fc1(t.view(t.size(0), -1)))
        t = self.fc2(t)
        return t

if __name__ == '__main__':
    sc = simple_CNN()
    tt = torch.randn((10, 1, 28, 28))
    oo = sc(tt)
    print(oo.shape)

