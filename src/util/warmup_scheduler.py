import torch
import torch.optim as optim


class WarmupLRScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_iter=1000):
        self.warmup_iter = warmup_iter

        super().__init__(optimizer)

    def get_lr(self):
        assert self.last_epoch <= self.warmup_iter, 'warmup update number should be less than warmup_iter'
        return [base_lr * (self.last_epoch) / self.warmup_iter for base_lr in self.base_lrs]


if __name__ == "__main__":
    a = torch.Tensor([0])
    b = torch.Tensor([0])
    init_lr = 0.1
    otm = optim.SGD([a,b], init_lr)
    warmup_sd = WarmupLRScheduler(otm, 100)

    # warmup
    for i in range(100):
        for param_group in otm.param_groups:
            print(param_group['lr'])
        print(warmup_sd.get_lr())
        warmup_sd.step()
    
    # train
    for i in range(10):
        for param_group in otm.param_groups:
            print(param_group['lr'])
