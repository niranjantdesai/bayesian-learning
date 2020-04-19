from torch.autograd import Variable
from torch.optim import Optimizer
from torch.optim import SGD


class Langevin_SGD(Optimizer):

    def __init__(self, params, lr, weight_decay=0, nesterov=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )

        defaults = dict(lr=lr, weight_decay=weight_decay)

        super(Langevin_SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if len(p.shape) == 1 and p.shape[0] == 1:
                    p.data.add_(-group['lr'], d_p)

                else:
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)

                    unit_noise = Variable(p.data.new(p.size()).normal_())

                    p.data.add_(-group['lr'], 0.5*d_p +
                                unit_noise/group['lr']**0.5)

        return loss
