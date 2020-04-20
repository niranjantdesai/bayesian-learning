
import torch

from pysgmcmc.optimizers.sgld import SGLD

import sgld_code.utils as sgld_utils

from sgld_code.net import Langevin_Model


class Langevin_Wrapper():
    def __init__(self, input_dim, output_dim, no_units, learn_rate, num_epochs, no_batches, num_hidden_layers=2, num_burn_in_steps=3000):

        self.learn_rate = learn_rate
        # self.batch_size = batch_size
        self.no_batches = no_batches

        self.cuda = torch.cuda.is_available()

        self.device = torch.device("cuda" if self.cuda else "cpu")

        self.network = Langevin_Model(input_dim=input_dim, output_dim=output_dim,
                                      no_units=no_units,
                                      num_hidden_layers=num_hidden_layers
                                      ).to(self.device)

        # self.optimizer = Langevin_SGD(self.network.parameters(
        # ), lr=self.learn_rate, weight_decay=weight_decay)
        self.optimizer = SGLD(self.network.parameters(),
                              lr=self.learn_rate,
                              num_pseudo_batches=self.no_batches,
                              num_burn_in_steps=num_burn_in_steps)
        # self.optimizer = torch.optim.SGD(
        #     self.network.parameters(), lr=self.learn_rate
        # )
        # self.loss_func = torch.nn.MSELoss(reduction='mean')
        self.loss_func = sgld_utils.log_gaussian_loss

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=int(num_epochs//6), gamma=0.5)  # to reduce by 100 over the course of optimization

    def fit(self, x, y):
        # x, y = sgld_utils.to_variable(var=(x, y), cuda=self.cuda)

        output = self.network(x)
        # loss = self.loss_func(output, y)

        # # compute the prior on parameters:
        # prior = 0.0
        # for param in self.network.net.parameters():
        #     if param.requires_grad:
        #         prior += (0.5)*torch.sum(torch.pow(param, 2))

        loss = self.loss_func(
            output, y, self.network.noise, 1)  # + 0.0*prior

        # reset gradient and total loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
