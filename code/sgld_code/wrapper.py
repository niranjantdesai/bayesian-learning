
import torch

import sgld_code.utils as sgld_utils

from sgld_code.net import Langevin_Model
from sgld_code.optimizer import Langevin_SGD


class Langevin_Wrapper:
    def __init__(self, input_dim, output_dim, no_units, learn_rate, batch_size, no_batches, init_log_noise, weight_decay, num_hidden_layers=2):

        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.no_batches = no_batches

        self.cuda = torch.cuda.is_available()

        self.device = torch.device("cuda" if self.cuda else "cpu")

        self.network = Langevin_Model(input_dim=input_dim, output_dim=output_dim,
                                      no_units=no_units, init_log_noise=init_log_noise,
                                      num_hidden_layers=num_hidden_layers
                                      ).to(self.device)

        self.optimizer = Langevin_SGD(self.network.parameters(
        ), lr=self.learn_rate, weight_decay=weight_decay)
        self.loss_func = sgld_utils.log_gaussian_loss

    def fit(self, x, y):
        x, y = sgld_utils.to_variable(var=(x, y), cuda=self.cuda)

        # reset gradient and total loss
        self.optimizer.zero_grad()

        output = self.network(x)
        loss = self.loss_func(output, y, torch.exp(self.network.log_noise), 1)

        loss.backward()
        self.optimizer.step()

        return loss
