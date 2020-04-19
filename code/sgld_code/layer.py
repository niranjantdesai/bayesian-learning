import torch
import torch.nn as nn


class Langevin_Layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Langevin_Layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weights = nn.Parameter(torch.Tensor(
            self.input_dim, self.output_dim).uniform_(-0.01, 0.01))
        self.biases = nn.Parameter(torch.Tensor(
            self.output_dim).uniform_(-0.01, 0.01))

    def forward(self, x):

        return torch.mm(x, self.weights) + self.biases
