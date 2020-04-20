import torch
import torch.nn as nn

# from sgld_code.layer import Langevin_Layer


class Langevin_Model(nn.Module):
    def __init__(self, input_dim, output_dim, no_units, num_hidden_layers=2):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if num_hidden_layers < 1:
            raise AttributeError('Number of hidden layers should be > 1')

        # add the first layer
        modules = [
            nn.Linear(input_dim, no_units),
            nn.ReLU()
        ]

        # add remaining hidden layers
        for _ in range(1, num_hidden_layers):
            modules.append(
                nn.Linear(no_units, no_units),
            )

            modules.append(nn.ReLU())

        # add the output layer
        modules.append(
            nn.Linear(no_units, output_dim),
        )

        self.net = nn.Sequential(
            *modules
        )

        self.noise = nn.Parameter(torch.FloatTensor([1.0]))

    def forward(self, x):

        x = x.view(-1, self.input_dim)

        return self.net(x)
