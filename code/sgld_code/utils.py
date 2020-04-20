import numpy as np
import torch

from torch.autograd import Variable


def log_gaussian_loss(output, target, sigma, no_dim):
    # exponent = -0.5*(target - output)**2/sigma**2
    # log_coeff = -no_dim*torch.log(sigma)

    # return - (log_coeff + exponent).mean()

    exponent = ((target - output)/sigma)**2
    log_coeff = no_dim*torch.log(sigma)

    return (log_coeff + 0.5*exponent).mean()


def get_kl_divergence(weights, prior, varpost):
    prior_loglik = prior.loglik(weights)

    varpost_loglik = varpost.loglik(weights)
    varpost_lik = varpost_loglik.exp()

    return (varpost_lik*(varpost_loglik - prior_loglik)).sum()


class gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def loglik(self, weights):
        exponent = -0.5*(weights - self.mu)**2/self.sigma**2
        log_coeff = -0.5*(np.log(2*np.pi) + 2*np.log(self.sigma))

        return (exponent + log_coeff).sum()


def to_variable(var=(), cuda=True, volatile=False):
    out = []
    for v in var:

        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v).type(torch.FloatTensor)

        if not v.is_cuda and cuda:
            v = v.cuda()

        if not isinstance(v, Variable):
            v = Variable(v, volatile=volatile)

        out.append(v)
    return out
