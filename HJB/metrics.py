import torch

import pde
import sampling
from autograd_utils import gradient_nn


def L2_error(net_u, N):
    samples = sampling.rho_1_sampler(N)
    real_solution = pde.u_real(samples)
    numerical = net_u(samples)
    diff = real_solution - numerical
    return torch.sqrt((diff * diff).mean())


def L2_error_with_samples(net_u, samples):
    real_solution = pde.u_real(samples)
    numerical = net_u(samples)
    diff = real_solution - numerical
    return torch.sqrt((diff * diff).mean())


def H1_error(network, N):
    samples = sampling.rho_1_sampler(N)
    nabla_nn = gradient_nn(network, samples)
    nabla_ureal = pde.nabla_u_real(samples)
    diff = nabla_nn - nabla_ureal
    return torch.sqrt(torch.sum(diff * diff, -1).mean())


def H1_error_with_samples(network, samples):
    nabla_nn = gradient_nn(network, samples)
    nabla_ureal = pde.nabla_u_real(samples)
    diff = nabla_nn - nabla_ureal
    return torch.sqrt(torch.sum(diff * diff, -1).mean())
