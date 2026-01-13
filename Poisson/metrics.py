import torch

import config
import pde
import sampling
from autograd_utils import gradient_nn


# Compute L2 error between the neural network solution and the real solution
def L2_error(net_u, N):
    samples = sampling.rho_1_sampler(N, config.dim)
    real_solution = pde.u_real(samples)
    numerical = net_u(samples).to(config.DEVICE)
    diff = real_solution - numerical
    return torch.sqrt((diff * diff).mean())


def L2_error_on_given_plane(net_u, given_coord, chosen_dim_0, chosen_dim_1, l, N):
    samples = sampling.rho_1_sampler(N)
    samples_given_coord_plane = given_coord * torch.ones(N, config.dim)
    samples_given_coord_plane[:, chosen_dim_0] = samples[:, chosen_dim_0]
    samples_given_coord_plane[:, chosen_dim_1] = samples[:, chosen_dim_1]
    real_solution = pde.u_real(samples_given_coord_plane)
    numerical_solution = net_u(samples_given_coord_plane).to(config.DEVICE)
    diff = real_solution - numerical_solution
    return torch.sqrt((diff * diff).mean())


# Compute H1 error (seminorm) between the neural network solution and the real solution
def H1_error(network, N):
    samples = sampling.rho_1_sampler(N)
    nabla_nn = gradient_nn(network, samples)
    nabla_ureal = pde.nabla_u_real(samples)
    diff = nabla_nn - nabla_ureal
    return torch.sqrt(torch.sum(diff * diff, -1).mean())
