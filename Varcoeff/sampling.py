import numpy as np
import torch

import config


def rho_1_sampler(n, d=None, l=None):
    if d is None:
        d = config.dim
    if l is None:
        l = config.L
    sample = l * torch.rand(n, d) - l / 2
    return sample.to(config.DEVICE)


def rho_bdry_sampler(n, d=None, l=None):
    if d is None:
        d = config.dim
    if l is None:
        l = config.L
    boundary_coord_randint = torch.tensor(
        np.random.randint(2, size=n) * l - l / 2, dtype=torch.float32
    ).to(config.DEVICE)
    index_randint = np.random.randint(d, size=n)
    sample = rho_1_sampler(n, d, l)
    sample[np.arange(n), index_randint] = boundary_coord_randint
    return sample.to(config.DEVICE)


def rho_bdry_sampler_X_bdd_norm(n, d=None, l=None): # used for computing X boundary norm
    if d is None:
        d = config.dim
    if l is None:
        l = config.L
    boundary_coord_randint = torch.tensor(
        np.random.randint(2, size=n) * l - l / 2, dtype=torch.float32
    ).to(config.DEVICE)
    index_randint = np.random.randint(d, size=n)
    sample = rho_1_sampler(n, d, l)
    sample[np.arange(n), index_randint] = boundary_coord_randint
    return sample.to(config.DEVICE), index_randint
