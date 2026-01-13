import math

import numpy as np
import torch

import config


def radial_sampler(R, N, d=None):
    if d is None:
        d = config.dim
    x0 = (torch.rand(N, 1, device=config.DEVICE) + 1e-8) / (1 + 1e-8)
    x1 = R * x0
    coeff = np.power(R, 1 - 1 / d)
    x2 = coeff * torch.pow(x1, 1 / d)
    return x2


def sphere_sampler(r, N, d=None):
    if d is None:
        d = config.dim
    x0 = torch.randn(N, d, device=config.DEVICE)
    norm_x0 = torch.sqrt(torch.sum(x0 * x0, -1)).unsqueeze(-1)
    x1 = r * x0 / (norm_x0 + 1e-8)
    return x1


def nball_sampler_new(R, N, d=None):
    if d is None:
        d = config.dim
    r = radial_sampler(R, N, d)
    w = torch.randn(N, d, device=config.DEVICE)
    norm_w = torch.sqrt(torch.sum(w * w, -1)).unsqueeze(-1)
    samples = r * w / (norm_w + 1e-8)
    return samples


def rho_1_sampler(N, d=None, R=None):
    if d is None:
        d = config.dim
    if R is None:
        R = config.R
    sample = nball_sampler_new(R, N, d)
    return sample.to(config.DEVICE)


def rho_bdry_sampler(n, d=None, R=None):
    if d is None:
        d = config.dim
    if R is None:
        R = config.R
    sample = sphere_sampler(R, n, d)
    return sample.to(config.DEVICE)
