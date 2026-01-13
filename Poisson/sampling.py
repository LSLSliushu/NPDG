import math

import numpy as np
import torch

import config


def rho_1_sampler(n, d=None, l=None):
    if d is None:
        d = config.dim
    if l is None:
        l = config.L
    sample = l * torch.rand(n, d)
    return sample.to(config.DEVICE)


def rho_bdry_sampler(n, d=None, l=None):
    if d is None:
        d = config.dim
    if l is None:
        l = config.L
    boundary_coord = torch.randint(0, 2, (n,), device=config.DEVICE, dtype=torch.float32) * l
    index_randint = torch.randint(0, d, (n,), device=config.DEVICE)
    sample = rho_1_sampler(n, d, l)
    sample[torch.arange(n, device=config.DEVICE), index_randint] = boundary_coord
    return sample


def deterministic_sampler(L, N): # not used currently
    square_list = []
    M = max(math.ceil(L) * N, 2)
    for i in range(1, M - 1):
        x = L / (M - 1) * i
        for j in range(1, M - 1):
            y = L / (M - 1) * j
            square_list.append([x, y])
    return torch.tensor(square_list, dtype=torch.float32, device=config.DEVICE)


def deterministic_bdry_sampler(L, N): # not used currently
    a_0 = 0
    b_0 = L
    a_1 = 0
    b_1 = L

    M_0 = np.maximum(math.ceil((b_0 - a_0) * N), 1)
    M_1 = np.maximum(math.ceil((b_1 - a_1) * N), 1)

    interval_x = np.reshape(np.linspace(a_0, b_0, M_0)[0 : M_0 - 1], (M_0 - 1, 1))
    interval_x_back = np.reshape(np.linspace(b_0, a_0, M_0)[0 : M_0 - 1], (M_0 - 1, 1))
    interval_y = np.reshape(np.linspace(a_1, b_1, M_1)[0 : M_1 - 1], (M_1 - 1, 1))
    interval_y_back = np.reshape(np.linspace(b_1, a_1, M_1)[0 : M_1 - 1], (M_1 - 1, 1))

    a_0_s = a_0 * np.ones(shape=(M_0 - 1, 1))
    b_0_s = b_0 * np.ones(shape=(M_1 - 1, 1))
    a_1_s = a_1 * np.ones(shape=(M_0 - 1, 1))
    b_1_s = b_1 * np.ones(shape=(M_0 - 1, 1))

    side_0 = np.concatenate([interval_x, a_1_s], axis=1)
    side_1 = np.concatenate([b_0_s, interval_y], axis=1)
    side_2 = np.concatenate([interval_x_back, b_1_s], axis=1)
    side_3 = np.concatenate([a_0_s, interval_y_back], axis=1)

    sides = [side_0, side_1, side_2, side_3]
    tensor_float = torch.tensor(np.reshape(np.array(sides), (-1, 2)), dtype=torch.float32)
    return tensor_float.to(config.DEVICE)
