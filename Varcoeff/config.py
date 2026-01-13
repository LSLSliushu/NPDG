import numpy as np
import torch


dim = 50
lambda_0 = 1
lambda_1 = 4
L = 2.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_device(device):
    global DEVICE
    DEVICE = device


def set_length(length):
    global L
    L = length


half_dim = dim / 2
u_real_norm = np.sqrt(
    (1 / lambda_0**2 + 1 / lambda_1**2) * (5 * half_dim**2 + 4 * half_dim) / 180
    + half_dim**2 / (18 * lambda_0 * lambda_1)
)

u_real_H1norm = np.sqrt(half_dim / 3 * (1 / lambda_0 + 1 / lambda_1))
