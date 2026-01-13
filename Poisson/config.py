import math
import numpy as np
import torch


dim = 50
L = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_device(device):
    global DEVICE
    DEVICE = device


def set_length(length):
    global L
    L = length


u_real_norm = np.sqrt(dim / 2 + 4 * dim * (dim - 1) / (math.pi**2))
u_real_H1norm = np.sqrt(math.pi**2 * dim / 8)
