import torch


dim = 5
R = 3.0
alpha = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_device(device):
    global DEVICE
    DEVICE = device


def set_radius(radius):
    global R
    R = radius


def set_alpha(value):
    global alpha
    alpha = value


u_real_norm = 0.62851041229
u_real_H1norm = 1.22176722241
