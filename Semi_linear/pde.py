import math

import torch
import torch.autograd as autograd

import config


def V(x, d=None):
    if d is None:
        d = config.dim
    rho = torch.sqrt(torch.sum(x * x, -1)).unsqueeze(-1)
    sin_rho = torch.sin(math.pi/2*rho)
    cos_rho = torch.cos(math.pi/2*rho)
    V_x = -math.pi**2/8 * sin_rho*sin_rho - math.pi**2/4 * cos_rho - math.pi * (d - 1) / (2 * rho) * sin_rho
    return V_x


def u_real(x):
    rho = torch.sqrt(torch.sum(x * x, -1))
    u_real_x = torch.cos(math.pi / 2 * rho).unsqueeze(-1)
    return u_real_x.to(config.DEVICE)


def nabla_u_real(x):
    x = x.to(config.DEVICE)
    input_variable = autograd.Variable(x, requires_grad=True)
    output_value = u_real(input_variable)
    gradients_x = autograd.grad(outputs=output_value, inputs=input_variable, grad_outputs=torch.ones(output_value.size()).to(config.DEVICE), create_graph=True, retain_graph=True, only_inputs=True)[0]
    return gradients_x.detach()
