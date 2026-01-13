import math

import torch
import torch.autograd as autograd

import config


def f(x):
    return (math.pi**2 / 4) * torch.sum(torch.sin(math.pi / 2 * x), -1).unsqueeze(-1)


def u_real(x):
    return torch.sum(torch.sin(math.pi / 2 * x), -1).unsqueeze(-1)


def nabla_u_real(x):
    x = x.to(config.DEVICE)
    input_variable = autograd.Variable(x, requires_grad=True)
    output_value = u_real(input_variable)
    gradients_x = autograd.grad(
        outputs=output_value,
        inputs=input_variable,
        grad_outputs=torch.ones(output_value.size()).to(config.DEVICE),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return gradients_x.detach()
