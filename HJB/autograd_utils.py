import torch
import torch.autograd as autograd
from torch.func import hessian, vmap

import config


def gradient_nn(network, x):
    input_variable = autograd.Variable(x, requires_grad=True)
    output_value = network(input_variable)
    gradients_x = autograd.grad(
        outputs=output_value,
        inputs=input_variable,
        grad_outputs=torch.ones(output_value.size()).to(config.DEVICE),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return gradients_x


def v_compute_laplacian(net, samples):
    def compute_laplacian(x):
        hessian_net = hessian(net, argnums=0)(x)
        laplacian_net = hessian_net.diagonal(0, -2, -1)
        return torch.sum(laplacian_net, -1)

    return vmap(compute_laplacian)(samples)
