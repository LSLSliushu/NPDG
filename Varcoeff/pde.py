import torch
import torch.autograd as autograd

import config


_LAMBDA_DIAG = torch.zeros(config.dim)
_LAMBDA_DIAG[0::2] = 1.0
_LAMBDA_DIAG[1::2] = 4.0

_INV_LAMBDA_DIAG = torch.zeros(config.dim)
_INV_LAMBDA_DIAG[0::2] = 1.0
_INV_LAMBDA_DIAG[1::2] = 1.0 / 4.0


def lambda_diag():
    return _LAMBDA_DIAG.to(config.DEVICE)


def inv_lambda_diag():
    return _INV_LAMBDA_DIAG.to(config.DEVICE)


def trace_inv_lambda():
    return torch.sum(inv_lambda_diag())


def f(x):
    nabla_sig_nabla_u = torch.sum(x * x, -1).unsqueeze(-1)
    lap_u = trace_inv_lambda()
    sig = sigma(x)
    f_value = -(nabla_sig_nabla_u + lap_u * sig)
    return f_value


def sigma(x):
    x_sqr = x * x
    x_lambda_x = (0.5 * torch.matmul(x_sqr, lambda_diag()) + 1).unsqueeze(-1)
    return x_lambda_x.to(config.DEVICE)


def u_real(x):
    x_sqr = x * x
    x_inv_lambda_x = 0.5 * torch.matmul(x_sqr, inv_lambda_diag()).unsqueeze(-1)
    return x_inv_lambda_x.to(config.DEVICE)


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
