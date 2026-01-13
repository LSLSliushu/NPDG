import torch

import config
import pde
import sampling
from autograd_utils import gradient_nn, v_compute_laplacian


def Loss_2(net_u, N):
    bd_samples = sampling.rho_bdry_sampler(N)
    real_u = pde.u_real(bd_samples)
    num_u = net_u(bd_samples)
    diff_u_real = num_u - real_u
    return (diff_u_real * diff_u_real).mean()


def Loss_2_use_samples(net_u, bd_samples):
    real_u = pde.u_real(bd_samples)
    num_u = net_u(bd_samples)
    diff_u_real = num_u - real_u
    return (diff_u_real * diff_u_real).mean()


def bdryloss_use_samples(net_u, bd_samples):
    real_u = pde.u_real(bd_samples)
    num_u = net_u(bd_samples)
    diff_u_real = num_u - real_u
    return (diff_u_real * diff_u_real).mean()


def PINN_Loss_use_samples(net_u, samples):
    lap_u = v_compute_laplacian(net_u, samples)
    f_x = pde.f(samples)
    residual = lap_u + f_x
    return (residual * residual).mean()


def PINN_Loss(net_u, N):
    samples = sampling.rho_1_sampler(N)
    return PINN_Loss_use_samples(net_u, samples)


def PINN_Loss_deterministic(net_u, N):
    samples = sampling.deterministic_sampler(config.L, N)
    return PINN_Loss_use_samples(net_u, samples)


def Deep_Ritz_loss(net_u, N, N_bd, bd_lambda):
    samples = sampling.rho_1_sampler(N)
    f_x = pde.f(samples)
    u_x = net_u(samples)
    nabla_u_x = gradient_nn(net_u, samples)
    loss_inner = 0.5 * (torch.sum(nabla_u_x * nabla_u_x, -1).unsqueeze(-1)).mean() - (
        f_x * u_x
    ).mean()

    bd_samples = sampling.rho_bdry_sampler(N_bd)
    bd_real_u_y = pde.u_real(bd_samples)
    bd_u_y = net_u(bd_samples)
    diff_bd_u_ureal = bd_real_u_y - bd_u_y
    loss_bd = (torch.sum(diff_bd_u_ureal * diff_bd_u_ureal, -1)).mean()

    return loss_inner + bd_lambda * loss_bd


def PDHG_loss_without_bd(net_u, net_phi, in_samples):
    grad_u = gradient_nn(net_u, in_samples)
    grad_phi = gradient_nn(net_phi, in_samples)
    f_x = pde.f(in_samples)
    phi_x = net_phi(in_samples)
    in_loss = (torch.sum(grad_u * grad_phi, -1).unsqueeze(-1)).mean() - (f_x * phi_x).mean()
    return in_loss


def PDHG_loss_without_bd_resampling(net_u, net_phi, N):
    samples = sampling.rho_1_sampler(N)
    return PDHG_loss_without_bd(net_u, net_phi, samples)


def PDHG_loss(net_u, net_phi, net_psi, in_samples, bd_samples, bd_lambda):
    grad_u = gradient_nn(net_u, in_samples)
    grad_phi = gradient_nn(net_phi, in_samples)
    f_x = pde.f(in_samples)
    phi_x = net_phi(in_samples)
    in_loss = (torch.sum(grad_u * grad_phi, -1).unsqueeze(-1)).mean() - (f_x * phi_x).mean()

    u_bdx = net_u(bd_samples)
    u_star_bdx = pde.u_real(bd_samples)
    net_psi_bdx = net_psi(bd_samples)
    bd_loss = ((u_bdx - u_star_bdx) * net_psi_bdx).mean()

    return in_loss + bd_lambda * bd_loss


def PDHG_loss_with_extraplt(
    net_u,
    net_phi_1,
    net_phi_0,
    net_psi_1,
    net_psi_0,
    in_samples,
    bd_samples,
    bd_lambda,
    omega,
):
    grad_u = gradient_nn(net_u, in_samples)
    grad_phi_0 = gradient_nn(net_phi_0, in_samples)
    grad_phi_1 = gradient_nn(net_phi_1, in_samples)
    grad_tilde_phi = grad_phi_1 + omega * (grad_phi_1 - grad_phi_0)

    f_x = pde.f(in_samples)
    phi_0_x = net_phi_0(in_samples)
    phi_1_x = net_phi_1(in_samples)
    tilde_phi_x = phi_1_x + omega * (phi_1_x - phi_0_x)

    in_loss = (torch.sum(grad_u * grad_tilde_phi, -1).unsqueeze(-1)).mean() - (f_x * tilde_phi_x).mean()

    u_bdx = net_u(bd_samples)
    u_star_bdx = pde.u_real(bd_samples)
    net_psi_0_bdx = net_psi_0(bd_samples)
    net_psi_1_bdx = net_psi_1(bd_samples)
    tilde_psi_bdx = net_psi_1_bdx + omega * (net_psi_1_bdx - net_psi_0_bdx)

    bd_loss = ((u_bdx - u_star_bdx) * tilde_psi_bdx).mean()

    return in_loss + bd_lambda * bd_loss


def L2_norm_sq_phi(net_phi, samples):
    phi_samples = net_phi(samples)
    return (phi_samples * phi_samples).mean()


def L2_norm_sq_phi_resampling(net_phi, N):
    samples = sampling.rho_1_sampler(N)
    return L2_norm_sq_phi(net_phi, samples)


def L2_norm_sq_Lap_phi(net_phi, samples):
    lap_phi = v_compute_laplacian(net_phi, samples)
    return torch.sum(lap_phi * lap_phi, 1).mean()


def L2_norm_sq_nabla_phi(net_phi, samples):
    grad_phi = gradient_nn(net_phi, samples)
    return torch.sum(grad_phi * grad_phi, 1).mean()


def L2_norm_sq_psi(net_psi, samples):
    psi_samples = net_psi(samples)
    return (psi_samples * psi_samples).mean()
