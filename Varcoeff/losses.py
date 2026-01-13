import numpy as np
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
    loss = (diff_u_real * diff_u_real).mean()

    return loss


def Loss_2_use_samples(net_u, bd_samples):
    real_u = pde.u_real(bd_samples)
    num_u = net_u(bd_samples)
    diff_u_real = num_u - real_u
    loss = (diff_u_real * diff_u_real).mean()

    return loss


def bdryloss_use_samples(net_u, bd_samples):
    real_u = pde.u_real(bd_samples)
    num_u = net_u(bd_samples)
    diff_u_real = num_u - real_u
    loss = (diff_u_real * diff_u_real).mean()

    return loss


def bdryloss_X_norm_use_samples(net_u, bd_samples, bdd_idx):
    real_u = pde.u_real(bd_samples)
    num_u = net_u(bd_samples)
    diff_u_real = num_u - real_u
    L2loss = (diff_u_real * diff_u_real).mean()

    N, d = bd_samples.shape
    mask = np.ones((N, d), dtype=bool)
    mask[np.arange(N), bdd_idx] = False
    nabla_u_x = gradient_nn(net_u, bd_samples)
    nabla_u_real_x = pde.nabla_u_real(bd_samples)
    projected_nabla_u = nabla_u_x[mask].reshape(N, d - 1)
    projected_nabla_u_real = nabla_u_real_x[mask].reshape(N, d - 1)
    diff_projected_nabla_u = projected_nabla_u - projected_nabla_u_real
    dotH1loss = torch.sum(diff_projected_nabla_u * diff_projected_nabla_u, -1).mean()

    return L2loss + dotH1loss


def PDHG_loss_without_bd(net_u, net_phi, in_samples):
    sigma_x = pde.sigma(in_samples)
    f_x = pde.f(in_samples)
    phi_x = net_phi(in_samples)
    grad_u = gradient_nn(net_u, in_samples)
    grad_phi = gradient_nn(net_phi, in_samples)
    loss = (sigma_x * torch.sum(grad_u * grad_phi, -1).unsqueeze(-1)).mean() - (
        f_x * phi_x
    ).mean()

    return loss


def PDHG_loss(net_u, net_phi, net_psi, in_samples, bd_samples, bdd_idx, bd_lambda):
    sigma_x = pde.sigma(in_samples)
    f_x = pde.f(in_samples)
    phi_x = net_phi(in_samples)
    grad_u = gradient_nn(net_u, in_samples)
    grad_phi = gradient_nn(net_phi, in_samples)
    in_loss = (sigma_x * torch.sum(grad_u * grad_phi, -1).unsqueeze(-1)).mean() - (
        f_x * phi_x
    ).mean()

    u_bdx = net_u(bd_samples)
    u_star_bdx = pde.u_real(bd_samples)
    net_psi_bdx = net_psi(bd_samples)
    bd_loss = ((u_bdx - u_star_bdx) * net_psi_bdx).mean()

    return in_loss + bd_lambda * bd_loss


def PDHG_loss_X_bdd_norm(net_u, net_phi, net_psi, in_samples, bd_samples, bdd_idx, bd_lambda):
    sigma_x = pde.sigma(in_samples)
    f_x = pde.f(in_samples)
    phi_x = net_phi(in_samples)
    grad_u = gradient_nn(net_u, in_samples)
    grad_phi = gradient_nn(net_phi, in_samples)
    in_loss = (sigma_x * torch.sum(grad_u * grad_phi, -1).unsqueeze(-1)).mean() - (
        f_x * phi_x
    ).mean()

    u_bdx = net_u(bd_samples)
    u_star_bdx = pde.u_real(bd_samples)
    net_psi_bdx = net_psi(bd_samples)
    L2_bd_loss = ((u_bdx - u_star_bdx) * net_psi_bdx).mean()

    N, d = bd_samples.shape
    mask = np.ones((N, d), dtype=bool)
    mask[np.arange(N), bdd_idx] = False
    nabla_u_bdx = gradient_nn(net_u, bd_samples)
    projected_nabla_u_bdx = nabla_u_bdx[mask].reshape(N, d - 1)
    nabla_u_real_bdx = pde.nabla_u_real(bd_samples)
    projected_nabla_u_real_bdx = nabla_u_real_bdx[mask].reshape(N, d - 1)

    nabla_psi_bdx = gradient_nn(net_psi, bd_samples)
    projected_nabla_psi_bdx = nabla_psi_bdx[mask].reshape(N, d - 1)

    dotH1_bd_loss = (
        (projected_nabla_u_bdx - projected_nabla_u_real_bdx) * projected_nabla_psi_bdx
    ).mean()

    return in_loss + bd_lambda * (L2_bd_loss + dotH1_bd_loss)


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
    sigma_x = pde.sigma(in_samples)
    f_x = pde.f(in_samples)

    grad_u = gradient_nn(net_u, in_samples)
    grad_phi_0 = gradient_nn(net_phi_0, in_samples)
    grad_phi_1 = gradient_nn(net_phi_1, in_samples)
    grad_tilde_phi = grad_phi_1 + omega * (grad_phi_1 - grad_phi_0)

    phi_0_x = net_phi_0(in_samples)
    phi_1_x = net_phi_1(in_samples)
    tilde_phi_x = phi_1_x + omega * (phi_1_x - phi_0_x)

    in_loss = (sigma_x * torch.sum(grad_u * grad_tilde_phi, -1).unsqueeze(-1)).mean() - (
        f_x * tilde_phi_x
    ).mean()

    u_bdx = net_u(bd_samples)
    u_star_bdx = pde.u_real(bd_samples)
    net_psi_0_bdx = net_psi_0(bd_samples)
    net_psi_1_bdx = net_psi_1(bd_samples)
    tilde_psi_bdx = net_psi_1_bdx + omega * (net_psi_1_bdx - net_psi_0_bdx)

    bd_loss = ((u_bdx - u_star_bdx) * tilde_psi_bdx).mean()

    return in_loss + bd_lambda * bd_loss


def PDHG_loss_X_bdd_norm_with_extraplt(
    net_u,
    net_phi_1,
    net_phi_0,
    net_psi_1,
    net_psi_0,
    in_samples,
    bd_samples,
    bdd_idx,
    bd_lambda,
    omega,
):
    sigma_x = pde.sigma(in_samples)
    f_x = pde.f(in_samples)

    grad_u = gradient_nn(net_u, in_samples)
    grad_phi_0 = gradient_nn(net_phi_0, in_samples)
    grad_phi_1 = gradient_nn(net_phi_1, in_samples)
    grad_tilde_phi = grad_phi_1 + omega * (grad_phi_1 - grad_phi_0)

    phi_0_x = net_phi_0(in_samples)
    phi_1_x = net_phi_1(in_samples)
    tilde_phi_x = phi_1_x + omega * (phi_1_x - phi_0_x)

    in_loss = (sigma_x * torch.sum(grad_u * grad_tilde_phi, -1).unsqueeze(-1)).mean() - (
        f_x * tilde_phi_x
    ).mean()

    u_bdx = net_u(bd_samples)
    u_star_bdx = pde.u_real(bd_samples)
    net_psi_0_bdx = net_psi_0(bd_samples)
    net_psi_1_bdx = net_psi_1(bd_samples)
    tilde_psi_bdx = net_psi_1_bdx + omega * (net_psi_1_bdx - net_psi_0_bdx)

    L2_bd_loss = ((u_bdx - u_star_bdx) * tilde_psi_bdx).mean()

    N, d = bd_samples.shape
    mask = np.ones((N, d), dtype=bool)
    mask[np.arange(N), bdd_idx] = False
    nabla_u_bdx = gradient_nn(net_u, bd_samples)
    projected_nabla_u_bdx = nabla_u_bdx[mask].reshape(N, d - 1)
    nabla_u_real_bdx = pde.nabla_u_real(bd_samples)
    projected_nabla_u_real_bdx = nabla_u_real_bdx[mask].reshape(N, d - 1)

    nabla_psi1_bdx = gradient_nn(net_psi_1, bd_samples)
    nabla_psi0_bdx = gradient_nn(net_psi_0, bd_samples)
    nabla_tilde_psi_bdx = nabla_psi1_bdx + omega * (nabla_psi1_bdx - nabla_psi0_bdx)
    projected_nabla_tilde_psi_bdx = nabla_tilde_psi_bdx[mask].reshape(N, d - 1)

    dotH1_bd_loss = (
        (projected_nabla_u_bdx - projected_nabla_u_real_bdx) * projected_nabla_tilde_psi_bdx
    ).mean()

    bd_loss = L2_bd_loss + dotH1_bd_loss

    return in_loss + bd_lambda * bd_loss


def L2_norm_sq_phi(net_phi, samples):
    phi_samples = net_phi(samples)
    norm_sq = (phi_samples * phi_samples).mean()
    return norm_sq


def L2_norm_sq_Lap_phi(net_phi, samples):
    lap_phi = v_compute_laplacian(net_phi, samples)
    norm_sq = torch.sum(lap_phi * lap_phi, 1).mean()
    return norm_sq


def L2_norm_sq_nabla_phi(net_phi, samples):
    grad_phi = gradient_nn(net_phi, samples)
    norm_sq = torch.sum(grad_phi * grad_phi, 1).mean()

    return norm_sq


def L2_norm_sq_sqrt_A_nabla_phi(net_phi, samples):
    grad_phi = gradient_nn(net_phi, samples)
    sigma_x = pde.sigma(samples)
    norm_sq = (sigma_x * torch.sum(grad_phi * grad_phi, 1)).mean()

    return norm_sq


def L2_norm_sq_psi(net_psi, samples):
    psi_samples = net_psi(samples)
    norm_sq = (psi_samples * psi_samples).mean()

    return norm_sq


def X_bdd_norm_psi(net_psi, samples, bdd_idx):
    psi_samples = net_psi(samples)
    L2_norm = (psi_samples * psi_samples).mean()

    N, d = samples.shape
    mask = np.ones((N, d), dtype=bool)
    mask[np.arange(N), bdd_idx] = False
    nabla_psi_samples = gradient_nn(net_psi, samples)
    projected_nabla_psi_samples = nabla_psi_samples[mask].reshape(N, d - 1)
    dotH1_norm = torch.sum(projected_nabla_psi_samples * projected_nabla_psi_samples, -1).mean()

    return L2_norm + dotH1_norm


def Deep_Ritz_loss(net_u, N, N_bd, bd_lambda):
    samples = sampling.rho_1_sampler(N)
    sigma_x = pde.sigma(samples)
    f_x = pde.f(samples)
    u_x = net_u(samples)
    nabla_u_x = gradient_nn(net_u, samples)
    Loss_inner = (
        0.5 * (sigma_x * torch.sum(nabla_u_x * nabla_u_x, -1).unsqueeze(-1)).mean()
        - (f_x * u_x).mean()
    )
    bd_samples = sampling.rho_bdry_sampler(N_bd)
    bd_real_u_y = pde.u_real(bd_samples)
    bd_u_y = net_u(bd_samples)
    diff_bd_u_ureal = bd_real_u_y - bd_u_y
    Loss_bd = (torch.sum(diff_bd_u_ureal * diff_bd_u_ureal, -1)).mean()

    return Loss_inner + bd_lambda * Loss_bd


def PINN_Loss(net_u, N):
    samples = sampling.rho_1_sampler(N)
    nabla_sigma_x = pde.lambda_diag().repeat(N, 1) * samples
    nabla_u_x = gradient_nn(net_u, samples)
    sigma_x = pde.sigma(samples)
    lap_u_x = v_compute_laplacian(net_u, samples)
    f_x = pde.f(samples)
    residual = torch.sum(nabla_sigma_x * nabla_u_x, -1).unsqueeze(-1) + (
        sigma_x * lap_u_x
    ) + f_x
    PINNloss = (residual * residual).mean()

    return PINNloss
