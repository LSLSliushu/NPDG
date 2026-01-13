import torch

import config
import pde
import sampling
from autograd_utils import gradient_nn, v_compute_laplacian



# PDHG loss with no boundary error
def PDHG_loss_without_bd(net_u, net_phi, in_samples):
    V_x = pde.V(in_samples)
    phi_x = net_phi(in_samples)
    grad_u = gradient_nn(net_u, in_samples)
    grad_u_sqr = torch.sum(grad_u * grad_u, -1).unsqueeze(-1)
    grad_phi = gradient_nn(net_phi, in_samples)
    loss = (torch.sum(grad_u * grad_phi, -1).unsqueeze(-1)).mean() + ((V_x + 1/2 * grad_u_sqr) * phi_x).mean()
    return loss


# PDHG loss
def PDHG_loss(net_u, net_phi, net_psi, in_samples, bd_samples, bd_lambda):

    # loss on inner region
    V_x = pde.V(in_samples)
    phi_x = net_phi(in_samples)
    grad_u = gradient_nn(net_u, in_samples)
    grad_u_sqr = torch.sum(grad_u * grad_u, -1).unsqueeze(-1)
    grad_phi = gradient_nn(net_phi, in_samples)
    in_loss = (torch.sum(grad_u * grad_phi, -1)).mean() + ((V_x + 1/2 * grad_u_sqr) * phi_x).mean()

    # loss on boundary
    u_bdx = net_u(bd_samples)
    u_star_bdx = pde.u_real(bd_samples) #equals 0
    net_psi_bdx = net_psi(bd_samples)
    bd_loss = ((u_bdx - u_star_bdx) * net_psi_bdx).mean()

    return in_loss + bd_lambda * bd_loss


# PDHG loss with extrapolation, i.e., replace phi in PDHG_loss by (1+\omega) * phi_k+1 - \omega * phi_k
#                                     replace psi in PDHG_loss by (1+\omega) * psi_k+1 - \omega * psi_k
def PDHG_loss_with_extraplt(net_u, net_phi_1, net_phi_0, net_psi_1, net_psi_0, in_samples, bd_samples, bd_lambda, omega):

    # loss on inner region
    V_x = pde.V(in_samples)

    grad_u = gradient_nn(net_u, in_samples)
    grad_u_sqr = torch.sum(grad_u * grad_u, -1).unsqueeze(-1)
    grad_phi_0 = gradient_nn(net_phi_0, in_samples)
    grad_phi_1 = gradient_nn(net_phi_1, in_samples)
    grad_tilde_phi = grad_phi_1 + omega * (grad_phi_1 - grad_phi_0)

    phi_0_x = net_phi_0(in_samples)
    phi_1_x = net_phi_1(in_samples)
    tilde_phi_x = phi_1_x + omega * (phi_1_x - phi_0_x)

    in_loss = (torch.sum(grad_u * grad_tilde_phi, -1).unsqueeze(-1)).mean() + ((V_x + 1/2 * grad_u_sqr) * tilde_phi_x).mean()

    # loss on boundary
    u_bdx = net_u(bd_samples)
    u_star_bdx = pde.u_real(bd_samples) #equals 0
    net_psi_0_bdx = net_psi_0(bd_samples)
    net_psi_1_bdx = net_psi_1(bd_samples)
    tilde_psi_bdx = net_psi_1_bdx + omega * (net_psi_1_bdx - net_psi_0_bdx)

    bd_loss = ((u_bdx-u_star_bdx)*tilde_psi_bdx).mean()

    return in_loss + bd_lambda * bd_loss


def L2_norm_sq_phi(net_phi, samples):
    phi_samples = net_phi(samples)
    norm_sq = (phi_samples * phi_samples).mean()
    return norm_sq


def L2_norm_sq_Lap_phi(net_phi, samples):
    lap_phi = v_compute_laplacian(net_phi, samples)
    norm_sq = torch.sum(lap_phi * lap_phi, -1).mean()
    return norm_sq


def L2_norm_sq_nabla_phi(net_phi, samples):
    grad_phi = gradient_nn(net_phi, samples)
    norm_sq = torch.sum(grad_phi * grad_phi, -1).mean()
    # print(grad_phi.size())
    return norm_sq


def L2_norm_sq_psi(net_psi, samples):
    psi_samples = net_psi(samples)
    norm_sq = (psi_samples * psi_samples).mean()
    return norm_sq


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


def bdryloss_use_samples_resampling(net_u, N):
    L = 1.0
    bd_samples = sampling.rho_bdry_sampler(N)
    real_u = pde.u_real(bd_samples)
    num_u = net_u(bd_samples)
    diff_u_real = num_u - real_u
    loss = (diff_u_real * diff_u_real).mean()
    return loss


def PINN_Loss(net_u, N):
    samples = sampling.rho_1_sampler(N)
    return PINN_Loss_use_samples(net_u, samples)


def PINN_Loss_use_samples(net_u, samples):
    V_x = pde.V(samples)
    nabla_u_x = gradient_nn(net_u, samples)
    Lap_u_x = v_compute_laplacian(net_u, samples)
    residual = -config.alpha * 0.5 * torch.sum(nabla_u_x * nabla_u_x, -1).unsqueeze(-1) + V_x - Lap_u_x
    return (residual * residual).mean()

