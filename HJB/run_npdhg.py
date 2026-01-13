import os
import torch
import config
from solvers.npdhg import PDHG_solver_extrapolation_in_func_space_Bdry_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.set_device(device)

    save_path = os.getcwd()

    iter = 10000
    phi_psi_iter = 1
    u_iter = 1

    R = 3.0
    config.set_radius(R)

    N_r = 4000
    N_b = 40 * config.dim 

    minres_max_iter = 800
    minres_tol = 1e-4

    network_length = 4
    hidden_dimension_net_u = 256
    hidden_dimension_net_phi = 256
    hidden_dimension_net_psi = 128

    omega = 1.0
    epsilon = 1

    plot_period = 500
    N_plot = 100
    flag_plot_real = True

    precond_type = "MpMd_nabla"

    bd_lambda = 10

    PDHG_solver_extrapolation_in_func_space_Bdry_loss(
        device,
        save_path,
        R,
        N_r,
        N_b,
        minres_max_iter,
        minres_tol,
        network_length,
        hidden_dimension_net_u,
        hidden_dimension_net_phi,
        hidden_dimension_net_psi,
        False,
        iter,
        phi_psi_iter,
        u_iter,
        omega,
        epsilon,
        plot_period,
        N_plot,
        flag_plot_real,
        precond_type,
        bd_lambda,
        tau_u=0.5 * 1e-1,
        tau_phi=0.95 * 1e-1,
        tau_psi=0.95 * 1e-1,
    )


if __name__ == "__main__":
    main()
