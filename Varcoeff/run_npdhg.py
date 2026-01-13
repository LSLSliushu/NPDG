import os

import torch

import config
from solvers.npdhg import PDHG_solver_extrapolation_in_func_space_Bdry_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.set_device(device)

    save_path = os.getcwd()

    iter = 3000
    phi_psi_iter = 1
    u_iter = 1

    L = 2.0
    config.set_length(L)
    
    if config.dim <= 20:
        N_r = 4000
    else:
        N_r = 6000
    N_b = 80 * config.dim

    minres_max_iter = 1000
    if config.dim <= 20:
        minres_tol = 0.5 * 1e-3
    else:
        minres_tol = 1e-4

    if config.dim <= 20:
        network_length = 4
    else:
        network_length = 6
    hidden_dimension_net_u = 256
    hidden_dimension_net_phi = 256
    hidden_dimension_net_psi = 128

    omega = 1.0
    epsilon = 1

    if config.dim <= 20:
        tau_u=0.1
        tau_phi=0.19
        tau_psi=0.19
    else:
        tau_u=0.05
        tau_phi=0.095
        tau_psi=0.095

    plot_period = 500
    print_period = 100
    N_plot = 100  # num_of_intervals used when making plots of function graph & error heatmap
    
    # making plots in dimension_0-dimension_1 plane
    chosen_dim_0 = 8 
    chosen_dim_1 = 9

    precond_type = "MpMd_sqrt_kappa_nabla_H1"  # tested in the paper
    # other choices:
    # "MpMd_nabla_H1"
    # "Mp_Id_Md_Laplace"
    # "MpMd_nabla"  # tested in the paper
    # "MpMd_Id"


    bd_lambda = 10

    PDHG_solver_extrapolation_in_func_space_Bdry_loss(
        device,
        save_path,
        L,
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
        print_period,
        N_plot,
        chosen_dim_0,
        chosen_dim_1,
        precond_type,
        bd_lambda,
        tau_u,
        tau_phi,
        tau_psi,
    )


if __name__ == "__main__":
    main()
