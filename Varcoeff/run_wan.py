import os

import torch

import config
from solvers.wan import PD_adam_WAN_solver


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.set_device(device)

    save_path = os.getcwd()

    iter = 12000
    u_iter = 1
    phi_psi_iter = 2

    L = 2.0
    config.set_length(L)

    N_r = 4000
    N_b = 80 * config.dim

    if config.dim <= 20:
        network_length = 4
    else:
        network_length = 6
    hidden_dimension_net_u = 256
    hidden_dimension_net_phi = 256


    plot_period = 1000
    print_period = 100
    N_plot = 100
    chosen_dim_0 = 8
    chosen_dim_1 = 9

    if config.dim <= 10:
        tau_u = 0.5 * 1e-2
        tau_phi = 0.5 * 1e-1
    else:
        tau_u = 0.5 * 1e-3
        tau_phi = 0.5 * 1e-2

    bd_lambda = 10000

    PD_adam_WAN_solver(
        device,
        save_path,
        L,
        N_r,
        N_b,
        network_length,
        hidden_dimension_net_u,
        hidden_dimension_net_phi,
        False,
        iter,
        u_iter,
        tau_u,
        phi_psi_iter,
        tau_phi,
        plot_period,
        print_period,
        N_plot,
        chosen_dim_0,
        chosen_dim_1,
        bd_lambda,
    )


if __name__ == "__main__":
    main()
