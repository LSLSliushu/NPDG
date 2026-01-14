import os

import torch

import config
from solvers.wan import PD_adam_solver


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.set_device(device)

    save_path = os.getcwd()

    iter = 1 # 20000
    u_iter = 2
    phi_iter = 1

    R = 3.0
    config.set_radius(R)

    N_r = 4000
    N_b = 40 * config.dim

    network_length = 4
    hidden_dimension_net_u = 256
    hidden_dimension_net_phi = 256

    plot_period = 1 # 500
    N_plot = 100

    tau_u = 0.5 * 1e-3
    tau_phi = 0.5 * 1e-2

    bd_lambda = 1000

    PD_adam_solver(
        device,
        save_path,
        R,
        N_r,
        N_b,
        network_length,
        hidden_dimension_net_u,
        hidden_dimension_net_phi,
        iter,
        u_iter,
        tau_u,
        phi_iter,
        tau_phi,
        plot_period,
        N_plot,
        bd_lambda,
    )

if __name__ == "__main__":
    main()
