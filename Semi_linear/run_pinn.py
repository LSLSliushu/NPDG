import os

import torch

import config
from solvers.pinn import PINN_solver


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.set_device(device)

    save_path = os.getcwd()

    iter = 1 # 20000

    R = 3.0
    config.set_radius(R)

    N_r = 4000
    N_b = 40 * config.dim

    network_length = 4
    flag_init = False
    hidden_dimension_net_u = 256

    tau_u = 1e-4

    plot_period = 1  # 500
    N_plot = 100
    flag_plot_real = True

    bd_lambda = 10000

    PINN_solver(
        device,
        save_path,
        R,
        N_r,
        N_b,
        network_length,
        hidden_dimension_net_u,
        flag_init,
        iter,
        tau_u,
        plot_period,
        N_plot,
        bd_lambda,
    )


if __name__ == "__main__":
    main()
