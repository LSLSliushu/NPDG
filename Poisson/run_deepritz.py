import os

import torch

import config
from solvers.deepritz import DeepRitz_solver


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.set_device(device)

    save_path = os.getcwd()

    iter = 10000

    L = 1.0
    config.set_length(L)

    N_r = 4000
    N_b = 2 * config.dim * 40

    network_length = 4
    hidden_dimension_net_u = 256

    tau_u = 1e-4

    plot_period = 1000 # 2000
    print_period = 100 # 100
    N_plot = 100
    chosen_dim_0 = 19
    chosen_dim_1 = 39

    bd_lambda = 10000

    DeepRitz_solver(
        device,
        save_path,
        L,
        N_r,
        N_b,
        network_length,
        hidden_dimension_net_u,
        False,
        iter,
        tau_u,
        plot_period,
        print_period,
        N_plot,
        chosen_dim_0,
        chosen_dim_1,
        bd_lambda,
    )


if __name__ == "__main__":
    main()
