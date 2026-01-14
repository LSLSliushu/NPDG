import os
import pickle
import time

import numpy as np
import torch
import torch.optim as optim

import config
import losses
import metrics
import models
import plotting_utils


def PINN_solver(
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
):
    torch.manual_seed(50)

    os.makedirs("PINN_experiments", exist_ok=True)
    save_path = os.path.join(save_path, "PINN_experiments")

    net_u = models.network_prim(network_length, config.dim, hidden_dimension_net_u, 1).to(device)
    optim_u = optim.Adam(net_u.parameters(), lr=tau_u, betas=(0.9, 0.99))
    if flag_init:
        net_u.initialization()

    comp_time = []
    total_time = 0
    l2error_list = []
    H1error_list = []
    bdryerr_list = []
    print(iter)
    for t in range(iter):
        t_0 = time.time()
        net_u.zero_grad()
        loss_Adam = losses.PINN_Loss(net_u, N_r) + bd_lambda * losses.bdryloss_use_samples_resampling(net_u, N_b)
        loss_Adam.backward(retain_graph=True)
        optim_u.step()

        t_1 = time.time()
        total_time = total_time + (t_1 - t_0)
        comp_time.append(total_time)

        if (t + 1) % plot_period == 0:
            z_min = -1.5
            z_max = 1.5
            plotting_utils.Two_D_plot_graph_nn_primal_with_real_solution(
                net_u, R, N_plot, t, True, save_path, z_min, z_max, device
            )
            for idx in range(config.dim-1):
                plotting_utils.heatmap_error_on_disk_domain(
                    t, 0, R, N_plot, idx, idx + 1, net_u, save_path, vmin=0, vmax=0.15
                )

        H1error = metrics.H1_error(net_u, 8000)
        H1error_list.append(H1error.cpu().detach())
        print("H1 error = {}".format(H1error))

        L2error = metrics.L2_error(net_u, 8000)
        l2error_list.append(L2error.cpu().detach())
        print("L2 error = {}".format(L2error))

        boundary_error = torch.sqrt(losses.bdryloss_use_samples_resampling(net_u, 4000))
        bdryerr_list.append(boundary_error.cpu().detach())
        print("boundary loss = {}".format(boundary_error))

    filename = os.path.join(save_path, "netu.pt")
    torch.save(net_u.state_dict(), filename)

    with open("H1error_list", "wb") as file0:
        pickle.dump(H1error_list, file0)
    with open("l2error_list", "wb") as file1:
        pickle.dump(l2error_list, file1)
    with open("boundary_error", "wb") as file3:
        pickle.dump(boundary_error, file3)
    with open("comp_time", "wb") as file_x:
        pickle.dump(comp_time, file_x)

    fig_plot = plotting_utils.plt.figure(figsize=(15, 15))
    plotting_utils.plt.plot(comp_time, np.log(np.array(l2error_list) / config.u_real_norm) / np.log(10), color="blue")
    plotting_utils.plt.xlabel("Iteration", fontsize=30)
    plotting_utils.plt.ylabel("Relative L2 error", fontsize=30)
    plotting_utils.plt.xticks(fontsize=20)
    plotting_utils.plt.yticks(fontsize=20)
    plotting_utils.plt.title("plot of log_10(Rel L2 error) vs. computation time\n", fontsize=40)
    fig_plot.savefig(os.path.join(save_path, "Plot of the relative log l2 error vs. comptime.pdf"))
    plotting_utils.plt.show()
    plotting_utils.plt.close()

    fig_plot = plotting_utils.plt.figure(figsize=(15, 15))
    plotting_utils.plt.plot(comp_time, np.log(np.array(H1error_list) / config.u_real_H1norm) / np.log(10), color="blue")
    plotting_utils.plt.xlabel("Iteration", fontsize=30)
    plotting_utils.plt.ylabel("Relative H1 error", fontsize=30)
    plotting_utils.plt.xticks(fontsize=20)
    plotting_utils.plt.yticks(fontsize=20)
    plotting_utils.plt.title("plot of log_10(Rel H1 error) vs. computation time\n", fontsize=40)
    fig_plot.savefig(os.path.join(save_path, "Plot of the relative log H1 error vs. comptime.pdf"))
    plotting_utils.plt.show()
    plotting_utils.plt.close()
