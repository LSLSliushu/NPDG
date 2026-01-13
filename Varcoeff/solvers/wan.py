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
import sampling


def PD_adam_WAN_solver(
    device,
    save_path,
    L,
    N_r,
    N_b,
    network_length,
    hidden_dimension_net_u,
    hidden_dimension_net_phi,
    flag_init,
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
):
    torch.manual_seed(50)

    os.makedirs("WAN_experiments", exist_ok=True)
    save_path = os.path.join(save_path, "WAN_experiments")

    net_u = models.network_prim(network_length, config.dim, hidden_dimension_net_u, 1).to(device)
    optim_u = optim.Adam(net_u.parameters(), lr=tau_u, betas=(0.9, 0.99))
    net_phi = models.network_dual(network_length, config.dim, hidden_dimension_net_phi, 1, L).to(device)
    optim_phi = optim.Adam(net_phi.parameters(), lr=tau_phi, betas=(0.9, 0.99))
    if flag_init:
        net_u.initialization()
        net_phi.initialization()

    comp_time = []
    total_time = 0
    l2error_list = []
    H1error_list = []
    l2res_list = []
    bdryerr_list = []

    for t in range(iter):
        t_0 = time.time()

        in_samples = sampling.rho_1_sampler(N_r)
        bd_samples = sampling.rho_bdry_sampler(N_b)
        net_u.zero_grad()
        net_phi.zero_grad()

        for _ in range(phi_psi_iter):
            in_loss = losses.PDHG_loss_without_bd(net_u, net_phi, in_samples)
            lossa = -torch.log(in_loss * in_loss) + torch.log(losses.L2_norm_sq_phi(net_phi, in_samples))
            lossa.backward(retain_graph=True)
            optim_phi.step()

        for _ in range(u_iter):
            in_loss = losses.PDHG_loss_without_bd(net_u, net_phi, in_samples)
            lossb = torch.log(in_loss * in_loss) + bd_lambda * losses.bdryloss_use_samples(net_u, bd_samples)
            lossb.backward(retain_graph=True)
            optim_u.step()

        t_1 = time.time()
        total_time = total_time + (t_1 - t_0)
        comp_time.append(total_time)

        if (t + 1) % plot_period == 0:
            plotting_utils.Two_D_plot_graph_nn_primal_with_real_solution(
                0, net_u, L, N_plot, chosen_dim_0, chosen_dim_1, t + 1, 1, save_path, -0.2, 1, device
            )
            plotting_utils.Two_D_plot_heatmap_error_cmap(
                0, net_u, L, N_plot, chosen_dim_0, chosen_dim_1, t + 1, 1, save_path, 0, 0.1, device
            )

        H1error = metrics.H1_error(net_u, 4000)
        H1error_list.append(H1error.cpu().detach())
        L2error = metrics.L2_error(net_u, 4000)
        l2error_list.append(L2error.cpu().detach())
        Residual = torch.sqrt(losses.PINN_Loss(net_u, 4000))
        l2res_list.append(Residual.cpu().detach())
        boundary_error = torch.sqrt(losses.Loss_2(net_u, 2000))
        bdryerr_list.append(boundary_error.cpu().detach())

        if (t + 1) % print_period == 0:
            print("Iter: {}, ".format(t))
            print("H1 error = {}".format(H1error))
            print("L2 error = {}".format(L2error))
            print("residual = {}".format(Residual))
            print("boundary loss = {}".format(boundary_error))

    filename = os.path.join(save_path, "netu_WAN.pt")
    torch.save(net_u.state_dict(), filename)

    filename = os.path.join(save_path, "netphi_WAN.pt")
    torch.save(net_phi.state_dict(), filename)

    with open("H1error_list", "wb") as file0:
        pickle.dump(H1error_list, file0)
    with open("l2error_list", "wb") as file1:
        pickle.dump(l2error_list, file1)
    with open("Residual", "wb") as file2:
        pickle.dump(Residual, file2)
    with open("boundary_error", "wb") as file3:
        pickle.dump(boundary_error, file3)
    with open("comp_time", "wb") as file_x:
        pickle.dump(comp_time, file_x)

    fig_plot = plotting_utils.plt.figure(figsize=(15, 15))
    plotting_utils.plt.plot(comp_time, np.log(l2error_list / config.u_real_norm) / np.log(10), color="blue")
    plotting_utils.plt.xlabel("Iteration", fontsize=30)
    plotting_utils.plt.ylabel("Relative L2 error", fontsize=30)
    plotting_utils.plt.xticks(fontsize=20)
    plotting_utils.plt.yticks(fontsize=20)
    plotting_utils.plt.title("plot of log_10(Rel L2 error) vs. computation time\n", fontsize=40)
    fig_plot.savefig(os.path.join(save_path, "Plot of the relative log l2 error vs. comptime.pdf"))
    plotting_utils.plt.show()
    plotting_utils.plt.close()

    fig_plot = plotting_utils.plt.figure(figsize=(15, 15))
    plotting_utils.plt.plot(comp_time, np.log(H1error_list / config.u_real_H1norm) / np.log(10), color="blue")
    plotting_utils.plt.xlabel("Iteration", fontsize=30)
    plotting_utils.plt.ylabel("Relative H1 error", fontsize=30)
    plotting_utils.plt.xticks(fontsize=20)
    plotting_utils.plt.yticks(fontsize=20)
    plotting_utils.plt.title("plot of log_10(Rel H1 error) vs. computation time\n", fontsize=40)
    fig_plot.savefig(os.path.join(save_path, "Plot of the relative log H1 error vs. comptime.pdf"))
    plotting_utils.plt.show()
    plotting_utils.plt.close()
