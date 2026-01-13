import os
import pickle
import time

import numpy as np
import torch

import config
import losses
import metrics
import minres_utils
import models
import plotting_utils
import sampling


def PDHG_solver_extrapolation_in_func_space_Bdry_loss(
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
    flag_init,
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
    tau_u=1e-2,
    tau_phi=1e-2,
    tau_psi=1e-2,
):
    torch.manual_seed(50)

    os.makedirs("NPDHG_experiments", exist_ok=True)
    save_path = os.path.join(save_path, "NPDHG_experiments")

    net_u = models.network_prim(network_length, config.dim, hidden_dimension_net_u, 1).to(device)
    net_phi = models.network_dual(network_length, config.dim, hidden_dimension_net_phi, 1, L).to(device)
    net_psi = models.network_dual_on_bdry(network_length, config.dim, hidden_dimension_net_psi, 1).to(device)
    if flag_init:
        net_u.initialization()
        net_phi.initialization()
        net_psi.initialization()

    if precond_type == "MpMd_Id":
        G_u_type = "14"
        loss_phi = losses.L2_norm_sq_phi
        G_phi_type = "1"
    elif precond_type == "MpMd_nabla":  # we use this preconditioner in the current implementation
        G_u_type = "24"
        loss_phi = losses.L2_norm_sq_nabla_phi
        G_phi_type = "2"
    elif precond_type == "Mp_Id_Md_Laplace":
        G_u_type = "14"
        loss_phi = losses.L2_norm_sq_Lap_phi
        G_phi_type = "3"
    else:
        raise ValueError("Unknown precond_type")

    loss_psi = losses.L2_norm_sq_psi
    G_psi_type = "4"

    comp_time_list = []
    computational_time = 0.0
    l2error_list = []
    H1error_list = []
    bdryerr_list = []

    for t in range(iter):
        t0 = time.time()

        in_samples = sampling.rho_1_sampler(N_r)
        bd_samples = sampling.rho_bdry_sampler(N_b)

        net_u.zero_grad()
        net_phi.zero_grad()
        net_psi.zero_grad()

        original_eta = torch.nn.utils.parameters_to_vector(net_phi.parameters())
        original_eta2 = torch.nn.utils.parameters_to_vector(net_psi.parameters())

        for _ in range(phi_psi_iter):
            lossa = losses.PDHG_loss(
                net_u, net_phi, net_psi, in_samples, bd_samples, bd_lambda
            ) - epsilon / 2 * loss_phi(net_phi, in_samples)
            nabla_eta_loss = torch.autograd.grad(
                lossa,
                net_phi.parameters(),
                grad_outputs=None,
                allow_unused=True,
                retain_graph=True,
                create_graph=True,
            )
            vectorized_nabla_eta_loss = torch.nn.utils.parameters_to_vector(nabla_eta_loss)

            net_phi_auxil = models.network_dual(
                network_length, config.dim, hidden_dimension_net_phi, 1, L
            ).to(device)
            net_phi_auxil.load_state_dict(net_phi.state_dict())

            G_inv_nabla_eta_loss, _ = minres_utils.minres_solver_G(
                net_phi,
                net_phi_auxil,
                in_samples,
                bd_samples,
                vectorized_nabla_eta_loss,
                device,
                bd_lambda,
                minres_max_iter,
                minres_tol,
                G_phi_type,
            )

            lossa2 = losses.PDHG_loss(
                net_u, net_phi, net_psi, in_samples, bd_samples, bd_lambda
            ) - epsilon / 2 * bd_lambda * loss_psi(net_psi, bd_samples)
            nabla_eta2_loss = torch.autograd.grad(
                lossa2,
                net_psi.parameters(),
                grad_outputs=None,
                allow_unused=True,
                retain_graph=True,
                create_graph=True,
            )
            vectorized_nabla_eta2_loss = torch.nn.utils.parameters_to_vector(nabla_eta2_loss)

            net_psi_auxil = models.network_dual_on_bdry(
                network_length, config.dim, hidden_dimension_net_psi, 1
            ).to(device)
            net_psi_auxil.load_state_dict(net_psi.state_dict())

            G_inv_nabla_eta2_loss, _ = minres_utils.minres_solver_G(
                net_psi,
                net_psi_auxil,
                in_samples,
                bd_samples,
                vectorized_nabla_eta2_loss,
                device,
                bd_lambda,
                minres_max_iter,
                minres_tol,
                G_psi_type,
            )

            original_eta = torch.nn.utils.parameters_to_vector(net_phi.parameters())
            original_eta2 = torch.nn.utils.parameters_to_vector(net_psi.parameters())
            updated_eta = original_eta + tau_phi * G_inv_nabla_eta_loss
            updated_eta2 = original_eta2 + tau_psi * G_inv_nabla_eta2_loss
            torch.nn.utils.vector_to_parameters(updated_eta, net_phi.parameters())
            torch.nn.utils.vector_to_parameters(updated_eta2, net_psi.parameters())

        net_phi_0 = models.network_dual(
            network_length, config.dim, hidden_dimension_net_phi, 1, L
        ).to(device)
        torch.nn.utils.vector_to_parameters(original_eta, net_phi_0.parameters())
        net_psi_0 = models.network_dual_on_bdry(
            network_length, config.dim, hidden_dimension_net_psi, 1
        ).to(device)
        torch.nn.utils.vector_to_parameters(original_eta2, net_psi_0.parameters())

        for _ in range(u_iter):
            lossb = losses.PDHG_loss_with_extraplt(
                net_u,
                net_phi,
                net_phi_0,
                net_psi,
                net_psi_0,
                in_samples,
                bd_samples,
                bd_lambda,
                omega,
            ) + bd_lambda * losses.bdryloss_use_samples(net_u, bd_samples)
            nabla_theta_loss = torch.autograd.grad(
                lossb,
                net_u.parameters(),
                grad_outputs=None,
                allow_unused=True,
                retain_graph=True,
                create_graph=True,
            )
            vectorized_nabla_theta_loss = torch.nn.utils.parameters_to_vector(nabla_theta_loss)

            net_u_auxil = models.network_prim(
                network_length, config.dim, hidden_dimension_net_u, 1
            ).to(device)
            net_u_auxil.load_state_dict(net_u.state_dict())

            G_inv_nabla_theta_loss, _ = minres_utils.minres_solver_G(
                net_u,
                net_u_auxil,
                in_samples,
                bd_samples,
                vectorized_nabla_theta_loss,
                device,
                bd_lambda,
                minres_max_iter,
                minres_tol,
                G_u_type,
            )

            original_theta = torch.nn.utils.parameters_to_vector(net_u.parameters())
            updated_theta = original_theta - tau_u * G_inv_nabla_theta_loss
            torch.nn.utils.vector_to_parameters(updated_theta, net_u.parameters())

        t1 = time.time()
        computational_time = computational_time + (t1 - t0)
        comp_time_list.append(computational_time)

        if (t + 1) % plot_period == 0:
            plotting_utils.Two_D_plot_graph_nn_primal_with_real_solution(
                0.5, net_u, L, N_plot, chosen_dim_0, chosen_dim_1, t + 1, 1, save_path, -0.5, 4, device
            )
            plotting_utils.Two_D_plot_heatmap_error_cmap(
                0.5, net_u, L, N_plot, chosen_dim_0, chosen_dim_1, t + 1, 1, save_path, 0, 0.5, device
            )

        H1error = metrics.H1_error(net_u, 1200)
        H1error_list.append(H1error.cpu().detach())
        L2error = metrics.L2_error(net_u, 1200)
        l2error_list.append(L2error.cpu().detach())
        boundary_error = torch.sqrt(losses.Loss_2(net_u, 1200))
        bdryerr_list.append(boundary_error.cpu().detach())

        if (t + 1) % print_period == 0:
            print("Iter: {}, ".format(t))
            print("H1 error = {}".format(H1error))
            print("L2 error = {}".format(L2error))
            print("boundary loss = {}".format(boundary_error))

    filename = os.path.join(save_path, "netu.pt")
    torch.save(net_u.state_dict(), filename)

    filename = os.path.join(save_path, "netphi.pt")
    torch.save(net_phi.state_dict(), filename)

    filename = os.path.join(save_path, "netpsi.pt")
    torch.save(net_psi.state_dict(), filename)

    with open("comp_time_list", "wb") as file_t:
        pickle.dump(comp_time_list, file_t)
    with open("H1error_list", "wb") as file0:
        pickle.dump(H1error_list, file0)
    with open("l2error_list", "wb") as file1:
        pickle.dump(l2error_list, file1)
    with open("boundary_error", "wb") as file3:
        pickle.dump(boundary_error, file3)

    fig_plot = plotting_utils.plt.figure(figsize=(15, 15))
    plotting_utils.plt.plot(comp_time_list, np.log(l2error_list / config.u_real_norm) / np.log(10), color="blue")
    plotting_utils.plt.xlabel("Iteration", fontsize=30)
    plotting_utils.plt.ylabel("Relative L2 error", fontsize=30)
    plotting_utils.plt.xticks(fontsize=20)
    plotting_utils.plt.yticks(fontsize=20)
    plotting_utils.plt.title("plot of log_10(Rel L2 error) vs. computation time\n", fontsize=40)
    fig_plot.savefig(os.path.join(save_path, "Plot of the relative log l2 error vs. comptime.pdf"))
    plotting_utils.plt.show()
    plotting_utils.plt.close()

    fig_plot = plotting_utils.plt.figure(figsize=(15, 15))
    plotting_utils.plt.plot(comp_time_list, np.log(H1error_list / config.u_real_H1norm) / np.log(10), color="blue")
    plotting_utils.plt.xlabel("Iteration", fontsize=30)
    plotting_utils.plt.ylabel("Relative H1 error", fontsize=30)
    plotting_utils.plt.xticks(fontsize=20)
    plotting_utils.plt.yticks(fontsize=20)
    plotting_utils.plt.title("plot of log_10(Rel H1 error) vs. computation time\n", fontsize=40)
    fig_plot.savefig(os.path.join(save_path, "Plot of the relative log H1 error vs. comptime.pdf"))
    plotting_utils.plt.show()
    plotting_utils.plt.close()
