import os

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

import config
import pde


def Two_D_plot_graph_nn_primal_with_real_solution(
    plotting_coord,
    net_primal,
    l,
    num_of_intervals,
    chosen_dim_0,
    chosen_dim_1,
    Iter,
    flag_plot_real,
    save_path,
    z_min,
    z_max,
    device,
    d=None,
):
    if d is None:
        d = config.dim

    x, y = np.meshgrid(
        np.linspace(0.0, l, num_of_intervals + 1),
        np.linspace(0.0, l, num_of_intervals + 1),
    )
    node_tensor = torch.stack((torch.tensor(x), torch.tensor(y)), 2).to(torch.float32)
    whole_dim_node_tensor = plotting_coord * torch.ones(
        num_of_intervals + 1, num_of_intervals + 1, d
    )
    whole_dim_node_tensor[:, :, chosen_dim_0] = node_tensor[:, :, 0]
    whole_dim_node_tensor[:, :, chosen_dim_1] = node_tensor[:, :, 1]

    u_nodes = net_primal(whole_dim_node_tensor.to(device))
    squeezed_u_nodes = u_nodes.squeeze()

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim([0, l])
    ax.set_ylim([0, l])
    # ax.set_zlim([z_min, z_max]) # optional
    ax.scatter(
        node_tensor.cpu()[:, :, 0],
        node_tensor.cpu()[:, :, 1],
        squeezed_u_nodes.cpu().detach().numpy(),
        color="blue",
        s=1,
    )

    if flag_plot_real == 1:
        u0 = pde.u_real(whole_dim_node_tensor.to(device))
        squeezed_u0 = u0.squeeze()
        ax.scatter(
            node_tensor.cpu()[:, :, 0],
            node_tensor.cpu()[:, :, 1],
            squeezed_u0.cpu().detach().numpy(),
            color="r",
            s=1,
        )

    ax.set_title(
        "Graph of u on {}-{} plane (with all remaining coordinates={})\n".format(
            chosen_dim_0 + 1, chosen_dim_1 + 1, round(plotting_coord, 2)
        ),
        fontsize=40,
    )
    ax.set_xlabel("x{} axis".format(chosen_dim_0+1), fontsize=30, labelpad=20)
    ax.set_ylabel("x{} axis".format(chosen_dim_1+1), fontsize=30, labelpad=20)
    ax.set_zlabel("u value", fontsize=30)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)

    filename = os.path.join(
        save_path,
        "({}th Iteration) graph of u on {}-{} plane [plotting on coord={}]".format(
            Iter, chosen_dim_0+1, chosen_dim_1+1, plotting_coord
        )
        + ".pdf",
    )
    plt.savefig(filename)
    plt.close()


def Two_D_plot_heatmap_error_cmap(
    plotting_coord,
    net_primal,
    l,
    num_of_intervals,
    chosen_dim_0,
    chosen_dim_1,
    Iter,
    flag_plotreal,
    save_path,
    z_min,
    z_max,
    device,
    d=None,
):
    if d is None:
        d = config.dim

    x, y = np.meshgrid(
        np.linspace(0.0, l, num_of_intervals + 1),
        np.linspace(0.0, l, num_of_intervals + 1),
    )
    node_tensor = torch.stack((torch.tensor(x), torch.tensor(y)), 2).to(torch.float32)
    whole_dim_node_tensor = plotting_coord * torch.ones(
        num_of_intervals + 1, num_of_intervals + 1, d
    )
    whole_dim_node_tensor[:, :, chosen_dim_0] = node_tensor[:, :, 0]
    whole_dim_node_tensor[:, :, chosen_dim_1] = node_tensor[:, :, 1]

    u_nodes = net_primal(whole_dim_node_tensor.to(device))
    squeezed_u_nodes = u_nodes.squeeze()
    real_u_nodes = pde.u_real(whole_dim_node_tensor.to(device))
    squeezed_real_u_nodes = real_u_nodes.squeeze()

    fig, ax = plt.subplots(figsize=[15, 12])
    c = ax.pcolormesh(
        x,
        y,
        torch.abs(squeezed_u_nodes - squeezed_real_u_nodes).cpu().detach().numpy(),
        cmap="rainbow",
        vmin=z_min,
        vmax=z_max,
    )
    ax.set_title(
        "Heatmap of |u_theta - u_*| on {}-{} plane\n (with all remaining coordinates={})\n".format(
            chosen_dim_0 + 1, chosen_dim_1 + 1, plotting_coord
        ),
        fontsize=40,
    )
    ax.set_title("Error heatmap", fontsize=40)
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    ax.axis("equal")
    ax.set_aspect("equal", "box")
    fig.colorbar(c, ax=ax).set_label(label="colorbar of the heat graph", size=30, weight="light")
    plt.xlabel("x_{}".format(chosen_dim_0 + 1), fontsize=15)
    plt.ylabel("x_{}".format(chosen_dim_1 + 1), fontsize=15)

    filename = os.path.join(
        save_path,
        "({}th Iteration) error heat map (rainbow) [plotting on coord={}]".format(
            Iter, plotting_coord
        )
        + ".pdf",
    )
    plt.savefig(filename)
    plt.close()

