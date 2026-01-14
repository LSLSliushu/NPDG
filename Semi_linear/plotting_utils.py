import os
import math

import matplotlib
import matplotlib.tri as tri
matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

import config
import pde



def Two_D_plot_graph_nn_primal_with_real_solution(
    net_primal, Radius, num_of_intervals, Iter, flag_plot_real, save_path, z_min, z_max, device, d=None
):
    if d is None:
        d = config.dim
    r, theta = np.meshgrid(
        np.linspace(0.0, Radius, num_of_intervals + 1),
        np.linspace(0.0, 2 * math.pi, num_of_intervals + 1),
    )
    r = torch.tensor(r)
    theta = torch.tensor(theta)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    node_tensor = torch.stack((x, y), 2)
    whole_dim_node_tensor = torch.zeros(num_of_intervals + 1, num_of_intervals + 1, d)
    whole_dim_node_tensor[:, :, 0] = node_tensor[:, :, 0]
    whole_dim_node_tensor[:, :, 1] = node_tensor[:, :, 1]

    u_nodes = net_primal(whole_dim_node_tensor.to(device))
    squeezed_u_nodes = u_nodes.squeeze()

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim([-Radius - 1, Radius + 1])
    ax.set_ylim([-Radius - 1, Radius + 1])
    ax.set_zlim([z_min, z_max])
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

    ax.set_title("Graph of u", fontsize=20)

    filename = os.path.join(save_path, "({}th Iteration) graph of u on the 0-1 plane".format(Iter) + ".pdf")
    plt.savefig(filename)
    plt.close()


def heatmap_error_on_disk_domain(iter, plotting_coord, Radius, num_of_intervals, chosen_dim_0, chosen_dim_1, net_u, save_path, vmin, vmax):

    device = torch.device('cuda:0')

    r, theta = np.meshgrid(np.linspace(0.0, Radius, num_of_intervals), np.linspace(0.0, 2 * math.pi, num_of_intervals))
    r = torch.tensor(r)
    theta = torch.tensor(theta)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    node_tensor = torch.stack((x, y), 2)
    whole_dim_node_tensor = plotting_coord * torch.ones(num_of_intervals , num_of_intervals , config.dim)
    whole_dim_node_tensor[:, :, chosen_dim_0] = node_tensor[:, :, 0]
    whole_dim_node_tensor[:, :, chosen_dim_1] = node_tensor[:, :, 1]

    u_nodes = net_u(whole_dim_node_tensor.to(device))
    u_nodes_1D = torch.flatten(u_nodes)
    np_u_nodes = np.array(u_nodes_1D.cpu().detach())
    u_real_nodes = pde.u_real(whole_dim_node_tensor)
    u_real_nodes = torch.flatten(u_real_nodes)
    np_u_real_nodes = np.array(u_real_nodes.cpu().detach())

    # First create the x and y coordinates of the points.
    n_angles = num_of_intervals
    n_radii = num_of_intervals
    min_radius = 0
    radii = np.linspace(min_radius, Radius, n_radii)
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
    x = (radii * np.cos(angles)).flatten()
    y = (radii * np.sin(angles)).flatten()
    z = np.abs(np_u_nodes - np_u_real_nodes)

    # Create the Triangulation; no triangles so Delaunay triangulation created.
    triang = tri.Triangulation(x, y)

    fig, ax = plt.subplots(figsize=(14, 14))   # publication-size square
    # fig, ax = plt.subplots(figsize=[15, 12])
    ax.set_aspect('equal')
    tpc = ax.tripcolor(triang, z, shading='gouraud', cmap='rainbow', vmin=vmin, vmax=vmax )

    # ---- colorbar (ONE only, height matches the axes) ----
    cbar = fig.colorbar(
        tpc,                # <-- the mappable
        ax=ax,              # <-- attach to the same axes
        fraction=0.046,     # height relative to axes
        pad=0.04            # gap to the plot
    )
    cbar.set_label('colorbar', fontsize=37, labelpad=20)
    cbar.ax.tick_params(labelsize=30, pad=8)

    plt.xlabel('x_{}'.format(chosen_dim_0+1), fontsize=30)
    plt.ylabel('x_{}'.format(chosen_dim_1+1), fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    ax.set_title('Heatmap of |u_theta - u_*| on {}-{} plane\n'.format(chosen_dim_0+1, chosen_dim_1+1, plotting_coord), fontsize = 40)

    filename = os.path.join(save_path, '[Iteration{}] Error heatmap on round domain on {}-{} plane'.format(iter, chosen_dim_0+1, chosen_dim_1+1)+'.jpg')
    plt.savefig(filename)
