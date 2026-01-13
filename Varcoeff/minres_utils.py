import numpy as np
import scipy
import torch
from scipy.sparse.linalg import LinearOperator

import config
import pde
from autograd_utils import gradient_nn, v_compute_laplacian
import losses
import models


def tensor_to_numpy(u):
    if u.device == "cpu":
        return u.detach().numpy()
    return u.cpu().detach().numpy()


#################################################################################
# In this document, we define various forms of the precondition matrix M(\theta),
# matrix M(\theta) can be viewed as a "metric tensor" in the parameter space,
# we denote the precondition matrix as "G" throughout the implementation.
#################################################################################

# Used to form the precondition matrix.
# Only works for small MLPs due to memory issue.
# Not used in the current code.
def form_metric_tensor(input_dim, net, G_samples, device):
    N = G_samples.size()[0]

    from torch.func import jacrev, vmap, functional_call

    Jacobi_NN = jacrev(functional_call, argnums=1)
    D_param_D_x_NN = vmap(jacrev(Jacobi_NN, argnums=2), in_dims=(None, None, 0))
    D_param_D_x_net_on_x = D_param_D_x_NN(net, dict(net.named_parameters()), G_samples)
    num_params = torch.nn.utils.parameters_to_vector(net.parameters()).size()[0]
    print("Number of params = {}".format(num_params))
    list_of_vectorized_param_gradients = []
    for param_gradients in dict(D_param_D_x_net_on_x).items():
        vectorized_param_gradients = param_gradients[1].view(N, -1, input_dim)
        list_of_vectorized_param_gradients.append(vectorized_param_gradients)
    total_vectorized_param_gradients = torch.cat(list_of_vectorized_param_gradients, 1)
    transpose_total_vectorized_param_gradients = torch.transpose(total_vectorized_param_gradients, 1, 2)
    batched_metric_tensor = torch.matmul(total_vectorized_param_gradients, transpose_total_vectorized_param_gradients)
    metric_tensor = torch.mean(batched_metric_tensor, 0)

    return metric_tensor


# Compute M(\theta) * vec
# M(\theta) is obtained from \mathcal M = Laplacian
# Not used in the current algorithm.
def metric_tensor_as_Laplace_op(net, net_auxil, G_samples, vec, device):
    net.zero_grad()
    net_auxil.zero_grad()
    laplace_net_x = v_compute_laplacian(net, G_samples)
    laplace_net_auxil_x = v_compute_laplacian(net_auxil, G_samples)
    ave_sqr_laplace_net = torch.sum(laplace_net_x * laplace_net_auxil_x) / G_samples.size()[0]
    nabla_theta_ave_sqr_laplace_net = torch.autograd.grad(
        ave_sqr_laplace_net,
        net_auxil.parameters(),
        grad_outputs=None,
        allow_unused=True,
        retain_graph=True,
        create_graph=True,
    )
    vectorize_nabla_theta_ave_sqr_laplace_net = torch.nn.utils.parameters_to_vector(
        nabla_theta_ave_sqr_laplace_net
    )
    vec_dot_nabla_theta_ave_sqr_laplace_net = vectorize_nabla_theta_ave_sqr_laplace_net.dot(vec)
    metric_tensor_mult_vec = torch.autograd.grad(
        vec_dot_nabla_theta_ave_sqr_laplace_net,
        net.parameters(),
        grad_outputs=None,
        allow_unused=True,
        retain_graph=True,
        create_graph=True,
    )
    vectorize_metric_tensor_mult_vec = torch.nn.utils.parameters_to_vector(metric_tensor_mult_vec)

    return vectorize_metric_tensor_mult_vec


# Compute M(\theta) * vec
# M(\theta) is obtained from \mathcal M = \nabla
def metric_tensor_as_nabla_op(net, net_auxil, G_samples, vec, device):
    net.zero_grad()
    net_auxil.zero_grad()
    grad_net_x = gradient_nn(net, G_samples)
    grad_net_auxil_x = gradient_nn(net_auxil, G_samples)
    ave_sqr_grad_net = torch.sum(grad_net_x * grad_net_auxil_x) / G_samples.size()[0]
    nabla_theta_ave_sqr_grad_net = torch.autograd.grad(
        ave_sqr_grad_net,
        net_auxil.parameters(),
        grad_outputs=None,
        allow_unused=True,
        retain_graph=True,
        create_graph=True,
    )
    vectorize_nabla_theta_ave_sqr_grad_net = torch.nn.utils.parameters_to_vector(
        nabla_theta_ave_sqr_grad_net
    )
    vec_dot_nabla_theta_ave_sqr_grad_net = vectorize_nabla_theta_ave_sqr_grad_net.dot(vec)
    metric_tensor_mult_vec = torch.autograd.grad(
        vec_dot_nabla_theta_ave_sqr_grad_net,
        net.parameters(),
        grad_outputs=None,
        allow_unused=True,
        retain_graph=True,
        create_graph=True,
    )
    vectorize_metric_tensor_mult_vec = torch.nn.utils.parameters_to_vector(metric_tensor_mult_vec)

    return vectorize_metric_tensor_mult_vec


# Compute M(\theta) * vec
# M(\theta) is obtained from \mathcal M = \sqrt{A(x)}\nabla
# Used in the current algorithm.
def metric_tensor_as_sqrt_A_nabla_op(net, net_auxil, G_samples, vec, device):
    net.zero_grad()
    net_auxil.zero_grad()
    grad_net_x = gradient_nn(net, G_samples)
    grad_net_auxil_x = gradient_nn(net_auxil, G_samples)
    sigma_x = pde.sigma(G_samples)
    ave_sqr_grad_net = (sigma_x * torch.sum(grad_net_x * grad_net_auxil_x, 1)).mean()
    nabla_theta_ave_sqr_grad_net = torch.autograd.grad(
        ave_sqr_grad_net,
        net_auxil.parameters(),
        grad_outputs=None,
        allow_unused=True,
        retain_graph=True,
        create_graph=True,
    )
    vectorize_nabla_theta_ave_sqr_grad_net = torch.nn.utils.parameters_to_vector(
        nabla_theta_ave_sqr_grad_net
    )
    vec_dot_nabla_theta_ave_sqr_grad_net = vectorize_nabla_theta_ave_sqr_grad_net.dot(vec)
    metric_tensor_mult_vec = torch.autograd.grad(
        vec_dot_nabla_theta_ave_sqr_grad_net,
        net.parameters(),
        grad_outputs=None,
        allow_unused=True,
        retain_graph=True,
        create_graph=True,
    )
    vectorize_metric_tensor_mult_vec = torch.nn.utils.parameters_to_vector(metric_tensor_mult_vec)

    return vectorize_metric_tensor_mult_vec


# Compute M(\theta) * vec
# M(\theta) is obtained from \mathcal M = Id
def metric_tensor_as_op_identity_part(net, net_auxil, G_samples, vec, device):
    net_x = net(G_samples)
    net_auxil_x = net_auxil(G_samples)
    ave_net = torch.sum(net_x * net_auxil_x) / G_samples.size()[0]
    nabla_theta_ave_net = torch.autograd.grad(
        ave_net,
        net_auxil.parameters(),
        grad_outputs=None,
        allow_unused=True,
        retain_graph=True,
        create_graph=True,
    )
    vectorize_nabla_theta_net = torch.nn.utils.parameters_to_vector(nabla_theta_ave_net)
    vec_dot_nabla_theta_ave_net = vectorize_nabla_theta_net.dot(vec)
    metric_tensor_mult_vec = torch.autograd.grad(
        vec_dot_nabla_theta_ave_net,
        net.parameters(),
        grad_outputs=None,
        allow_unused=True,
        retain_graph=True,
        create_graph=True,
    )
    vectorize_metric_tensor_mult_vec = torch.nn.utils.parameters_to_vector(metric_tensor_mult_vec)

    return vectorize_metric_tensor_mult_vec


# Compute M(\theta) * vec
# M(\theta) is obtained from \mathcal M = Trace operator
def metric_tensor_as_trace_op(net, net_auxil, G_samples, vec, device): # G_samples are boundary samples
    net_x = net(G_samples)
    net_auxil_x = net_auxil(G_samples)
    ave_net = torch.sum(net_x * net_auxil_x) / G_samples.size()[0]
    nabla_theta_ave_net = torch.autograd.grad(
        ave_net,
        net_auxil.parameters(),
        grad_outputs=None,
        allow_unused=True,
        retain_graph=True,
        create_graph=True,
    )
    vectorize_nabla_theta_net = torch.nn.utils.parameters_to_vector(nabla_theta_ave_net)
    vec_dot_nabla_theta_ave_net = vectorize_nabla_theta_net.dot(vec)
    metric_tensor_mult_vec = torch.autograd.grad(
        vec_dot_nabla_theta_ave_net,
        net.parameters(),
        grad_outputs=None,
        allow_unused=True,
        retain_graph=True,
        create_graph=True,
    )
    vectorize_metric_tensor_mult_vec = torch.nn.utils.parameters_to_vector(metric_tensor_mult_vec)

    return vectorize_metric_tensor_mult_vec


# Compute M(\theta) * vec
# M(\theta) is obtained from \mathcal M = [Trace operator, \nabla^{∂Ω}]' c.f. eq. (52) and discussion below in the paper.
def metric_tensor_as_H1_trace_op(net, net_auxil, G_samples, bdd_idx, vec, device):
    net_x = net(G_samples)
    net_auxil_x = net_auxil(G_samples)
    nabla_net_x = gradient_nn(net, G_samples)
    nabla_net_auxil_x = gradient_nn(net_auxil, G_samples)
    N, d = G_samples.shape
    mask = np.ones((N, d), dtype=bool)
    mask[np.arange(N), bdd_idx] = False
    projected_nabla_net_x = nabla_net_x[mask].reshape(N, d - 1)
    projected_nabla_net_auxil_x = nabla_net_auxil_x[mask].reshape(N, d - 1)

    ave_net_L2 = torch.sum(net_x * net_auxil_x) / G_samples.size()[0]
    ave_net_dotH1 = torch.sum(projected_nabla_net_x * projected_nabla_net_auxil_x) / G_samples.size()[0]
    ave_net = ave_net_L2 + ave_net_dotH1

    nabla_theta_ave_net = torch.autograd.grad(
        ave_net,
        net_auxil.parameters(),
        grad_outputs=None,
        allow_unused=True,
        retain_graph=True,
        create_graph=True,
    )
    vectorize_nabla_theta_net = torch.nn.utils.parameters_to_vector(nabla_theta_ave_net)
    vec_dot_nabla_theta_ave_net = vectorize_nabla_theta_net.dot(vec)
    metric_tensor_mult_vec = torch.autograd.grad(
        vec_dot_nabla_theta_ave_net,
        net.parameters(),
        grad_outputs=None,
        allow_unused=True,
        retain_graph=True,
        create_graph=True,
    )
    vectorize_metric_tensor_mult_vec = torch.nn.utils.parameters_to_vector(metric_tensor_mult_vec)

    return vectorize_metric_tensor_mult_vec


# Apply MINRES to solve G * sol = RHS_vec
def minres_solver_G(
    net,
    net_auxil,
    interior_samples,
    boundary_samples,
    bdd_idx,
    RHS_vec,
    device,
    bd_lambda,
    max_iternum,
    minres_tolerance,
    G_type,
):
    num_params = torch.nn.utils.parameters_to_vector(net.parameters()).size()[0]

    # G1: \mathcal M = Id operator
    # G2: \mathcal M = ▽ operator
    # G2b: \mathcal M = sqrt(A)\nabla operator
    # G3: \mathcal M = -△ operator
    # G4: \mathcal M = lambda T (Trace) operator
    # G5: \mathcal M = [Trace operator, \nabla^{∂Ω}]' c.f. eq. (52) and discussion below in the paper.
    # G14: \mathcal M = [Id, T]'
    # G24: \mathcal M = [▽, sqrt{lambda}T]'
    # G34: \mathcal M = [-△, sqrt{lambda}T]'
    # G25: \mathcal M = [▽, sqrt{lambda}Trace operator, sqrt{lambda}\nabla^{∂Ω}]' 

    # In this implementation, we use 
    # either G24 for u, G2 for \phi, G4 for \psi 
    # or G2b5 for u, G2b for \phi, G5 for \psi 
    # in the current paper.
    def G1_as_operator(vec):
        tensorized_vec = torch.Tensor(vec).to(device)
        Gv = metric_tensor_as_op_identity_part(net, net_auxil, interior_samples, tensorized_vec, device)
        return tensor_to_numpy(Gv)

    def G2_as_operator(vec):
        tensorized_vec = torch.Tensor(vec).to(device)
        Gv = metric_tensor_as_nabla_op(net, net_auxil, interior_samples, tensorized_vec, device)
        return tensor_to_numpy(Gv)

    def G2b_as_operator(vec):
        tensorized_vec = torch.Tensor(vec).to(device)
        Gv = metric_tensor_as_sqrt_A_nabla_op(net, net_auxil, interior_samples, tensorized_vec, device)
        return tensor_to_numpy(Gv)

    def G3_as_operator(vec):
        tensorized_vec = torch.Tensor(vec).to(device)
        Gv = metric_tensor_as_Laplace_op(net, net_auxil, interior_samples, tensorized_vec, device)
        return tensor_to_numpy(Gv)

    def G4_as_operator(vec):
        tensorized_vec = torch.Tensor(vec).to(device)
        Gv = bd_lambda * metric_tensor_as_trace_op(net, net_auxil, boundary_samples, tensorized_vec, device)
        return tensor_to_numpy(Gv)

    def G5_as_operator(vec):
        tensorized_vec = torch.Tensor(vec).to(device)
        Gv = bd_lambda * metric_tensor_as_H1_trace_op(net, net_auxil, boundary_samples, bdd_idx, tensorized_vec, device)
        return tensor_to_numpy(Gv)

    def G14_as_operator(vec):
        tensorized_vec = torch.Tensor(vec).to(device)
        Gv = metric_tensor_as_op_identity_part(net, net_auxil, interior_samples, tensorized_vec, device) + bd_lambda * metric_tensor_as_trace_op(net, net_auxil, boundary_samples, tensorized_vec, device)
        return tensor_to_numpy(Gv)

    def G24_as_operator(vec):
        tensorized_vec = torch.Tensor(vec).to(device)
        Gv = metric_tensor_as_nabla_op(net, net_auxil, interior_samples, tensorized_vec, device) + bd_lambda * metric_tensor_as_trace_op(net, net_auxil, boundary_samples, tensorized_vec, device)
        return tensor_to_numpy(Gv)

    def G2b4_as_operator(vec):
        tensorized_vec = torch.Tensor(vec).to(device)
        Gv = metric_tensor_as_sqrt_A_nabla_op(net, net_auxil, interior_samples, tensorized_vec, device) + bd_lambda * metric_tensor_as_trace_op(net, net_auxil, boundary_samples, tensorized_vec, device)
        return tensor_to_numpy(Gv)

    def G34_as_operator(vec):
        tensorized_vec = torch.Tensor(vec).to(device)
        Gv = metric_tensor_as_Laplace_op(net, net_auxil, interior_samples, tensorized_vec, device) + bd_lambda * metric_tensor_as_trace_op(net, net_auxil, boundary_samples, tensorized_vec, device)
        return tensor_to_numpy(Gv)

    def G25_as_operator(vec):
        tensorized_vec = torch.Tensor(vec).to(device)
        Gv = metric_tensor_as_nabla_op(net, net_auxil, interior_samples, tensorized_vec, device) + bd_lambda * metric_tensor_as_H1_trace_op(net, net_auxil, boundary_samples, bdd_idx, tensorized_vec, device)
        return tensor_to_numpy(Gv)

    def G2b5_as_operator(vec):
        tensorized_vec = torch.Tensor(vec).to(device)
        Gv = metric_tensor_as_sqrt_A_nabla_op(net, net_auxil, interior_samples, tensorized_vec, device) + bd_lambda * metric_tensor_as_H1_trace_op(net, net_auxil, boundary_samples, bdd_idx, tensorized_vec, device)
        return tensor_to_numpy(Gv)

    if G_type == "1":
        G_operator = LinearOperator((num_params, num_params), matvec=G1_as_operator)
    elif G_type == "2":
        G_operator = LinearOperator((num_params, num_params), matvec=G2_as_operator)
    elif G_type == "2b":
        G_operator = LinearOperator((num_params, num_params), matvec=G2b_as_operator)
    elif G_type == "3":
        G_operator = LinearOperator((num_params, num_params), matvec=G3_as_operator)
    elif G_type == "4":
        G_operator = LinearOperator((num_params, num_params), matvec=G4_as_operator)
    elif G_type == "5":
        G_operator = LinearOperator((num_params, num_params), matvec=G5_as_operator)
    elif G_type == "14":
        G_operator = LinearOperator((num_params, num_params), matvec=G14_as_operator)
    elif G_type == "24":
        G_operator = LinearOperator((num_params, num_params), matvec=G24_as_operator)
    elif G_type == "2b4":
        G_operator = LinearOperator((num_params, num_params), matvec=G2b4_as_operator)
    elif G_type == "34":
        G_operator = LinearOperator((num_params, num_params), matvec=G34_as_operator)
    elif G_type == "25":
        G_operator = LinearOperator((num_params, num_params), matvec=G25_as_operator)
    elif G_type == "2b5":
        G_operator = LinearOperator((num_params, num_params), matvec=G2b5_as_operator)
    else:
        raise ValueError("Wrong G_type")
    np_RHS_vec = tensor_to_numpy(RHS_vec)
    sol_vec, info = scipy.sparse.linalg.minres(G_operator, np_RHS_vec, rtol=minres_tolerance)
    if torch.max(torch.isnan(torch.tensor(sol_vec))) > 0:
        print("MINRES got NAN!")
        sol_vec = np_RHS_vec  # skip the preconditioning step
        info = 0
    tensorized_sol_vec = torch.Tensor(sol_vec).to(device)
    return tensorized_sol_vec, info


