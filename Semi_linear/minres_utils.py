import numpy as np
import scipy
import torch
from scipy.sparse.linalg import LinearOperator

from autograd_utils import gradient_nn, v_compute_laplacian


def tensor_to_numpy(u):
    if u.device == "cpu":
        return u.detach().numpy()
    return u.cpu().detach().numpy()


#################################################################################
# In this document, we define various forms of the precondition matrix M(\theta),
# matrix M(\theta) can be viewed as a "metric tensor" in the parameter space,
# we denote the precondition matrix as "G" throughout the implementation.
#################################################################################

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
def metric_tensor_as_trace_op(net, net_auxil, G_samples, vec, device):
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


# Apply MINRES to solve G * sol = RHS_vec
def minres_solver_G(
    net,
    net_auxil,
    interior_samples,
    boundary_samples,
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
    # G14: \mathcal M = [Id, T]'
    # G24: \mathcal M = [▽, sqrt{lambda}T]'
    # G34: \mathcal M = [-△, sqrt{lambda}T]'

    # In this implementation, we use 
    # G24 for u, G2 for \phi, G4 for \psi 
    # in the current paper.
    def G1_as_operator(vec):
        tensorized_vec = torch.Tensor(vec).to(device)
        Gv = metric_tensor_as_op_identity_part(net, net_auxil, interior_samples, tensorized_vec, device)
        return tensor_to_numpy(Gv)

    def G2_as_operator(vec):
        tensorized_vec = torch.Tensor(vec).to(device)
        Gv = metric_tensor_as_nabla_op(net, net_auxil, interior_samples, tensorized_vec, device)
        return tensor_to_numpy(Gv)

    def G3_as_operator(vec):
        tensorized_vec = torch.Tensor(vec).to(device)
        Gv = metric_tensor_as_Laplace_op(net, net_auxil, interior_samples, tensorized_vec, device)
        return tensor_to_numpy(Gv)

    def G4_as_operator(vec):
        tensorized_vec = torch.Tensor(vec).to(device)
        Gv = bd_lambda * metric_tensor_as_trace_op(net, net_auxil, boundary_samples, tensorized_vec, device)
        return tensor_to_numpy(Gv)

    def G14_as_operator(vec):
        tensorized_vec = torch.Tensor(vec).to(device)
        Gv = metric_tensor_as_op_identity_part(
            net, net_auxil, interior_samples, tensorized_vec, device
        ) + bd_lambda * metric_tensor_as_trace_op(net, net_auxil, boundary_samples, tensorized_vec, device)
        return tensor_to_numpy(Gv)

    def G24_as_operator(vec):
        tensorized_vec = torch.Tensor(vec).to(device)
        Gv = metric_tensor_as_nabla_op(
            net, net_auxil, interior_samples, tensorized_vec, device
        ) + bd_lambda * metric_tensor_as_trace_op(net, net_auxil, boundary_samples, tensorized_vec, device)
        return tensor_to_numpy(Gv)

    def G34_as_operator(vec):
        tensorized_vec = torch.Tensor(vec).to(device)
        Gv = metric_tensor_as_Laplace_op(
            net, net_auxil, interior_samples, tensorized_vec, device
        ) + bd_lambda * metric_tensor_as_trace_op(net, net_auxil, boundary_samples, tensorized_vec, device)
        return tensor_to_numpy(Gv)

    if G_type == "1":
        G_operator = LinearOperator((num_params, num_params), matvec=G1_as_operator)
    elif G_type == "2":
        G_operator = LinearOperator((num_params, num_params), matvec=G2_as_operator)
    elif G_type == "3":
        G_operator = LinearOperator((num_params, num_params), matvec=G3_as_operator)
    elif G_type == "4":
        G_operator = LinearOperator((num_params, num_params), matvec=G4_as_operator)
    elif G_type == "14":
        G_operator = LinearOperator((num_params, num_params), matvec=G14_as_operator)
    elif G_type == "24":
        G_operator = LinearOperator((num_params, num_params), matvec=G24_as_operator)
    elif G_type == "34":
        G_operator = LinearOperator((num_params, num_params), matvec=G34_as_operator)
    else:
        raise ValueError("Wrong G_type")

    np_rhs_vec = tensor_to_numpy(RHS_vec)
    sol_vec, info = scipy.sparse.linalg.minres(
        G_operator, np_rhs_vec, rtol=minres_tolerance, maxiter=max_iternum
    )

    if torch.max(torch.isnan(torch.tensor(sol_vec))) > 0:
        print("MINRES got NAN!")
        sol_vec = np_rhs_vec  # skip the preconditioning step
        info = 0

    tensorized_sol_vec = torch.Tensor(sol_vec).to(device)
    return tensorized_sol_vec, info
