#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Holt'

import torch
import torch.nn as nn
from prettytable import PrettyTable

__doc__ = """
This file contains a serial of useful functions,
like load CS matirx, all dataset, compute Gsure loss, and so on.
"""

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = None
def set_device(dev):
    """Store the torch device used by legacy utility modules.
    
    :param dev: Torch device used by tensors created in this module.
    :return: None.
    """
    global device
    device = dev


def model_parameters(net):
    """Build a table of trainable parameter counts.

    :param net: Torch module whose trainable parameters are listed.
    :return: PrettyTable containing each parameter name and size.
    """
    table = PrettyTable(["Network Parameters name", "# Parameters size"])
    num_params = 0
    for name, parameter in net.named_parameters():
        if not parameter.requires_grad:
            continue
        table.add_row([name, parameter.numel()])
        num_params += parameter.numel()
    table.add_row(['TOTAL', num_params])
    return table


class NMSE_loss(nn.Module):
    """Normalized MSE loss for batch-first channel estimates."""

    def __init__(self):
        """Create the NMSE loss module."""
        super().__init__()

    def forward(self, h_predict, h_label):
        """Compute mean NMSE = E[||h_predict - h_label||_2^2 / ||h_label||_2^2].
        
        :param h_predict: Predicted channels with shape (batch, dim).
        :param h_label: Reference channels with shape (batch, dim).
        :return: Scalar NMSE loss.
        """
        NMSE = torch.mean(torch.sum(torch.abs(h_predict - h_label) ** 2, 1) / torch.sum(torch.abs(h_label) ** 2, 1))
        loss = NMSE
        return loss


def compute_NMSE(h_predict, h_label):
    """Compute global NMSE in dB.
    
    :param h_predict: Predicted channels with shape (batch, dim).
    :param h_label: Reference channels with shape (batch, dim).
    :return: Scalar tensor containing 10 * log10(||h_predict - h_label||_2^2 / ||h_label||_2^2).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    h_predict = h_predict.to(device)
    h_label = h_label.to(device)

    with torch.no_grad():
        NMSE = torch.sum(torch.abs(h_predict - h_label) ** 2) / torch.sum(torch.abs(h_label) ** 2)
        NMSE = 10 * torch.log10(NMSE)
    return NMSE


class GSURE(nn.Module):
    """Generalized SURE loss for the projected MSE induced by A.

    GSURE estimates ||P(xhat - x)||_2^2 from the sufficient statistic
    u = A.T @ inv(Sigma) @ y, where P = pinv(A) @ A is the projection onto
    the row space of A. The divergence term is estimated by Monte Carlo
    perturbations of u.
    """

    def __init__(self, net, A, Sigma) -> None:
        """Create the GSURE loss for one sensing model.
        
        :param net: Estimator mapping batch-first sufficient statistics to channels.
        :param A: Real sensing matrix with shape (2M, 2N).
        :param Sigma: Real measurement covariance basis with shape (2M, 2M).
        :return: None.
        """
        super().__init__()
        self.net = net.to(device)
        self.A = A.to(device)
        self.Sigma = Sigma.to(device)
        self.pinvA = torch.linalg.pinv(self.A).to(device)
        self.pinvA_Sig_AT = self.pinvA @ self.Sigma @ self.pinvA.T
        self.P = torch.matmul(self.pinvA, self.A).to(device)
        self.E = torch.eye(self.P.shape[0], dtype=self.P.dtype, device=self.net.device())
        self.M, self.N = self.A.shape

    def name(self,):
        """Return the loss name used by legacy logging."""
        return "GSURE"

    def proj_Matrix_acquire(self):
        """Return projection matrices used in the GSURE terms.

        :return: Tuple (proj_for_u, proj_div, covariance_basis), each with shape (2N, 2N).
        """
        proj_for_u = self.pinvA_Sig_AT
        proj_div = self.P
        covariance_basis = self.pinvA_Sig_AT
        return proj_for_u, proj_div, covariance_basis

    def monte_Carlo_div(self, xhat, u, proj_div):
        """Estimate trace(proj_div @ dxhat/du) by Monte Carlo perturbation.

        :param xhat: Column-wise channel estimate with shape (2N, batch).
        :param u: Column-wise sufficient statistic with shape (2N, batch).
        :param proj_div: Projection matrix applied before the divergence inner product, shape (2N, 2N).
        :return: Divergence estimate with shape (batch,).
        """
        # generate perturbation
        epsilon = torch.maximum(.001 * torch.max(torch.abs(u)), torch.tensor(.00001, device=u.device))
        epsilon = epsilon.to(u.device)
        eta = torch.randn(u.shape, dtype=self.A.dtype, device=u.device)

        # calculate divergence
        if torch.is_complex(eta):
            u_perturbed_r = u + torch.real(torch.mul(eta, epsilon))
            x_hat_perturbed_r = self.net(u_perturbed_r.T).T
            del_x_hat_perturbed_r = torch.real(torch.matmul(proj_div, x_hat_perturbed_r - xhat))
            eta_dx_r = torch.sum(torch.mul(torch.real(eta), del_x_hat_perturbed_r), 0)
            MC_div_r = torch.div(eta_dx_r, epsilon)

            u_perturbed_i = u + 1j * torch.imag(torch.mul(eta, epsilon))
            x_hat_perturbed_i = self.net(u_perturbed_i.T).T
            del_x_hat_perturbed_i = torch.imag(torch.matmul(proj_div, x_hat_perturbed_i - xhat))
            eta_dx_i = torch.sum(torch.mul(torch.imag(eta), del_x_hat_perturbed_i), 0)
            MC_div_i = torch.div(eta_dx_i, epsilon)

            MC_div = (MC_div_r + MC_div_i)
        elif torch.is_floating_point(eta):
            u_perturbed = u + torch.mul(eta, epsilon)
            x_hat_perturbed = self.net(u_perturbed.T).T
            del_x_hat_perturbed = torch.matmul(proj_div, x_hat_perturbed - xhat)
            eta_dx = torch.sum(torch.mul(eta, del_x_hat_perturbed), 0)
            MC_div = torch.div(eta_dx, epsilon)
        else:
            raise TypeError("Neither real or complex.")

        return MC_div

    def square_Term(self, xhat, u, proj_for_u):
        """Compute ||P xhat - proj_for_u u||_2^2 for each sample.

        :param xhat: Column-wise channel estimate with shape (2N, batch).
        :param u: Column-wise sufficient statistic with shape (2N, batch).
        :param proj_for_u: Matrix mapping u to the projected ML estimate, shape (2N, 2N).
        :return: Squared projected error term with shape (batch,).
        """
        Pxhat = torch.matmul(self.P, xhat)
        proj_u = torch.matmul(proj_for_u, u)
        squareTerm = torch.sum(torch.square(torch.abs(Pxhat - proj_u)), 0)
        return squareTerm

    def trace_Term(self, covariance_basis):
        """Compute trace(covariance_basis) for the GSURE correction.
        
        :param covariance_basis: Real covariance matrix with shape (2N, 2N).
        :return: Scalar trace term.
        """
        return torch.trace(covariance_basis)

    def forward(self, xhat, u, sigma_squared):
        """Compute batch-mean GSURE for projected channel MSE.

        GSURE = ||P xhat - proj_for_u u||_2^2 + 2 sigma_squared * div - sigma_squared * trace.

        :param xhat: Batch-first channel estimate with shape (batch, 2N).
        :param u: Batch-first sufficient statistic A.T @ inv(Sigma) @ y with shape (batch, 2N).
        :param sigma_squared: Noise variance with shape (batch, 1), for y = A h + z.
        :return: Scalar GSURE loss.
        """
        xhat = xhat.T
        u = u.T
        sigma_squared = sigma_squared.T
        proj_for_u, proj_div, covariance_basis = self.proj_Matrix_acquire()

        # first term
        squareTerm = self.square_Term(xhat, u, proj_for_u)

        # divergence term
        div = self.monte_Carlo_div(xhat, u, proj_div)

        # trace term
        traceTerm = self.trace_Term(covariance_basis)

        loss = torch.mean(squareTerm + 2 * sigma_squared * div - sigma_squared * traceTerm)

        return loss


