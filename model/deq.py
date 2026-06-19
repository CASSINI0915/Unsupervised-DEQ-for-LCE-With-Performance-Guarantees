#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Holt"

import torch
import torch.nn as nn


__doc__ = """
Deep equilibrium channel estimator for sufficient-statistic inputs.

The model maps u = A.T @ inv(Sigma) @ y to h_hat by iterating a learned
linear-plus-nonlinear fixed-point operator. It supports supervised NMSE
training and unsupervised GSURE training.
"""


device = None

def set_device(dev):
    """Set the module device."""
    global device
    device = dev


measurements = torch.Tensor
latent_solution = torch.Tensor


class DEQ(nn.Module):
    """Deep equilibrium channel estimator for sufficient-statistic inputs."""

    def __init__(self, A, Sigma, lat_layers=4, contraction_factor=0.99, eps=1.0e-2, max_depth=15, num_channels=32, bias=False):
        """Initialize the DEQ fixed-point channel estimator.

        :param A: Real sensing matrix with shape (2M, 2N).
        :param Sigma: Real covariance basis with shape (2M, 2M).
        :param lat_layers: Number of residual convolution blocks in the nonlinear estimator.
        :param contraction_factor: Target local contraction factor for the safeguard.
        :param eps: Fixed-point stopping tolerance.
        :param max_depth: Maximum number of fixed-point iterations.
        :param num_channels: Hidden Conv1d channel count.
        :param bias: Whether Conv1d and LayerNorm modules use bias terms.
        :return: None.
        """
        super().__init__()
        self.A = A.to(device)
        self.Sigma = Sigma.to(device)
        self.inv_Sigma = torch.linalg.inv(self.Sigma).to(device)
        self.ATinvSigmaA = torch.matmul(self.A.T, torch.matmul(self.inv_Sigma, self.A)).to(device)
        self.pinvA = torch.linalg.pinv(self.A).to(device)
        self.P = torch.matmul(self.pinvA, self.A).to(device)
        self.M, self.N = self.A.shape

        self.step = 2.0 / torch.max(torch.linalg.eigvalsh(self.ATinvSigmaA))
        self._lat_layers = lat_layers
        self.gamma = contraction_factor
        self.eps = eps
        self.max_depth = max_depth
        self.depth = 0.0

        self.record_layerwise = False
        self.layerwise_h = []

        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.05, inplace=True)
        self.unflatten = nn.Unflatten(1, (2, 256))

        self.use_contraction_safeguard = True

        self.input_convs = nn.Conv1d(2, num_channels, kernel_size=9, stride=1, padding=(9 - 1)//2, bias=bias)
        self.input_layer_norm = nn.LayerNorm([num_channels, 256], bias=bias)
        self.latent_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(num_channels, num_channels, kernel_size=9, stride=1, padding=(9 - 1)//2, bias=bias),
                self.relu,
                nn.Conv1d(num_channels, num_channels, kernel_size=9, stride=1, padding=(9 - 1)//2, bias=bias),
                self.relu,
            )
            for _ in range(lat_layers)
        ])
        self.latent_layer_norm = nn.ModuleList([nn.LayerNorm([num_channels, 256], bias=bias)])
        self.output_convs = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=1, stride=1, bias=bias),
            self.leaky_relu,
            nn.Conv1d(num_channels, 2, kernel_size=1, stride=1, bias=bias),
        )

    def name(self):
        """Return the model name."""
        return "deq"

    def device(self):
        """Return the device of the network parameters."""
        return next(self.parameters()).device

    def linear_estimator(self, u: measurements, h: latent_solution) -> latent_solution:
        """Apply one gradient-like linear step h + step * (u - A.T @ inv(Sigma) @ A @ h).

        :param u: Column-wise sufficient statistic with shape (2N, batch).
        :param h: Column-wise latent channel iterate with shape (2N, batch).
        :return: Updated latent iterate with shape (2N, batch).
        """
        return self.step * (u - torch.matmul(self.ATinvSigmaA, h)) + h

    def nonlinear_estimator(self, h: latent_solution) -> latent_solution:
        """Apply the learned Conv1d nonlinear estimator g(h).

        :param h: Column-wise latent channel tensor with shape (2N, batch).
        :return: Column-wise nonlinear estimate with shape (2N, batch).
        """
        batch_size = h.shape[1]
        h = self.unflatten(h.T)
        h = self.leaky_relu(self.input_convs(h))
        h = self.input_layer_norm(h)

        residual = h
        for conv in self.latent_convs:
            residual = conv(residual)
        h = self.latent_layer_norm[0](h + residual)

        h = self.output_convs(h)
        return h.view(batch_size, -1).T

    def latent_space_forward(self, u: measurements, h: latent_solution) -> latent_solution:
        """Apply one fixed-point update f(h; u) = nonlinear_estimator(linear_estimator(u, h)).

        :param u: Column-wise sufficient statistic with shape (2N, batch).
        :param h: Column-wise latent channel iterate with shape (2N, batch).
        :return: Updated latent channel estimate with shape (2N, batch).
        """
        h = self.linear_estimator(u, h)
        return self.nonlinear_estimator(h)

    def _scale_conv(self, conv, factor):
        """Scale a convolution layer's weights and optional bias in place.

        :param conv: Conv1d layer to scale.
        :param factor: Multiplicative scale factor.
        :return: None.
        """
        conv.weight.data.mul_(factor)
        if conv.bias is not None:
            conv.bias.data.mul_(factor)

    def normalize_lip_const(self, u: measurements, h: latent_solution):
        """Rescale the nonlinear estimator when a local perturbation violates the target contraction.

        The DEQ iteration is expected to be contractive around the current fixed-point iterate. This
        routine draws a random perturbation of h and compares the response of one latent update with
        the perturbation magnitude. If the measured ratio is larger than gamma, the convolutional
        weights are scaled down so the local fixed-point map is closer to the desired contraction.

        :param u: Sufficient statistic in column-major form, shape=(dim, batch).
        :param h: Current latent iterate in column-major form, shape=(dim, batch).
        :return: None.
        """
        noise_h = torch.randn_like(h)
        h_hat1 = self.latent_space_forward(u, h + noise_h)
        h_hat2 = self.latent_space_forward(u, h)
        diff_norm = torch.mean(torch.linalg.vector_norm(h_hat1 - h_hat2, ord=2, dim=0))
        noise_norm = torch.mean(torch.linalg.vector_norm(noise_h, ord=2, dim=0)).clamp_min(1e-12)
        if diff_norm <= self.gamma * noise_norm:
            return

        normalize_factor = (self.gamma * noise_norm / diff_norm.clamp_min(1e-12)) ** (1.0 / (2 * self._lat_layers))
        for conv in self.latent_convs:
            self._scale_conv(conv[0], normalize_factor)
            self._scale_conv(conv[2], normalize_factor)
        self._scale_conv(self.input_convs, normalize_factor)
        self._scale_conv(self.output_convs[0], normalize_factor)
        self._scale_conv(self.output_convs[2], normalize_factor)

    @torch.no_grad()
    def l2_estimate(self, u: measurements, h_star: latent_solution, sigma: float = 1.0, eps_denom: float = 1e-12) -> torch.Tensor:
        """Estimate the Lipschitz constant of nonlinear_estimator around h_star.

            L_hat = sum_i || g(h* + delta_i) - g(h*) ||_2 / sum_i ||delta_i||_2
            where delta_i ~ N(0, sigma^2 I).

        :param u: Sufficient statistic, shape=(batch, dim).
        :param h_star: Fixed-point channel estimate, shape=(batch, dim).
        :param sigma: Standard deviation of Gaussian perturbation.
        :param eps_denom: Small positive value used to avoid division by zero.
        :return: Scalar tensor containing the estimated Lipschitz ratio.
        """
        h_star = h_star.T
        delta = sigma * torch.randn_like(h_star)
        g0 = self.nonlinear_estimator(h_star)
        g1 = self.nonlinear_estimator(h_star + delta)
        num = torch.linalg.vector_norm(g1 - g0, ord=2, dim=0)
        den = torch.linalg.vector_norm(delta, ord=2, dim=0)
        return num.sum() / den.sum().clamp_min(eps_denom)

    @torch.no_grad()
    def l1andl2_estimate(self, u: measurements, h_star: latent_solution, sigma: float = 1.0, eps_denom: float = 1e-12):
        """Estimate the Lipschitz constant of the full fixed-point operator f_Theta(h; u).

        The operator is implemented by latent_space_forward, i.e. LE followed by NLE:
            f(h;u) := latent_space_forward(u, h)
            L_hat = sum_i || f(h* + delta_i;u) - f(h*;u) ||_2 / sum_i ||delta_i||_2

        :param u: Sufficient statistic, shape=(batch, dim).
        :param h_star: Fixed-point channel estimate, shape=(batch, dim).
        :param sigma: Standard deviation of Gaussian perturbation.
        :param eps_denom: Small positive value used to avoid division by zero.
        :return: Scalar tensor containing the estimated Lipschitz ratio.
        """
        u = u.T
        h_star = h_star.T
        delta = sigma * torch.randn_like(h_star)
        f0 = self.latent_space_forward(u, h_star)
        f1 = self.latent_space_forward(u, h_star + delta)
        num = torch.linalg.vector_norm(f1 - f0, ord=2, dim=0)
        den = torch.linalg.vector_norm(delta, ord=2, dim=0)
        return num.sum() / den.sum().clamp_min(eps_denom)

    def forward(self, u: measurements, depth_warning=False):
        """
        Estimate h from batch-first sufficient statistics.

        The public input and output shape is (batch, 2N). Internally the fixed-point
        iterations use column-major tensors with shape (2N, batch), starting from
        the zero vector and stopping when the iterate residual is below eps or
        max_depth is reached.

        :param u: Batch-first sufficient statistic, shape=(batch, 2N).
        :param depth_warning: Whether to print a warning when max_depth is reached.
        :return: Estimated channel h_hat, shape=(batch, 2N).
        """
        u = u.T
        with torch.no_grad():
            self.depth = 0.0
            h = torch.zeros(self.N, u.shape[1], dtype=u.dtype, device=self.device())

            if (not self.training) and self.record_layerwise:
                self.layerwise_h = []

            h_prev = torch.empty_like(h).fill_(float("inf"))
            termination = False
            while not termination and self.depth < self.max_depth:
                h_prev = h.clone()
                h = self.latent_space_forward(u, h)

                if (not self.training) and self.record_layerwise:
                    self.layerwise_h.append(h.T.detach().cpu())

                res_norm = torch.linalg.vector_norm(h - h_prev, ord=2, dim=0)
                max_res_norm = torch.max(res_norm)
                self.depth += 1.0
                termination = max_res_norm <= self.eps

            if self.training and self.use_contraction_safeguard:
                self.normalize_lip_const(u, h_prev)

        if self.depth >= self.max_depth and depth_warning:
            print("\nWarning: Max Depth Reached - Break Forward Loop\n")

        h = self.latent_space_forward(u, h)
        if self.training:
            return h.T

        if self.record_layerwise:
            self.layerwise_h.append(h.T.detach().cpu())
        return h.T.detach()
