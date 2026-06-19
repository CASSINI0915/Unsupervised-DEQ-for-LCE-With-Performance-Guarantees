#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Holt'

import torch
import torch.nn as nn

__doc__ = """
LDGEC baseline reconstructed from:
H. He, R. Wang, W. Jin, S. Jin, C.-K. Wen, and G. Y. Li,
"Beamspace Channel Estimation for Wideband Millimeter-Wave MIMO:
A Model-Driven Unsupervised Learning Approach," IEEE Trans. Wireless
Commun., vol. 22, no. 3, pp. 1808-1822, 2022.

The model is an unfolded denoising-based GEC network. In this repository,
one LDGEC iteration is implemented as four computational blocks:
Module_A estimates the z-domain posterior/extrinsic variables, Module_C1
maps z-domain messages to h-domain denoiser inputs, Module_B applies a
DnCNN denoiser and estimates h-domain extrinsic variables, and Module_C2
maps the denoised h-domain messages back to z-domain variables.

The paper states that source code would be available at
https://github.com/hehengtao/LDGEC, but that repository was unavailable
during this reimplementation. This baseline is therefore an independent
reconstruction from the paper equations and empirical debugging, not an
official reproduction of the authors' code.

A fragile point is Algorithm 1, line 10: v1h_post is estimated from the
denoiser divergence multiplied by avg(v1h). For a learned denoiser, this
divergence is approximated by Monte Carlo finite differences and is not
guaranteed to be nonnegative. If the estimate becomes negative or too
small, the subsequent extrinsic variance updates for v1h/v2h can become
invalid and may lead to divergence or NaN gradients.

For the experiments in this repository, conservative LDGEC settings are
recommended for numerical stability: beta = 0.8, 5 epochs for SURE
layer-wise training, 20 epochs for NMSE training, StepLR step_size = 3
for SURE and 12 for NMSE, and gamma = 0.5. These settings are more
conservative than the paper's beta = 0.8 and 50-epoch training setup.
"""

device = None


def set_device(dev):
    """Set the module device."""
    global device
    device = dev


def _current_device():
    """Return the configured LDGEC device, defaulting to CPU."""
    return device if device is not None else torch.device("cpu")


measurements = torch.Tensor
latent_solution = torch.Tensor


class LDGEC(nn.Module):
    """Layered denoising GEC network for raw measurement-domain inputs."""

    def __init__(self, num_layers, A, in_channels=2, out_channels=2, nc=64, nb=20) -> None:
        """Initialize the unfolded LDGEC network.

        :param num_layers: Number of unfolded LDGEC iterations.
        :param A: Real sensing matrix with shape (2M, 2N).
        :param in_channels: Denoiser input channel count after unflattening.
        :param out_channels: Denoiser output channel count after unflattening.
        :param nc: Hidden Conv1d channel count in the DnCNN denoiser.
        :param nb: Number of DnCNN convolution blocks.
        :return: None.
        """
        super(LDGEC, self).__init__()
        self.A = A.to(_current_device())
        self.B = torch.matmul(self.A.conj().t(), self.A).to(_current_device())
        self.M, self.N = self.A.shape
        self.max_depth = num_layers
        self.Layers = nn.ModuleList(
            [
                Single_layer(
                    self.A,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    nc=nc,
                    nb=nb,
                )
                for _ in range(num_layers)
            ]
        )
        self.depth = 1
        self.Ph = 1
        self.Pz = self.Ph * torch.trace(self.B) / self.N
        self.return_state = False

    def device(self):
        """Return the device of model parameters."""
        return next(self.parameters()).device

    def name(self):
        """Return the model name used by training utilities."""
        return "ldgec"

    def forward(self, y: measurements, sigma_squared):
        """Run LDGEC on batch-first raw measurements.

        :param y: Real-equivalent measurement tensor with shape (batch, 2M).
        :param sigma_squared: Noise variance with shape (batch, 1).
        :return: Estimated real-equivalent channels with shape (batch, 2N).
        """
        batch_size = y.shape[0]
        model_device = self.device()
        dtype = y.dtype
        r1z = torch.zeros_like(y, device=model_device)
        v1z = self.Pz.to(device=model_device, dtype=dtype) * torch.ones(
            (batch_size, self.M), device=model_device, dtype=dtype
        )
        r2h = torch.zeros((batch_size, self.N), device=model_device, dtype=dtype)
        v2h = self.Pz.to(device=model_device, dtype=dtype) * torch.ones(
            (batch_size, self.N), device=model_device, dtype=dtype
        )
        h_prev = torch.full((batch_size, self.N), float("inf"), device=model_device, dtype=dtype)

        for single_layer in self.Layers[: self.depth]:
            h1_hat, r1z, v1z, r2h, v2h = single_layer(y, sigma_squared, r1z, v1z, r2h, v2h)
            h_prev = h1_hat

        if self.return_state:
            return h_prev, r1z, v1z, r2h, v2h
        return h_prev


class Single_layer(nn.Module):
    """One unfolded LDGEC iteration with modules A, B, C1 and C2."""

    def __init__(self, A, in_channels=2, out_channels=2, nc=64, nb=20) -> None:
        """Initialize one unfolded LDGEC iteration.

        :param A: Real sensing matrix with shape (2M, 2N).
        :param in_channels: Denoiser input channel count after unflattening.
        :param out_channels: Denoiser output channel count after unflattening.
        :param nc: Hidden Conv1d channel count in the DnCNN denoiser.
        :param nb: Number of DnCNN convolution blocks.
        :return: None.
        """
        super(Single_layer, self).__init__()
        self.A = A.to(_current_device())
        self.device = _current_device()
        self.beta = 0.8
        self.denoiser = Denoiser(
            in_channels=in_channels,
            out_channels=out_channels,
            nc=nc,
            nb=nb,
        )

    def monte_Carlo_div(self, h1_hat, r1h):
        """Estimate denoiser divergence with a Monte Carlo perturbation.

        :param h1_hat: Denoised h estimate with shape (batch, 2N).
        :param r1h: Noisy h-domain denoiser input with shape (batch, 2N).
        :return: Divergence estimate with shape (batch,).
        """
        epsilon = torch.maximum(0.001 * torch.max(torch.abs(r1h)), r1h.new_tensor(0.00001))
        eta = torch.randn_like(r1h)

        u_perturbed = r1h + eta * epsilon
        x_hat_perturbed = self.denoiser(u_perturbed)
        del_x_hat_perturbed = x_hat_perturbed - h1_hat
        eta_dx = torch.sum(eta * del_x_hat_perturbed, 1)
        return eta_dx / epsilon

    def _linear_h_posterior(self, r2z, v2z, r2h, v2h):
        """Compute the h posterior with the Woodbury broadcast form.

        :param r2z: z-domain extrinsic mean with shape (batch, 2M).
        :param v2z: z-domain extrinsic variance with shape (batch, 2M).
        :param r2h: h-domain extrinsic mean with shape (batch, 2N).
        :param v2h: h-domain extrinsic variance with shape (batch, 2N).
        :return: Tuple (h2_hat, Q2h, A) with h2_hat shape (batch, 2N).
        """
        A = self.A.to(r2z.device).unsqueeze(0)
        A_H = A.conj().transpose(-2, -1)

        A_v2h = A * v2h.unsqueeze(1)
        ADAH = torch.matmul(A_v2h, A_H)
        D_AH = v2h.unsqueeze(-1) * A_H
        middle_inv = torch.linalg.inv(torch.diag_embed(v2z) + ADAH)
        Q2h = torch.diag_embed(v2h) - torch.matmul(torch.matmul(D_AH, middle_inv), A_v2h)

        r2h_div_v2h = (r2h / v2h).unsqueeze(-1)
        r2z_div_v2z = (r2z / v2z).unsqueeze(-1)
        rhs = r2h_div_v2h + torch.matmul(A_H, r2z_div_v2z)
        h2_hat = torch.matmul(Q2h, rhs).squeeze(-1)
        return h2_hat, Q2h, A

    def Module_A(self, r1z, v1z, y, sigma_squared):
        """Compute z posterior and extrinsic information from y = z + noise.

        :param r1z: Prior z mean with shape (batch, 2M).
        :param v1z: Prior z variance with shape (batch, 2M).
        :param y: Raw measurement tensor with shape (batch, 2M).
        :param sigma_squared: Noise variance with shape (batch, 1).
        :return: Tuple (r2z, v2z), each with shape (batch, 2M).
        """
        sigma_squared = sigma_squared.expand_as(v1z)
        z1_hat = r1z + (v1z / (v1z + sigma_squared)) * (y - r1z)
        v1z_post = v1z - v1z ** 2 / (v1z + sigma_squared)

        avg_v1z_post = torch.mean(v1z_post, 1, keepdim=True).expand_as(z1_hat)
        avg_v1z_post = torch.clamp(avg_v1z_post, min=5e-7)
        v2z = 1 / (1 / avg_v1z_post - 1 / v1z)
        r2z = v2z * (z1_hat / avg_v1z_post - r1z / v1z)
        return r2z, v2z

    def Module_C1(self, r2z, v2z, r2h, v2h):
        """Map z-domain extrinsic information to h-domain denoiser input.

        :param r2z: z-domain extrinsic mean with shape (batch, 2M).
        :param v2z: z-domain extrinsic variance with shape (batch, 2M).
        :param r2h: h-domain extrinsic mean with shape (batch, 2N).
        :param v2h: h-domain extrinsic variance with shape (batch, 2N).
        :return: Tuple (r1h, v1h), each with shape (batch, 2N).
        """
        h2_hat, Q2h, _ = self._linear_h_posterior(r2z, v2z, r2h, v2h)

        dQ2h = torch.diagonal(Q2h, offset=0, dim1=-2, dim2=-1)
        dQ2h = torch.mean(dQ2h, 1, keepdim=True).expand_as(h2_hat)
        v1h = 1 / (1 / dQ2h - 1 / v2h)
        r1h = v1h * (h2_hat / dQ2h - r2h / v2h)
        return r1h, v1h

    def Module_C2(self, r1z, v1z, r2z, v2z, r2h, v2h):
        """Map h-domain denoiser output back to z-domain input information.

        :param r1z: Previous z-domain input mean with shape (batch, 2M).
        :param v1z: Previous z-domain input variance with shape (batch, 2M).
        :param r2z: z-domain extrinsic mean with shape (batch, 2M).
        :param v2z: z-domain extrinsic variance with shape (batch, 2M).
        :param r2h: h-domain extrinsic mean with shape (batch, 2N).
        :param v2h: h-domain extrinsic variance with shape (batch, 2N).
        :return: Tuple (r1z, v1z), each with shape (batch, 2M).
        """
        h2_hat, Q2h, A = self._linear_h_posterior(r2z, v2z, r2h, v2h)

        Q2z = A @ Q2h @ A.transpose(1, 2)
        z2_hat = torch.matmul(h2_hat, self.A.to(h2_hat.device).T)

        dQ2z = torch.diagonal(Q2z, offset=0, dim1=-2, dim2=-1)
        dQ2z = torch.mean(dQ2z, 1, keepdim=True).expand_as(z2_hat)
        v1z = self.beta * torch.clamp(1 / (1 / dQ2z - 1 / v2z), min=5e-7) + (1 - self.beta) * v1z
        r1z = self.beta * v1z * (z2_hat / dQ2z - r2z / v2z) + (1 - self.beta) * r1z
        return r1z, v1z

    def Module_B(self, r1h, v1h):
        """Apply the denoiser and compute h-domain extrinsic information.

        :param r1h: h-domain denoiser input mean with shape (batch, 2N).
        :param v1h: h-domain denoiser input variance with shape (batch, 2N).
        :return: Tuple (h1_hat, r2h, v2h), each with shape (batch, 2N).
        """
        feature_dim = r1h.shape[1]
        h1_hat = self.denoiser(r1h)
        v1h_post = (
            self.monte_Carlo_div(h1_hat, r1h)
            * torch.mean(v1h, 1)
            / feature_dim
        ).unsqueeze(-1).expand_as(h1_hat)

        avg_v1h_post = torch.mean(v1h_post, 1, keepdim=True).expand_as(h1_hat)
        v2h = 1 / (1 / avg_v1h_post - 1 / v1h)
        r2h = v2h * (h1_hat / avg_v1h_post - r1h / v1h)
        return h1_hat, r2h, v2h

    def forward(self, y: measurements, sigma_squared, r1z, v1z, r2h, v2h) -> latent_solution:
        """Run one LDGEC iteration.

        :param y: Raw measurement tensor with shape (batch, 2M).
        :param sigma_squared: Noise variance with shape (batch, 1).
        :param r1z: z-domain input mean with shape (batch, 2M).
        :param v1z: z-domain input variance with shape (batch, 2M).
        :param r2h: h-domain extrinsic mean with shape (batch, 2N).
        :param v2h: h-domain extrinsic variance with shape (batch, 2N).
        :return: Tuple (h1_hat, r1z, v1z, r2h, v2h).
        """
        r2z, v2z = self.Module_A(r1z, v1z, y, sigma_squared)
        r1h, v1h = self.Module_C1(r2z, v2z, r2h, v2h)
        h1_hat, r2h, v2h = self.Module_B(r1h, v1h)
        r1z, v1z = self.Module_C2(r1z, v1z, r2z, v2z, r2h, v2h)
        return h1_hat, r1z, v1z, r2h, v2h


class Denoiser(nn.Module):
    """DnCNN residual denoiser used as the nonlinear LDGEC module."""

    def __init__(self, in_channels=2, out_channels=2, nc=64, nb=20):
        """Initialize the DnCNN residual denoiser.

        :param in_channels: Input channel count after reshaping h to (batch, channels, 256).
        :param out_channels: Output channel count after reshaping.
        :param nc: Hidden Conv1d channel count.
        :param nb: Number of convolution blocks.
        :return: None.
        """
        super(Denoiser, self).__init__()
        self.unflatten = nn.Unflatten(1, (2, 256))
        self.relu = nn.ReLU(inplace=True)
        self.head_convs = nn.Conv1d(
            in_channels=in_channels,
            out_channels=nc,
            kernel_size=9,
            stride=1,
            padding=(9 - 1) // 2,
            bias=True,
        )
        self.body_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_channels=nc, out_channels=nc, kernel_size=9, stride=1, padding=(9 - 1) // 2),
                    nn.BatchNorm1d(nc, momentum=0.09, eps=1e-04, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),
                )
                for _ in range(nb - 2)
            ]
        )
        self.tail_convs = nn.Conv1d(
            in_channels=nc,
            out_channels=out_channels,
            kernel_size=9,
            stride=1,
            padding=(9 - 1) // 2,
            bias=True,
        )

        for body_conv in self.body_convs:
            nn.init.kaiming_normal_(body_conv[0].weight)

    def forward(self, r1h):
        """Return residual denoising output h_hat = r1h - noise.

        :param r1h: Batch-first real-equivalent channel tensor with shape (batch, 2N).
        :return: Denoised channel tensor with shape (batch, 2N).
        """
        batch_size = r1h.shape[0]
        noise = self.head_convs(self.unflatten(r1h))
        noise = self.relu(noise)
        for conv in self.body_convs:
            noise = conv(noise)
        noise = self.tail_convs(noise)
        noise = noise.view(batch_size, -1)
        return r1h - noise
