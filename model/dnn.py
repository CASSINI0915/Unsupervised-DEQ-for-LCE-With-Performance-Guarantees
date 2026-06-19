#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict

import torch
import torch.nn as nn


device = None
def set_device(dev):
    """Set the module device selected by main.py/test.py."""
    global device
    device = dev


class DNN(nn.Module):
    """Fully connected channel estimator for sufficient-statistic inputs."""

    def __init__(self, dim_in=256, dim_out=256):
        """Fully-connected channel estimator with 4 hidden layers.

        The input is the sufficient statistic u = A.T * inv(Sigma) * y in real
        equivalent form, arranged as (batch, 2N). The output is h_hat with the
        same column-wise layout.

        :param dim_in: Complex input dimension N before real-equivalent stacking.
        :param dim_out: Complex output dimension N before real-equivalent stacking.
        :return: None.
        """
        super().__init__()
        self.model = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(dim_in * 2, 2048)),
            ("relu1", nn.ReLU()),
            ("linear2", nn.Linear(2048, 2048)),
            ("relu2", nn.ReLU()),
            ("linear3", nn.Linear(2048, 2048)),
            ("relu3", nn.ReLU()),
            ("linear4", nn.Linear(2048, 2048)),
            ("relu4", nn.ReLU()),
            ("linear5", nn.Linear(2048, dim_out * 2)),
        ]))
        self.depth = 5
        self.eps = None

    def name(self):
        """Return the model name used by logging/checkpoints."""
        return "dnn"

    def device(self):
        """Return the device of the network parameters."""
        return next(self.parameters()).device

    def forward(self, u):
        """Map batch-first sufficient statistics to channel estimates.

        :param u: Real-equivalent sufficient statistic with shape (batch, 2N).
        :return: Real-equivalent channel estimate with shape (batch, 2N).
        """
        return self.model(u)
