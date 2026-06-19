#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

import numpy as np
import torch


def nmse_db_torch(h_hat, h):
    """Compute NMSE in dB for torch tensors.
    
    :param h_hat: Predicted channels with shape (batch, dim) or (dim, batch).
    :param h: Reference channels with the same shape as h_hat.
    :return: Scalar tensor containing 10 * log10(||h_hat - h||_2^2 / ||h||_2^2).
    """
    with torch.no_grad():
        val = torch.sum(torch.abs(h_hat - h) ** 2) / torch.sum(torch.abs(h) ** 2).clamp_min(1e-12)
        return 10 * torch.log10(val)


def nmse_db_numpy(h_hat, h):
    """Compute NMSE in dB for numpy arrays.
    
    :param h_hat: Predicted channels with shape (batch, dim) or (dim, batch).
    :param h: Reference channels with the same shape as h_hat.
    :return: Scalar NMSE value in dB.
    """
    val = np.sum(np.abs(h_hat - h) ** 2) / max(np.sum(np.abs(h) ** 2), 1e-12)
    return 10 * np.log10(val)


def l1_l2_ratio(x):
    """Compute the mean ||x||_1 / ||x||_2 ratio across a batch.
    
    :param x: Batch-first tensor with shape (batch, dim).
    :return: Scalar tensor containing the mean sparsity proxy.
    """
    with torch.no_grad():
        return torch.mean(torch.linalg.vector_norm(x, ord=1, dim=1) / torch.linalg.vector_norm(x, ord=2, dim=1).clamp_min(1e-12))


def beta_ratio(P, h_hat, h):
    """Compute projected reconstruction error relative to projected channel power.
    
    :param P: Projection matrix with shape (dim, dim).
    :param h_hat: Predicted channels with shape (batch, dim).
    :param h: Reference channels with shape (batch, dim).
    :return: Scalar tensor containing mean ||(h_hat - h)P.T||_2 / mean ||hP.T||_2.
    """
    with torch.no_grad():
        num = torch.mean(torch.linalg.vector_norm((h_hat - h) @ P.T, ord=2, dim=1))
        den = torch.mean(torch.linalg.vector_norm(h @ P.T, ord=2, dim=1)).clamp_min(1e-12)
        return num / den


def timed_call(fn, *args, cuda_device=None, **kwargs):
    """Call a function and measure wall-clock runtime.
    
    :param fn: Callable to execute.
    :param args: Positional arguments passed to fn.
    :param cuda_device: Optional CUDA device synchronized before and after the call.
    :param kwargs: Keyword arguments passed to fn.
    :return: Tuple (fn_output, elapsed_seconds).
    """
    if cuda_device is not None and torch.cuda.is_available():
        torch.cuda.synchronize(cuda_device)
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    if cuda_device is not None and torch.cuda.is_available():
        torch.cuda.synchronize(cuda_device)
    return out, time.perf_counter() - start
