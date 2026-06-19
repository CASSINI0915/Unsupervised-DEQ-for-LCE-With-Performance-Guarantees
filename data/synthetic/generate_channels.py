#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
from scipy.io import savemat


def generate_random_support(n, samples, k):
    """Generate exact k-sparse support masks.

    :param n: Signal dimension.
    :param samples: Number of samples.
    :param k: Number of active entries per sample.
    :return: Boolean support matrix with shape (n, samples).
    """
    support = np.zeros((n, samples), dtype=bool)
    for col in range(samples):
        indices = np.random.choice(n, k, replace=False)
        support[indices, col] = True
    return support


def k_sparse_channel_generate(n, k, var, samples, mode="exact"):
    """Generate k-sparse complex channel samples.

    :param n: Signal dimension.
    :param k: Sparsity level.
    :param var: Active-entry variance.
    :param samples: Number of samples.
    :param mode: exact or inexact support mode.
    :return: Complex channel matrix with shape (n, samples).
    """
    if mode == "exact":
        support = generate_random_support(n, samples, k)
        x = np.sqrt(var / 2) * (
            np.random.randn(n, samples) + 1j * np.random.randn(n, samples)
        ) * support
    elif mode == "inexact":
        # Gaussian - Bernoulli
        support = np.random.rand(n, samples) < k / n
        x = np.sqrt(var / 2) * (
            np.random.randn(n, samples) + 1j * np.random.randn(n, samples)
        ) * support
    else:
        raise ValueError("mode must be either 'exact' or 'inexact'.")
    return x


def preprocess(data):
    """Preprocess nonzero entries sample by sample.

    :param data: Real array with shape (n, samples).
    :return: Preprocessed real array with shape (n, samples).
    """
    non_zero_mask = data != 0
    min_values = np.where(non_zero_mask, data, np.inf).min(axis=0)
    max_values = np.where(non_zero_mask, data, -np.inf).max(axis=0)
    normalized = np.where(non_zero_mask, (data - min_values) / (max_values - min_values), 0)
    mean_values = np.where(non_zero_mask, normalized, 0).sum(axis=0) / non_zero_mask.sum(axis=0)
    std_values = np.sqrt(
        np.where(non_zero_mask, (normalized - mean_values) ** 2, 0).sum(axis=0) / data.shape[0]
    )
    return np.where(non_zero_mask, (normalized - mean_values) / std_values, 0)


def save_split(out_dir, split, samples, n, k, var, mode):
    """Generate and save one batch-first synthetic split.

    :param out_dir: Output directory.
    :param split: Dataset split name.
    :param samples: Number of samples in this split.
    :param n: Signal dimension.
    :param k: Sparsity level.
    :param var: Active-entry variance.
    :param mode: exact or inexact support mode.
    :return: None.
    """
    h = k_sparse_channel_generate(n, k, var, samples, mode)
    h.real = preprocess(h.real)
    h.imag = preprocess(h.imag)
    path = out_dir / f"synthetic_{split}_channels.mat"
    savemat(path, {"h": h.T})
    print(f"Saved {path}")


def main():
    """Generate all synthetic k-sparse channel splits.

    :return: None.
    """
    parser = argparse.ArgumentParser(description="Generate k-sparse synthetic channel data.")
    parser.add_argument("--N", type=int, default=256)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--training_size", type=int, default=80000)
    parser.add_argument("--val_size", type=int, default=2000)
    parser.add_argument("--test_size", type=int, default=2000)
    parser.add_argument("--mode", choices=["exact", "inexact"], default="exact")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    np.random.seed(args.seed)
    out_dir = Path(__file__).resolve().parent / "channels"
    out_dir.mkdir(parents=True, exist_ok=True)
    var = args.N / args.k

    save_split(out_dir, "training", args.training_size, args.N, args.k, var, args.mode)
    save_split(out_dir, "val", args.val_size, args.N, args.k, var, args.mode)
    save_split(out_dir, "test", args.test_size, args.N, args.k, var, args.mode)


if __name__ == "__main__":
    main()
