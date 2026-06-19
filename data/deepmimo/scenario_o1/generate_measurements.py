#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


SNR_LIST = [0, 5, 10, 15, 20]
A_TYPES = ["gaussian", "bernoulli", "unitary"]
NUM_MEASUREMENTS = 128
NUM_ANTENNAS = 256


def root_dir():
    """Return the project root directory.

    :return: Path to the tsp_deq_gsure root.
    """
    return Path(__file__).resolve().parents[3]


def load_A(A_type):
    """Load one complex sensing matrix.

    :param A_type: Matrix type in gaussian, bernoulli or unitary.
    :return: Complex array with shape (128, 256).
    """
    path = root_dir() / "data" / "sensing_matrix" / "matrices" / f"A_{A_type}{NUM_ANTENNAS}{NUM_MEASUREMENTS}.mat"
    return loadmat(path)["A"]


def channel_path(split):
    """Return a DeepMIMO O1 channel path.

    :param split: Dataset split.
    :return: Path to the channel .mat file.
    """
    return Path(__file__).resolve().parent / "channels" / f"deepmimo_o1_28_{split}_channels.mat"


def output_path(A_type, split, snr):
    """Return the output path for one O1 measurement file.

    :param A_type: Matrix type in gaussian, bernoulli or unitary.
    :param split: Dataset split.
    :param snr: Fixed SNR in dB.
    :return: Path to the measurement .mat file.
    """
    out_dir = Path(__file__).resolve().parent / "measurements" / A_type
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"deepmimo_o1_28_{split}_128_measurements_{snr}dB.mat"


def generate_measurements(h, A, snr_db):
    """Generate batch-first noisy measurements y = A(h + n).

    :param h: Complex channel array with shape (batch, 256).
    :param A: Complex sensing matrix with shape (128, 256).
    :param snr_db: Fixed SNR in dB.
    :return: Tuple (y, sigma_squared, Sigma).
    """
    snr_linear = 10 ** (snr_db / 10)
    channel_power = np.sum(np.abs(h) ** 2, axis=1, keepdims=True)
    sigma_squared = channel_power / (h.shape[1] * snr_linear)
    noise_h = np.sqrt(sigma_squared / 2) * (
        np.random.randn(*h.shape) + 1j * np.random.randn(*h.shape)
    )
    y = (h + noise_h) @ A.T
    Sigma = A @ A.conj().T
    return y, sigma_squared, Sigma


def main():
    """Generate fixed-SNR DeepMIMO O1 measurements.

    :return: None.
    """
    for A_type in A_TYPES:
        A = load_A(A_type)
        for split in ["training", "val", "test"]:
            h = loadmat(channel_path(split))["h"]
            for snr in SNR_LIST:
                y, sigma_squared, Sigma = generate_measurements(h, A, snr)
                savemat(output_path(A_type, split, snr), {
                    "y": y,
                    "sigma_squared": sigma_squared,
                    "Sigma": Sigma,
                })
                print(f"Saved {output_path(A_type, split, snr)}")


if __name__ == "__main__":
    main()
