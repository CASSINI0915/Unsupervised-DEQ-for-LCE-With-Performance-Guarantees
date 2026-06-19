#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


SNR_LIST = [0, 5, 10, 15, 20]
NUM_MEASUREMENTS = 128
NUM_ANTENNAS = 256


def root_dir():
    """Return the project root directory.

    :return: Path to the tsp_deq_gsure root.
    """
    return Path(__file__).resolve().parents[2]


def load_A():
    """Load the unitary sensing matrix.

    :return: Complex array with shape (128, 256).
    """
    path = root_dir() / "data" / "sensing_matrix" / "matrices" / f"A_unitary{NUM_ANTENNAS}{NUM_MEASUREMENTS}.mat"
    return loadmat(path)["A"]


def channel_path(split):
    """Return a synthetic channel path.

    :param split: Dataset split.
    :return: Path to the synthetic channel .mat file.
    """
    return Path(__file__).resolve().parent / "channels" / f"synthetic_{split}_channels.mat"


def load_channel(path):
    """Load a batch-first synthetic channel matrix.

    :param path: Path to a channel .mat file.
    :return: Complex array with shape (batch, 256).
    """
    data = loadmat(path)
    if "h" not in data:
        raise KeyError(f"{path} must contain variable 'h'.")
    return data["h"]


def output_path(split, snr):
    """Return the output path for one synthetic measurement file.

    :param split: Dataset split.
    :param snr: Fixed SNR in dB.
    :return: Path to the measurement .mat file.
    """
    out_dir = Path(__file__).resolve().parent / "measurements" / "unitary"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"synthetic_{split}_128_measurements_{snr}dB.mat"


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
    """Generate fixed-SNR synthetic k-sparse measurements.

    :return: None.
    """
    A = load_A()
    for split in ["training", "val", "test"]:
        h = load_channel(channel_path(split))
        for snr in SNR_LIST:
            y, sigma_squared, Sigma = generate_measurements(h, A, snr)
            savemat(output_path(split, snr), {"y": y, "sigma_squared": sigma_squared, "Sigma": Sigma})
            print(f"Saved {output_path(split, snr)}")


if __name__ == "__main__":
    main()
