#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


SNR_LIST = [0, 5, 10, 15, 20]
NUM_MEASUREMENTS = 128
NUM_ANTENNAS = 256
SCENES = [
    "scene1to96_bs1grid3",
    "scene97to196_bs1grid1",
    "scene197to296_bs1grid1",
    "scene297to345_bs1grid1",
    "scene346to445_bs2grid1",
    "scene446to545_bs2grid1",
    "scene546to617_bs2grid1",
    "scene618to713_bs2grid2",
]


def root_dir():
    """Return the project root directory.

    :return: Path to the tsp_deq_gsure root.
    """
    return Path(__file__).resolve().parents[3]


def load_A():
    """Load the unitary sensing matrix.

    :return: Complex array with shape (128, 256).
    """
    path = root_dir() / "data" / "sensing_matrix" / "matrices" / f"A_unitary{NUM_ANTENNAS}{NUM_MEASUREMENTS}.mat"
    return loadmat(path)["A"]


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
    """Generate fixed-SNR DeepMIMO O2 measurements.

    :return: None.
    """
    base = Path(__file__).resolve().parent
    out_dir = base / "measurements" / "unitary"
    out_dir.mkdir(parents=True, exist_ok=True)
    A = load_A()
    for scene in SCENES:
        h = loadmat(base / "channels" / f"deepmimo_o2_3p5_{scene}.mat")["h"]
        for snr in SNR_LIST:
            y, sigma_squared, Sigma = generate_measurements(h, A, snr)
            out = out_dir / f"deepmimo_o2_3p5_{scene}_128_measurements_{snr}dB.mat"
            savemat(out, {"y": y, "sigma_squared": sigma_squared, "Sigma": Sigma})
            print(f"Saved {out}")


if __name__ == "__main__":
    main()
