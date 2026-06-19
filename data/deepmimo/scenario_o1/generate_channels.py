#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
from scipy.io import savemat


NUM_ANTENNAS = 256
TRAIN_SIZE = 80000
VAL_SIZE = 2000
TEST_SIZE = 2000


def scenario_root():
    """Return the DeepMIMO O1 support directory.

    :return: Path to the deepmimo/O1_28 directory.
    """
    return Path(__file__).resolve().parents[1]


def output_dir():
    """Return the directory used to save generated channel files.

    :return: Path to scenario_o1/channels.
    """
    path = Path(__file__).resolve().parent / "channels"
    path.mkdir(parents=True, exist_ok=True)
    return path


def space_to_angular(data):
    """Transform spatial-domain channels to angular-domain channels.

    :param data: Complex channel array with shape (batch, 256).
    :return: Complex angular-domain array with shape (batch, 256).
    """
    d = 0.5
    a = np.arange(-(NUM_ANTENNAS - 1), NUM_ANTENNAS, 2).reshape(-1, 1)
    dft = (1 / np.sqrt(NUM_ANTENNAS)) * np.exp(1j * 2 * np.pi * a * a.T / 2 * d * (1 / NUM_ANTENNAS))
    return data @ dft.T


def preprocess(data):
    """Normalize and standardize each channel sample.

    :param data: Real array with shape (batch, 256).
    :return: Preprocessed real array with shape (batch, 256).
    """
    data_min = np.min(data, axis=1, keepdims=True)
    data_max = np.max(data, axis=1, keepdims=True)
    data = (data - data_min) / (data_max - data_min)
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    return (data - mean) / std


def save_split(h):
    """Save train, val and test splits as batch-first channel arrays.

    :param h: Complex channel array with shape (batch, 256).
    :return: None.
    """
    if h.shape[0] < TRAIN_SIZE + VAL_SIZE + TEST_SIZE:
        raise ValueError("Not enough DeepMIMO O1 samples for the requested split sizes.")
    perm = np.random.permutation(h.shape[0])
    train_idx = perm[:TRAIN_SIZE]
    val_idx = perm[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
    test_idx = perm[TRAIN_SIZE + VAL_SIZE:TRAIN_SIZE + VAL_SIZE + TEST_SIZE]
    out = output_dir()
    savemat(out / "deepmimo_o1_28_training_channels.mat", {"h": h[train_idx]})
    savemat(out / "deepmimo_o1_28_val_channels.mat", {"h": h[val_idx]})
    savemat(out / "deepmimo_o1_28_test_channels.mat", {"h": h[test_idx]})


def main():
    """Generate DeepMIMO O1 channels and save batch-first splits.

    :return: None.
    """
    try:
        import DeepMIMO
    except ImportError as exc:
        raise ImportError("DeepMIMO is required to regenerate O1 channels.") from exc

    params = DeepMIMO.default_params()
    params["scenario"] = "O1_28"
    params["dataset_folder"] = str(scenario_root())
    params["active_BS"] = np.arange(3, 7)
    params["user_row_first"] = 1000
    params["user_row_last"] = 1300
    params["enable_BS2BS"] = False
    params["num_paths"] = 3
    params["bs_antenna"]["shape"] = np.array([1, NUM_ANTENNAS, 1])
    params["ue_antenna"]["shape"] = np.array([1, 1, 1])
    params["OFDM"]["subcarriers_limit"] = 1
    params["OFDM"]["subcarriers_sampling"] = 1

    dataset = DeepMIMO.generate_data(params)
    rows = []
    for bs in range(len(params["active_BS"])):
        users = dataset[bs]["user"]["channel"]
        for user_idx in range(users.shape[0]):
            rows.append(users[user_idx][:, :, 0].reshape(-1))
    h = np.asarray(rows)
    h = space_to_angular(h)
    h.real = preprocess(h.real)
    h.imag = preprocess(h.imag)
    save_split(h)


if __name__ == "__main__":
    main()
