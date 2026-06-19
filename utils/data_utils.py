#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset

from . import utils as legacy


SNR_LIST = [0, 5, 10, 15, 20]
NUM_MEASUREMENTS = 128
NUM_ANTENNAS = 256
A_TYPES = ["gaussian", "bernoulli", "unitary"]
DEEP_MIMO_O2_SCENES = [
    "scene1to96_bs1grid3",
    "scene97to196_bs1grid1",
    "scene197to296_bs1grid1",
    "scene297to345_bs1grid1",
    "scene346to445_bs2grid1",
    "scene446to545_bs2grid1",
    "scene546to617_bs2grid1",
    "scene618to713_bs2grid2",
]


def project_root():
    """Return the project root directory.

    :return: Path to the tsp_deq_gsure root directory.
    """
    return Path(__file__).resolve().parents[1]


def data_root():
    """Return the data root directory.

    :return: Path to the data directory.
    """
    return project_root() / "data"


def set_device(dev):
    """Forward the global device setting to legacy utilities.

    :param dev: Torch device used by legacy helper code.
    :return: None.
    """
    legacy.set_device(dev)


def complex_matrix_to_real(A):
    """Convert a complex matrix to its equivalent real block matrix.

    :param A: Complex matrix with shape (M, N).
    :return: Real tensor with shape (2M, 2N).
    """
    return torch.from_numpy(np.vstack((
        np.hstack((np.real(A), -np.imag(A))),
        np.hstack((np.imag(A), np.real(A))),
    ))).float()


def complex_vectors_to_real(x):
    """Convert batch-first complex vectors to real vectors.

    :param x: Complex array with shape (batch, dim) or (dim,).
    :return: Real tensor with shape (batch, 2 * dim).
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2-D array, got shape {x.shape}.")
    return torch.from_numpy(np.hstack((np.real(x), np.imag(x)))).float()


def projection_from_A(A):
    """Compute the orthogonal projection matrix induced by A.

    :param A: Real sensing matrix with shape (2M, 2N).
    :return: Projection matrix with shape (2N, 2N).
    """
    temp = A @ A.T
    pinv_A = A.T @ torch.linalg.inv(temp)
    return pinv_A @ A


def sensing_matrix_path(A_type="unitary"):
    """Return the path of a sensing matrix file.

    :param A_type: Matrix type in gaussian, bernoulli or unitary.
    :return: Path to the requested .mat file.
    """
    if A_type not in A_TYPES:
        raise ValueError(f"A_type must be one of {A_TYPES}")
    return data_root() / "sensing_matrix" / "matrices" / f"A_{A_type}{NUM_ANTENNAS}{NUM_MEASUREMENTS}.mat"


def load_complex_A(A_type="unitary"):
    """Load a complex sensing matrix.

    :param A_type: Matrix type in gaussian, bernoulli or unitary.
    :return: Complex numpy array with shape (128, 256).
    """
    path = sensing_matrix_path(A_type)
    if not path.exists():
        raise FileNotFoundError(f"Sensing matrix not found: {path}")
    return loadmat(path)["A"]


def load_cs_matrix(A_type="unitary", num_measurements=NUM_MEASUREMENTS, num_antennas=NUM_ANTENNAS):
    """Load the real sensing matrix and its projection.

    :param A_type: Matrix type in gaussian, bernoulli or unitary.
    :param num_measurements: Number of complex measurements.
    :param num_antennas: Number of complex channel coefficients.
    :return: Tuple (A, P) in equivalent real form.
    """
    if num_measurements != NUM_MEASUREMENTS or num_antennas != NUM_ANTENNAS:
        raise ValueError("This release expects A to have shape 128 x 256.")
    A = complex_matrix_to_real(load_complex_A(A_type))
    return A, projection_from_A(A)


def covariance_to_real(Sigma):
    """Convert a complex covariance matrix to real block form.

    :param Sigma: Complex covariance basis with shape (M, M).
    :return: Real covariance tensor with shape (2M, 2M).
    """
    return complex_matrix_to_real(Sigma)


def load_covariance(measurement_path, finite_snapshot=0.0, diagonal=False):
    """Load and optionally perturb a measurement covariance basis.

    :param measurement_path: Path to a measurement .mat file.
    :param finite_snapshot: Variance of the covariance perturbation.
    :param diagonal: Whether to perturb only diagonal entries.
    :return: Real covariance tensor with shape (2M, 2M).
    """
    Sigma = loadmat(measurement_path)["Sigma"]
    if finite_snapshot and finite_snapshot > 0:
        perturb = np.sqrt(finite_snapshot) * (
            np.random.randn(*Sigma.shape) + 1j * np.random.randn(*Sigma.shape)
        ) / np.sqrt(2)
        if diagonal:
            perturb = np.diag(np.diag(perturb))
        Sigma = Sigma + perturb
        Sigma = (Sigma + Sigma.conj().T) / 2
    return covariance_to_real(Sigma)


def load_channel_matrix(channel_path):
    """Load a batch-first complex channel matrix.

    :param channel_path: Path to a channel .mat file.
    :return: Complex array with shape (batch, 256).
    """
    data = loadmat(channel_path)
    if "h" not in data:
        raise KeyError(f"{channel_path} must contain variable 'h'.")
    return data["h"]


def _split_name(split):
    """Normalize dataset split names.

    :param split: Split name, accepting training, val, validation or test.
    :return: Canonical split name.
    """
    if split in {"validation", "val"}:
        return "val"
    if split not in {"training", "test"}:
        raise ValueError("split must be one of: training, val, validation, test")
    return split


def channel_path_o1(split):
    """Return the DeepMIMO O1 channel path.

    :param split: Dataset split.
    :return: Path to the O1 channel .mat file.
    """
    split = _split_name(split)
    return data_root() / "deepmimo" / "scenario_o1" / "channels" / f"deepmimo_o1_28_{split}_channels.mat"


def measurement_path_o1(A_type="unitary", split="test", snr=20):
    """Return the DeepMIMO O1 measurement path.

    :param A_type: Matrix type in gaussian, bernoulli or unitary.
    :param split: Dataset split.
    :param snr: Fixed SNR in dB.
    :return: Path to the O1 measurement .mat file.
    """
    split = _split_name(split)
    if snr not in SNR_LIST:
        raise ValueError(f"snr must be one of {SNR_LIST}")
    return data_root() / "deepmimo" / "scenario_o1" / "measurements" / A_type / f"deepmimo_o1_28_{split}_128_measurements_{snr}dB.mat"


def channel_path_o2(scene):
    """Return a DeepMIMO O2 scene channel path.

    :param scene: Lowercase O2 scene identifier.
    :return: Path to the O2 channel .mat file.
    """
    return data_root() / "deepmimo" / "scenario_o2" / "channels" / f"deepmimo_o2_3p5_{scene}.mat"


def measurement_path_o2(scene, snr=20):
    """Return a DeepMIMO O2 scene measurement path.

    :param scene: Lowercase O2 scene identifier.
    :param snr: Fixed SNR in dB.
    :return: Path to the O2 measurement .mat file.
    """
    if snr not in SNR_LIST:
        raise ValueError(f"snr must be one of {SNR_LIST}")
    return data_root() / "deepmimo" / "scenario_o2" / "measurements" / "unitary" / f"deepmimo_o2_3p5_{scene}_128_measurements_{snr}dB.mat"


def synthetic_channel_path(split):
    """Return the synthetic k-sparse channel path.

    :param split: Dataset split.
    :return: Path to the synthetic channel .mat file.
    """
    split = _split_name(split)
    return data_root() / "synthetic" / "channels" / f"synthetic_{split}_channels.mat"


def synthetic_measurement_path(split, snr=20):
    """Return the synthetic k-sparse measurement path.

    :param split: Dataset split.
    :param snr: Fixed SNR in dB.
    :return: Path to the synthetic measurement .mat file.
    """
    split = _split_name(split)
    if snr not in SNR_LIST:
        raise ValueError(f"snr must be one of {SNR_LIST}")
    return data_root() / "synthetic" / "measurements" / "unitary" / f"synthetic_{split}_128_measurements_{snr}dB.mat"


def load_deq_arrays(measurement_path, channel_path, A, Sigma):
    """Load batch-first DEQ inputs and labels.

    :param measurement_path: Path to a measurement .mat file.
    :param channel_path: Path to a channel .mat file.
    :param A: Real sensing matrix with shape (2M, 2N).
    :param Sigma: Real covariance basis with shape (2M, 2M).
    :return: Tuple (channels, sufficient_statistics, sigma_squared), all batch-first.
    """
    channel = complex_vectors_to_real(load_channel_matrix(channel_path))
    y = complex_vectors_to_real(loadmat(measurement_path)["y"])
    inv_Sigma = torch.linalg.inv(Sigma)
    measurement = y @ inv_Sigma.T @ A
    sigma_squared = torch.from_numpy(loadmat(measurement_path)["sigma_squared"] / 2).float()
    return channel, measurement, sigma_squared


def load_raw_arrays(measurement_path, channel_path, sigma_squared_scale=1.0):
    """Load batch-first raw y and h arrays.

    :param measurement_path: Path to a measurement .mat file.
    :param channel_path: Path to a channel .mat file.
    :param sigma_squared_scale: Multiplicative scale for sigma_squared.
    :return: Tuple (channels, measurements, sigma_squared), all batch-first.
    """
    channel = complex_vectors_to_real(load_channel_matrix(channel_path))
    measurement = complex_vectors_to_real(loadmat(measurement_path)["y"])
    sigma_squared = torch.from_numpy(loadmat(measurement_path)["sigma_squared"] * sigma_squared_scale).float()
    return channel, measurement, sigma_squared


class ChannelDataset(Dataset):
    """Batch-first dataset for channel estimation samples.

    :param channels: Tensor with shape (batch, dim_h).
    :param measurements: Tensor with shape (batch, dim_y).
    :param sigma_squared: Tensor with shape (batch, 1).
    :return: Dataset returning one row per sample.
    """

    def __init__(self, channels, measurements, sigma_squared):
        """Store batch-first channel, measurement and noise tensors.

        :param channels: Tensor with shape (batch, dim_h).
        :param measurements: Tensor with shape (batch, dim_y).
        :param sigma_squared: Tensor with shape (batch, 1).
        :return: None.
        """
        self.channels = channels
        self.measurements = measurements
        self.sigma_squared = sigma_squared

    def __len__(self):
        """Return the number of samples.

        :return: Dataset length.
        """
        return self.channels.shape[0]

    def __getitem__(self, index):
        """Return one batch-first sample.

        :param index: Sample index.
        :return: Tuple (measurement, channel, sigma_squared).
        """
        return self.measurements[index], self.channels[index], self.sigma_squared[index]


def make_loaders(train_data, val_data, batch_size=128, split_val_data=False):
    """Create train and validation data loaders.

    :param train_data: Tuple of batch-first training tensors.
    :param val_data: Tuple of batch-first validation tensors.
    :param batch_size: Training mini-batch size.
    :param split_val_data: Whether to split validation into mini-batches.
    :return: Tuple (train_loader, val_loader).
    """
    train_channels, train_measurements, train_sigma_squared = train_data
    val_channels, val_measurements, val_sigma_squared = val_data
    train_dataset = ChannelDataset(train_channels, train_measurements, train_sigma_squared)
    val_dataset = ChannelDataset(val_channels, val_measurements, val_sigma_squared)
    val_batch_size = batch_size if split_val_data else val_dataset.channels.shape[0]
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=val_batch_size, drop_last=True, num_workers=0, pin_memory=True)
    return train_loader, val_loader


def prepare_deq_training_data(args, snr, batch_size=128):
    """Prepare batch-first DEQ or DNN training data.

    :param args: Parsed command-line arguments.
    :param snr: Fixed SNR in dB.
    :param batch_size: Training mini-batch size.
    :return: Tuple (A, Sigma, (train_loader, validation_loader)).
    """
    A, _ = load_cs_matrix(args.A_type)
    if args.data == "deepmimo":
        train_measurement = measurement_path_o1(args.A_type, split="training", snr=snr)
        val_measurement = measurement_path_o1(args.A_type, split="val", snr=snr)
        train_channel = channel_path_o1("training")
        val_channel = channel_path_o1("val")
    elif args.data == "synthetic":
        if args.A_type != "unitary":
            raise ValueError("Synthetic measurements are available only for A_type=unitary in this release.")
        train_measurement = synthetic_measurement_path("training", snr=snr)
        val_measurement = synthetic_measurement_path("val", snr=snr)
        train_channel = synthetic_channel_path("training")
        val_channel = synthetic_channel_path("val")
    else:
        raise ValueError("data must be either deepmimo or synthetic.")
    Sigma = load_covariance(train_measurement, finite_snapshot=args.finite_snapshot, diagonal=args.diagonal)
    train_data = load_deq_arrays(train_measurement, train_channel, A, Sigma)
    val_data = load_deq_arrays(val_measurement, val_channel, A, Sigma)
    return A, Sigma, make_loaders(train_data, val_data, batch_size=batch_size, split_val_data=False)


def prepare_raw_o1_loaders(A_type="unitary", snr=20, batch_size=128):
    """Prepare raw DeepMIMO O1 loaders for LDGEC.

    :param A_type: Matrix type in gaussian, bernoulli or unitary.
    :param snr: Fixed SNR in dB.
    :param batch_size: Training mini-batch size.
    :return: Tuple (train_loader, validation_loader).
    """
    train_data = load_raw_arrays(measurement_path_o1(A_type, split="training", snr=snr), channel_path_o1("training"))
    val_data = load_raw_arrays(measurement_path_o1(A_type, split="val", snr=snr), channel_path_o1("val"))
    return make_loaders(train_data, val_data, batch_size=batch_size, split_val_data=False)


def load_deq_test_data(A_type="unitary", snr=20, split="test", Sigma=None):
    """Load batch-first DEQ test data.

    :param A_type: Matrix type in gaussian, bernoulli or unitary.
    :param snr: Fixed SNR in dB.
    :param split: Dataset split.
    :param Sigma: Optional real covariance basis.
    :return: Tuple (channels, measurements, sigma_squared, Sigma), all data tensors batch-first.
    """
    A, _ = load_cs_matrix(A_type)
    measurement_path = measurement_path_o1(A_type, split=split, snr=snr)
    if Sigma is None:
        Sigma = load_covariance(measurement_path)
    channels, measurements, sigma_squared = load_deq_arrays(measurement_path, channel_path_o1(split), A, Sigma)
    return channels, measurements, sigma_squared, Sigma


def load_raw_o1_data(A_type="unitary", snr=20, split="test"):
    """Load batch-first complex DeepMIMO O1 raw data.

    :param A_type: Matrix type in gaussian, bernoulli or unitary.
    :param snr: Fixed SNR in dB.
    :param split: Dataset split.
    :return: Tuple (channels, measurements, sigma_squared) as numpy arrays.
    """
    measurement_path = measurement_path_o1(A_type, split=split, snr=snr)
    channel_path = channel_path_o1(split)
    channels = load_channel_matrix(channel_path)
    measurements = loadmat(measurement_path)["y"]
    sigma_squared = loadmat(measurement_path)["sigma_squared"]
    return channels, measurements, sigma_squared


def load_raw_synthetic_data(snr=20, split="test"):
    """Load batch-first complex synthetic raw data.

    :param snr: Fixed SNR in dB.
    :param split: Dataset split.
    :return: Tuple (channels, measurements, sigma_squared) as numpy arrays.
    """
    measurement_path = synthetic_measurement_path(split, snr=snr)
    channel_path = synthetic_channel_path(split)
    channels = load_channel_matrix(channel_path)
    measurements = loadmat(measurement_path)["y"]
    sigma_squared = loadmat(measurement_path)["sigma_squared"]
    return channels, measurements, sigma_squared


def load_raw_o2_data(snr=20):
    """Load batch-first complex DeepMIMO O2 raw data.

    :param snr: Fixed SNR in dB.
    :return: Tuple (channels, measurements, sigma_squared) concatenated over scenes.
    """
    channels, measurements, sigma_squared_values = [], [], []
    for scene in DEEP_MIMO_O2_SCENES:
        ch_path = channel_path_o2(scene)
        y_path = measurement_path_o2(scene, snr=snr)
        if not ch_path.exists() or not y_path.exists():
            continue
        channels.append(load_channel_matrix(ch_path))
        measurements.append(loadmat(y_path)["y"])
        sigma_squared_values.append(loadmat(y_path)["sigma_squared"])
    if not channels:
        raise FileNotFoundError("No DeepMIMO O2 pedestrian data was found.")
    return np.concatenate(channels, axis=0), np.concatenate(measurements, axis=0), np.concatenate(sigma_squared_values, axis=0)


def raw_to_deq_sufficient_statistic(y_complex, h_complex, sigma_squared, A_complex, Sigma_real=None):
    """Convert raw batch-first complex data to DEQ sufficient statistics.

    :param y_complex: Complex measurements with shape (batch, M).
    :param h_complex: Complex channels with shape (batch, N).
    :param sigma_squared: Noise variance with shape (batch, 1).
    :param A_complex: Complex sensing matrix with shape (M, N).
    :param Sigma_real: Optional real covariance basis with shape (2M, 2M).
    :return: Tuple (h, u, sigma_squared, A, Sigma), with h/u/sigma_squared batch-first.
    """
    y = complex_vectors_to_real(y_complex)
    h = complex_vectors_to_real(h_complex)
    A = complex_matrix_to_real(A_complex)
    if Sigma_real is None:
        Sigma_real = torch.eye(A.shape[0])
    u = y @ torch.linalg.inv(Sigma_real).T @ A
    sigma_squared_real = torch.from_numpy(sigma_squared / 2).float()
    return h, u, sigma_squared_real, A, Sigma_real
