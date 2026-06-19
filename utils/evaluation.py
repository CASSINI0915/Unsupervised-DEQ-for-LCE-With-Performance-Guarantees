#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

import utils.utils as legacy_utils
from model import deq as deq_module
from model import dnn as dnn_module
from model import ldgec as ldgec_module
from model.deq import DEQ
from model.dnn import DNN
from utils.checkpoint import find_checkpoint, infer_training_data_from_checkpoint, load_state
from utils.data_utils import (
    A_TYPES,
    channel_path_o1,
    complex_vectors_to_real,
    load_covariance,
    load_complex_A,
    load_cs_matrix,
    load_deq_arrays,
    load_raw_o1_data,
    load_raw_o2_data,
    measurement_path_o1,
    raw_to_deq_sufficient_statistic,
    synthetic_channel_path,
    synthetic_measurement_path,
)
from utils.metrics import nmse_db_torch, timed_call


VALID_LOSSES = {
    "deq": {"gsure", "nmse"},
    "dnn": {"gsure", "nmse"},
    "ldgec": {"sure", "nmse"},
}

TEST_DATASETS = {"deepmimo_o1", "deepmimo_o2", "synthetic"}


def set_all_devices(device):
    """Set the torch device for copied model and utility modules."""
    legacy_utils.set_device(device)
    deq_module.set_device(device)
    dnn_module.set_device(device)
    ldgec_module.set_device(device)


def normalize_test_data(data):
    """Normalize public test dataset names and legacy aliases."""
    if data == "deepmimo":
        return "deepmimo_o1"
    return data


def checkpoint_data_for_test_data(data):
    """Return the checkpoint dataset tag used for a requested test dataset."""
    data = normalize_test_data(data)
    if data == "synthetic":
        return "synthetic"
    return "deepmimo"


def validate_eval_config(model, loss, data="deepmimo", A_type="unitary", enforce_release_matrix=True):
    """Validate a public evaluation configuration."""
    data = normalize_test_data(data)
    if model not in VALID_LOSSES:
        raise ValueError(f"--model must be one of {{{', '.join(sorted(VALID_LOSSES))}}}; got {model}.")
    if loss not in VALID_LOSSES[model]:
        allowed = ", ".join(sorted(VALID_LOSSES[model]))
        raise ValueError(f"--model {model} supports --loss in {{{allowed}}}; got {loss}.")
    if data not in TEST_DATASETS:
        raise ValueError("--data must be one of: deepmimo_o1, deepmimo_o2, synthetic.")
    if A_type not in A_TYPES:
        raise ValueError(f"--A_type must be one of {{{', '.join(A_TYPES)}}}; got {A_type}.")
    if data == "deepmimo_o2" and A_type != "unitary":
        raise ValueError("DeepMIMO O2 measurements are available only for A_type=unitary.")
    if data == "synthetic" and A_type != "unitary":
        raise ValueError("Synthetic measurements are available only for A_type=unitary.")
    if data == "synthetic" and model != "deq":
        raise ValueError("Synthetic evaluation in this release supports only model=deq.")
    if enforce_release_matrix and A_type in {"gaussian", "bernoulli"} and model != "deq":
        raise ValueError("For A_type=gaussian or bernoulli, this test entry supports only model=deq.")


def result_label(model, loss):
    """Return the console label used for one model/loss pair."""
    return f"{model.upper()}-{loss.upper()}"


def load_deq_eval_data(data="deepmimo_o1", A_type="unitary", snr=20, split="test", Sigma=None):
    """Load batch-first sufficient-statistic data for DEQ/DNN evaluation."""
    data = normalize_test_data(data)
    validate_eval_config("deq", "gsure", data=data, A_type=A_type, enforce_release_matrix=False)
    A, _ = load_cs_matrix(A_type)
    if data == "deepmimo_o1":
        measurement_path = measurement_path_o1(A_type, split=split, snr=snr)
        channel_path = channel_path_o1(split)
        if Sigma is None:
            Sigma = load_covariance(measurement_path)
        channels, measurements, sigma_squared = load_deq_arrays(measurement_path, channel_path, A, Sigma)
        return channels, measurements, sigma_squared, Sigma
    if data == "synthetic":
        measurement_path = synthetic_measurement_path(split, snr=snr)
        channel_path = synthetic_channel_path(split)
        if Sigma is None:
            Sigma = load_covariance(measurement_path)
        channels, measurements, sigma_squared = load_deq_arrays(measurement_path, channel_path, A, Sigma)
        return channels, measurements, sigma_squared, Sigma

    channels_c, y_c, sigma_squared = load_raw_o2_data(snr)
    h, u, sigma_squared, _, Sigma = raw_to_deq_sufficient_statistic(
        y_c,
        channels_c,
        sigma_squared,
        load_complex_A(A_type),
        Sigma_real=Sigma,
    )
    return h, u, sigma_squared, Sigma


def build_deq(A_type, snr, device, data="deepmimo_o1", max_depth=50):
    """Build a DEQ model for one test configuration."""
    A, _ = load_cs_matrix(A_type)
    _, _, _, Sigma = load_deq_eval_data(data=data, A_type=A_type, snr=snr, split="test")
    net = DEQ(
        A=A,
        Sigma=Sigma,
        lat_layers=4,
        contraction_factor=0.99,
        eps=1e-2,
        max_depth=max_depth,
        num_channels=32,
        bias=False,
    ).to(device)
    return net, Sigma


def evaluate_deq(loss, snr, device, A_type="unitary", data="deepmimo_o1", record_layerwise=False, finite_snapshot=0.0, diagonal=False):
    """Evaluate one DEQ checkpoint on batch-first test data."""
    data = normalize_test_data(data)
    validate_eval_config("deq", loss, data=data, A_type=A_type, enforce_release_matrix=False)
    net, _ = build_deq(A_type, snr, device, data=data)
    ckpt_data = checkpoint_data_for_test_data(data)
    ckpt = find_checkpoint("deq", loss, snr, A_type=A_type, data=ckpt_data, finite_snapshot=finite_snapshot, diagonal=diagonal)
    if ckpt is None:
        print(f"DEQ/{loss} checkpoint missing for train data={ckpt_data}, test data={data}, A={A_type}, SNR={snr} dB.")
        return None
    load_state(net, ckpt, device)
    net.eval()
    channels, measurements, _, _ = load_deq_eval_data(
        data=data,
        A_type=A_type,
        snr=snr,
        split="test",
        Sigma=net.Sigma.detach().cpu(),
    )
    channels = channels.to(device)
    measurements = measurements.to(device)
    if record_layerwise:
        net.record_layerwise = True
        net.layerwise_h = []
    with torch.no_grad():
        pred, elapsed = timed_call(net, measurements, cuda_device=device)
    nmse = float(nmse_db_torch(pred, channels).item())
    lip = None
    try:
        lip = float(net.l1andl2_estimate(measurements, pred, sigma=1.0).item())
    except Exception:
        pass
    return {
        "label": result_label("deq", loss),
        "model": "deq",
        "loss": loss,
        "train_data": infer_training_data_from_checkpoint(ckpt),
        "test_data": data,
        "data": data,
        "A_type": A_type,
        "snr": snr,
        "nmse": nmse,
        "runtime": elapsed,
        "checkpoint": ckpt,
        "lip": lip,
        "net": net,
        "channels": channels,
        "pred": pred,
    }


def evaluate_dnn(loss, snr, device, A_type="unitary", data="deepmimo_o1"):
    """Evaluate one DNN checkpoint on batch-first test data."""
    data = normalize_test_data(data)
    validate_eval_config("dnn", loss, data=data, A_type=A_type, enforce_release_matrix=False)
    _, _, _, Sigma = load_deq_eval_data(data=data, A_type=A_type, snr=snr, split="test")
    net = DNN(dim_in=256, dim_out=256).to(device)
    ckpt_data = checkpoint_data_for_test_data(data)
    ckpt = find_checkpoint("dnn", loss, snr, A_type=A_type, data=ckpt_data)
    if ckpt is None:
        print(f"DNN/{loss} checkpoint missing for train data={ckpt_data}, test data={data}, A={A_type}, SNR={snr} dB.")
        return None
    load_state(net, ckpt, device)
    net.eval()
    channels, measurements, _, _ = load_deq_eval_data(data=data, A_type=A_type, snr=snr, split="test", Sigma=Sigma)
    channels = channels.to(device)
    measurements = measurements.to(device)
    with torch.no_grad():
        pred, elapsed = timed_call(net, measurements, cuda_device=device)
    return {
        "label": result_label("dnn", loss),
        "model": "dnn",
        "loss": loss,
        "train_data": infer_training_data_from_checkpoint(ckpt),
        "test_data": data,
        "data": data,
        "A_type": A_type,
        "snr": snr,
        "nmse": float(nmse_db_torch(pred, channels).item()),
        "runtime": elapsed,
        "checkpoint": ckpt,
    }


def evaluate_ldgec(loss, snr, device, A_type="unitary", data="deepmimo_o1"):
    """Evaluate one LDGEC checkpoint on raw DeepMIMO test data."""
    data = normalize_test_data(data)
    validate_eval_config("ldgec", loss, data=data, A_type=A_type, enforce_release_matrix=False)
    A, _ = load_cs_matrix(A_type)
    net = ldgec_module.LDGEC(num_layers=8, A=A).to(device)
    ckpt_data = checkpoint_data_for_test_data(data)
    ckpt = find_checkpoint("ldgec", loss, snr, A_type=A_type, data=ckpt_data)
    if ckpt is None:
        print(f"LDGEC/{loss} checkpoint missing for train data={ckpt_data}, test data={data}, A={A_type}, SNR={snr} dB.")
        return None
    load_state(net, ckpt, device)
    net.eval()
    if data == "deepmimo_o2":
        channels_c, y_c, sigma_squared_c = load_raw_o2_data(snr)
    else:
        channels_c, y_c, sigma_squared_c = load_raw_o1_data(A_type, snr=snr, split="test")
    channels = complex_vectors_to_real(channels_c).to(device)
    y = complex_vectors_to_real(y_c).to(device)
    sigma_squared = torch.from_numpy(sigma_squared_c).float().to(device)
    with torch.no_grad():
        pred, elapsed = timed_call(net, y, sigma_squared, cuda_device=device)
    return {
        "label": result_label("ldgec", loss),
        "model": "ldgec",
        "loss": loss,
        "train_data": infer_training_data_from_checkpoint(ckpt),
        "test_data": data,
        "data": data,
        "A_type": A_type,
        "snr": snr,
        "nmse": float(nmse_db_torch(pred, channels).item()),
        "runtime": elapsed,
        "checkpoint": ckpt,
    }


def evaluate_model(model, loss, snr, device, A_type="unitary", data="deepmimo_o1", finite_snapshot=0.0, diagonal=False):
    """Evaluate one public model/loss/data/A/SNR configuration."""
    data = normalize_test_data(data)
    validate_eval_config(model, loss, data=data, A_type=A_type, enforce_release_matrix=True)
    if model == "deq":
        return evaluate_deq(
            loss,
            snr,
            device,
            A_type=A_type,
            data=data,
            finite_snapshot=finite_snapshot,
            diagonal=diagonal,
        )
    if model == "dnn":
        return evaluate_dnn(loss, snr, device, A_type=A_type, data=data)
    if model == "ldgec":
        return evaluate_ldgec(loss, snr, device, A_type=A_type, data=data)
    raise ValueError(f"Unsupported model: {model}")
