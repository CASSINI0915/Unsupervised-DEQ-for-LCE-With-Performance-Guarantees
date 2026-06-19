#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import torch


def root_dir():
    """Return the project root directory.
    
    :return: Path to the tsp_deq_gsure root directory.
    """
    return Path(__file__).resolve().parents[1]


def _format_float(value):
    """Format numeric checkpoint tags without unnecessary trailing zeros."""
    return f"{float(value):g}"


def checkpoint_name(model, loss, snr, timestamp=None, data="deepmimo", biased_R=False, finite_snapshot=0.0, diagonal=False):
    """Build the checkpoint filename for one training configuration.
    
    :param model: Model name in deq, dnn or ldgec.
    :param loss: Training loss name supported by the selected model.
    :param snr: Fixed SNR in dB.
    :param timestamp: Optional timestamp suffix.
    :param data: Dataset name. deepmimo keeps legacy names; synthetic adds the syn tag.
    :param biased_R: Whether the DEQ estimator uses bias terms.
    :param finite_snapshot: Variance of the noisy covariance perturbation.
    :param diagonal: Whether the covariance perturbation is diagonal-only.
    :return: Checkpoint filename.
    """
    model = model.lower()
    loss = loss.lower()
    data = data.lower()
    valid_losses = {
        "deq": {"gsure", "nmse"},
        "dnn": {"gsure", "nmse"},
        "ldgec": {"sure", "nmse"},
    }
    if model not in valid_losses or loss not in valid_losses[model]:
        raise ValueError(f"Unsupported checkpoint kind: {model}/{loss}")
    if data not in {"deepmimo", "synthetic"}:
        raise ValueError(f"Unsupported checkpoint data: {data}")
    if data == "synthetic" and model != "deq":
        raise ValueError("Synthetic checkpoints are supported only for DEQ.")

    tags = []
    if biased_R:
        tags.append("biased_R")
    if finite_snapshot and finite_snapshot > 0:
        covariance_tag = "noisy_diagC" if diagonal else "noisy_C"
        tags.append(f"{covariance_tag}{_format_float(finite_snapshot)}")

    data_tags = ["syn"] if data == "synthetic" else []
    parts = [model] + data_tags + [loss] + tags + [f"{snr}dB"]
    base = "_".join(parts)
    if timestamp:
        base += f"_{timestamp}"
    return base + ".pth"


def checkpoint_dir(A_type="unitary"):
    """Return the checkpoint directory for one sensing matrix type.
    
    :param A_type: Matrix type in gaussian, bernoulli or unitary.
    :return: Path to the checkpoint directory.
    """
    return root_dir() / "trained network" / A_type


def infer_training_data_from_checkpoint(path):
    """Infer the training dataset from a checkpoint filename."""
    if path is None:
        return "unknown"
    parts = Path(path).stem.split("_")
    if len(parts) >= 2 and parts[1] == "syn":
        return "synthetic"
    return "deepmimo_o1"


def candidate_paths(model, loss, snr, A_type="unitary", data="deepmimo", biased_R=False, finite_snapshot=0.0, diagonal=False):
    """Return candidate checkpoint paths for one configuration.
    
    :param model: Model name in deq, dnn or ldgec.
    :param loss: Training loss name supported by the selected model.
    :param snr: Fixed SNR in dB.
    :param A_type: Matrix type in gaussian, bernoulli or unitary.
    :param data: Dataset name. deepmimo keeps legacy names; synthetic adds the syn tag.
    :param biased_R: Whether the DEQ estimator uses bias terms.
    :param finite_snapshot: Variance of the noisy covariance perturbation.
    :param diagonal: Whether the covariance perturbation is diagonal-only.
    :return: List of checkpoint paths to try.
    """
    data = data.lower()
    if data == "synthetic" and A_type != "unitary":
        raise ValueError("Synthetic checkpoints are available only for A_type='unitary'.")
    root = checkpoint_dir(A_type)
    name = checkpoint_name(
        model,
        loss,
        snr,
        data=data,
        biased_R=biased_R,
        finite_snapshot=finite_snapshot,
        diagonal=diagonal,
    )
    return [root / name]


def find_checkpoint(model, loss, snr, A_type="unitary", data="deepmimo", required=False, biased_R=False, finite_snapshot=0.0, diagonal=False):
    """Find an existing checkpoint for one configuration.
    
    :param model: Model name in deq, dnn or ldgec.
    :param loss: Training loss name supported by the selected model.
    :param snr: Fixed SNR in dB.
    :param A_type: Matrix type in gaussian, bernoulli or unitary.
    :param data: Dataset name. deepmimo keeps legacy names; synthetic adds the syn tag.
    :param required: Whether to raise FileNotFoundError when no checkpoint exists.
    :param biased_R: Whether the DEQ estimator uses bias terms.
    :param finite_snapshot: Variance of the noisy covariance perturbation.
    :param diagonal: Whether the covariance perturbation is diagonal-only.
    :return: Existing checkpoint path, or None when not required.
    """
    data = data.lower()
    for path in candidate_paths(model, loss, snr, A_type, data=data, biased_R=biased_R, finite_snapshot=finite_snapshot, diagonal=diagonal):
        if path.exists():
            return path
    if required:
        tried = "\n".join(
            str(p)
            for p in candidate_paths(model, loss, snr, A_type, data=data, biased_R=biased_R, finite_snapshot=finite_snapshot, diagonal=diagonal)
        )
        raise FileNotFoundError(f"No checkpoint found. Tried:\n{tried}")
    return None


def load_state(model, path, device):
    """Load a checkpoint state dict into a model.
    
    :param model: Torch module receiving checkpoint weights.
    :param path: Checkpoint path.
    :param device: Device used by torch.load map_location.
    :return: Model with loaded weights.
    """
    state = torch.load(path, map_location=device)
    model_state = model.state_dict()
    remapped_state = {}
    for key, value in state.items():
        remapped_key = key
        if ".denoiser." not in key:
            parts = key.split(".")
            if len(parts) >= 4 and parts[0] == "Layers" and parts[2] in {"head_convs", "body_convs", "tail_convs"}:
                remapped_key = ".".join(parts[:2] + ["denoiser"] + parts[2:])
        remapped_state[remapped_key] = value
    state = remapped_state
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        compatible = {
            key: value
            for key, value in state.items()
            if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
        }
        model_state.update(compatible)
        model.load_state_dict(model_state, strict=False)
        skipped = sorted(set(state) - set(compatible))
        print(f"Warning: loaded checkpoint with compatible-key fallback from {path}.")
        print(f"Loaded {len(compatible)} tensors; skipped {len(skipped)} incompatible/unexpected tensors.")
        if not compatible:
            raise exc
    return model

