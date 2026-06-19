#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from model.classic_algorithms import ClassicAlgorithms
from utils.checkpoint import find_checkpoint
from utils.data_utils import SNR_LIST
from utils.evaluation import evaluate_deq, evaluate_dnn, evaluate_ldgec, set_all_devices


__doc__ = """
Plotting utilities for the released TSP DEQ-GSURE experiments.

This script exposes six mutually exclusive command-line plotting tasks.
All experiment settings that should not be user-facing, such as GPU index,
sensing-matrix type, finite-snapshot variance, and diagonal perturbation mode,
are fixed internally by the corresponding plotting routine.

Command-line arguments
----------------------

--scenario_o1
    Plot NMSE-versus-SNR curves on the DeepMIMO O1 test set.
    The figure compares OMP, AMP, OAMP, DNN-GSURE, DNN-NMSE,
    LDGEC-SURE, LDGEC-NMSE, DEQ-GSURE, and DEQ-NMSE in one axis.

--scenario_o2
    Plot NMSE-versus-SNR generalization curves involving the DeepMIMO O2
    test set. The figure includes OMP, AMP, OAMP, LDGEC-SURE,
    DEQ-GSURE, and DNN-NMSE tested on O2, together with the selected
    O1 reference curves for LDGEC-SURE, DEQ-GSURE, and DNN-NMSE.

--nFEs
    Plot layerwise DEQ-GSURE diagnostics over fixed-point iterations.
    The output is a two-panel figure: residual versus iteration on the
    left, and NMSE versus iteration on the right, for SNR values
    0, 5, 10, 15, and 20 dB.

--beta_omega
    Plot the beta and omega diagnostic figure from the supplied figure data.
    The left panel shows optimal beta versus SNR for DEQ-GSURE and DEQ-NMSE.
    The right panel shows NMSE versus SNR with a secondary omega axis.

--nmse_vs_A_type
    Plot DEQ NMSE-versus-SNR curves under different sensing matrix types.
    The figure compares GSURE and NMSE training losses for unitary,
    Gaussian, and Bernoulli sensing matrices.

--nmse_vs_sigma_e
    Plot NMSE versus covariance perturbation strength sigma_e^2 at 10 dB.
    The figure compares the infinite-snapshot reference, full covariance
    perturbation, and diagonal covariance perturbation.

Checkpoint handling
-------------------

For model-based NMSE-versus-SNR curves, a curve is plotted only when the
required checkpoints exist for all five SNR values in SNR_LIST. If any SNR
checkpoint is missing for a model curve, that whole curve is skipped while
the remaining valid curves are still plotted.

Classic OMP, AMP, and OAMP baselines are loaded from cache when available
and recomputed otherwise.

Outputs
-------

Figures are saved under figures/ and numeric results are saved under results/.
If those directories are not writable in the current environment, the script
falls back to generated_figures/, generated_results/, or generated_model/.
"""


FIGURE_DIR = Path("figures")
RESULT_DIR = Path("results")
FALLBACK_FIGURE_DIR = Path("generated_figures")
FALLBACK_RESULT_DIR = Path("generated_results")
FALLBACK_MODEL_DIR = Path("generated_model")
UNITARY = "unitary"
MARKER_SIZE = 6


def rgb(red, green, blue):
    """Return a matplotlib RGB tuple from 0-255 channel values."""
    return (red / 255.0, green / 255.0, blue / 255.0)


RED = rgb(252, 41, 30)
BLUE = rgb(0, 70, 222)
DARK_TEAL = rgb(5, 80, 91)
PURPLE = (0.49, 0.18, 0.56)
GREEN = rgb(124, 187, 0)
CYAN = rgb(0, 161, 241)
GRAY_TEAL = rgb(84, 134, 135)
YELLOW = rgb(255, 187, 0)
TEAL = rgb(48, 151, 164)
DARK_GRAY = (0.35, 0.35, 0.35)


def build_parser():
    """Build the command-line parser for the fixed plotting entry points."""
    parser = argparse.ArgumentParser(description="Plot released TSP DEQ-GSURE figures.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenario_o1", action="store_true")
    group.add_argument("--scenario_o2", action="store_true")
    group.add_argument("--nFEs", action="store_true")
    group.add_argument("--beta_omega", action="store_true")
    group.add_argument("--nmse_vs_A_type", action="store_true")
    group.add_argument("--nmse_vs_sigma_e", action="store_true")
    return parser


def checkpoint_data_tag(data):
    """Return the training-data checkpoint tag used by one test dataset."""
    return "synthetic" if data == "synthetic" else "deepmimo"


def torch_load(path):
    """Load torch files across PyTorch versions with explicit CPU mapping."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_figure(fig, output_path):
    """Save and close one matplotlib figure."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    try:
        fig.savefig(output_path, bbox_inches="tight")
    except PermissionError:
        fallback = unique_output_path(fallback_output_path(output_path))
        fallback.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cannot write {output_path}; saving to {fallback} instead.")
        fig.savefig(fallback, bbox_inches="tight")
        output_path = fallback
    plt.close(fig)
    print(f"Saved figure: {output_path}")
    return output_path


def fallback_output_path(output_path):
    """Return a writable fallback path for common output directories."""
    output_path = Path(output_path)
    if output_path.parent == FIGURE_DIR:
        return FALLBACK_FIGURE_DIR / output_path.name
    if output_path.parent == RESULT_DIR:
        return FALLBACK_RESULT_DIR / output_path.name
    if output_path.parent == Path("model"):
        return FALLBACK_MODEL_DIR / output_path.name
    return output_path.with_name(f"{output_path.stem}_new{output_path.suffix}")


def unique_output_path(output_path):
    """Return a non-existing variant of a path."""
    output_path = Path(output_path)
    if not output_path.exists():
        return output_path
    index = 1
    while True:
        candidate = output_path.with_name(f"{output_path.stem}_new{index}{output_path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def save_torch(obj, output_path):
    """Save a torch object, falling back when the preferred directory is not writable."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(obj, output_path)
    except (PermissionError, RuntimeError) as exc:
        if isinstance(exc, RuntimeError) and "cannot be opened" not in str(exc):
            raise
        fallback = unique_output_path(fallback_output_path(output_path))
        fallback.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cannot write {output_path}; saving to {fallback} instead.")
        torch.save(obj, fallback)
        output_path = fallback
    print(f"Saved results: {output_path}")
    return output_path


def apply_snr_axes(ax, ylabel="NMSE (dB)", label_fontsize=16, tick_fontsize=16, legend_fontsize=12, legend_loc="best"):
    """Apply the shared SNR-vs-metric axis formatting."""
    ax.set_xlabel("SNR (dB)", fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.set_xticks(SNR_LIST)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(True)
    ax.legend(loc=legend_loc, fontsize=legend_fontsize)


def plot_curve(ax, snrs, values, style):
    """Plot one curve using a MATLAB-like style dictionary."""
    kwargs = {
        "color": style.get("color"),
        "marker": style.get("marker", "o"),
        "linestyle": style.get("linestyle", "-"),
        "linewidth": style.get("linewidth", 1.5),
        "markersize": style.get("markersize", MARKER_SIZE),
        "label": style["label"],
    }
    if "markerfacecolor" in style:
        kwargs["markerfacecolor"] = style["markerfacecolor"]
    ax.plot(snrs, values, **kwargs)


def model_checkpoint_complete(spec, snrs=SNR_LIST):
    """Return True only if a model curve has checkpoints for all requested SNRs."""
    missing = []
    data = spec.get("data", "deepmimo_o1")
    for snr in snrs:
        checkpoint = find_checkpoint(
            spec["model"],
            spec["loss"],
            snr,
            A_type=spec.get("A_type", UNITARY),
            data=checkpoint_data_tag(data),
            finite_snapshot=spec.get("finite_snapshot", 0.0),
            diagonal=spec.get("diagonal", False),
            biased_R=spec.get("biased_R", False),
        )
        if checkpoint is None:
            missing.append(snr)

    if missing:
        snr_text = ", ".join(f"{snr} dB" for snr in missing)
        print(f"Skip {spec['label']}: missing checkpoint(s) at {snr_text}.")
        return False
    return True


def evaluate_model_curve(spec, device, snrs=SNR_LIST):
    """Evaluate one complete model curve."""
    if not model_checkpoint_complete(spec, snrs):
        return None

    values = []
    for snr in snrs:
        model = spec["model"]
        loss = spec["loss"]
        data = spec.get("data", "deepmimo_o1")
        A_type = spec.get("A_type", UNITARY)
        if model == "deq":
            out = evaluate_deq(
                loss,
                snr,
                device,
                A_type=A_type,
                data=data,
                finite_snapshot=spec.get("finite_snapshot", 0.0),
                diagonal=spec.get("diagonal", False),
            )
        elif model == "dnn":
            out = evaluate_dnn(loss, snr, device, A_type=A_type, data=data)
        elif model == "ldgec":
            out = evaluate_ldgec(loss, snr, device, A_type=A_type, data=data)
        else:
            raise ValueError(f"Unsupported model: {model}")

        if out is None:
            print(f"Skip {spec['label']}: evaluation returned no result at {snr} dB.")
            return None
        values.append(out["nmse"])
    return values


def classic_cache_candidates(data):
    """Return cache paths for classic algorithm results."""
    filename = f"classic_algorithms_results_{data}.pth"
    candidates = [Path("model") / filename, FALLBACK_MODEL_DIR / filename]
    if data == "deepmimo_o1":
        candidates.append(Path("model") / "classic_algorithms_results.pth")
        candidates.append(FALLBACK_MODEL_DIR / "classic_algorithms_results.pth")
    return candidates


def load_or_run_classic(data):
    """Load cached OMP/AMP/OAMP results or compute them if the cache is absent."""
    for cache_path in classic_cache_candidates(data):
        if cache_path.exists():
            state = torch_load(cache_path)
            nmse_db = state.get("nmse_db", {})
            if all(alg in nmse_db for alg in ("omp", "amp", "oamp")):
                print(f"Loaded classic baseline cache: {cache_path}")
                return state

    cache_path = classic_cache_candidates(data)[0]
    runner = ClassicAlgorithms(A_type=UNITARY, data=data)
    state = runner.evaluate(["omp", "amp", "oamp"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_torch(state, cache_path)
    print(f"Saved classic baseline cache: {cache_path}")
    return state


def add_classic_curves(ax, data, ordered_styles, results):
    """Add OMP/AMP/OAMP curves in the requested order."""
    state = load_or_run_classic(data)
    nmse_db = state["nmse_db"]
    for alg, style in ordered_styles:
        values = nmse_db.get(alg)
        if values is None:
            print(f"Skip {style['label']}: missing cached classic result for {alg}.")
            continue
        plot_curve(ax, SNR_LIST, values, style)
        results[style["label"]] = values


def run_scenario_o1(device):
    """Plot the DeepMIMO O1 NMSE-vs-SNR comparison."""
    results = {}
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    classic_styles = [
        ("omp", {"label": "OMP", "color": RED, "marker": "d", "markerfacecolor": "none"}),
        ("amp", {"label": "AMP", "color": BLUE, "marker": "*", "markerfacecolor": BLUE}),
        ("oamp", {"label": "OAMP", "color": DARK_TEAL, "marker": "h", "markerfacecolor": DARK_TEAL}),
    ]
    add_classic_curves(ax, "deepmimo_o1", classic_styles, results)

    model_specs = [
        (
            {"label": "DNN-GSURE", "model": "dnn", "loss": "gsure", "data": "deepmimo_o1"},
            {"label": "DNN-GSURE", "color": PURPLE, "marker": "^", "markerfacecolor": PURPLE},
        ),
        (
            {"label": "DNN-NMSE", "model": "dnn", "loss": "nmse", "data": "deepmimo_o1"},
            {"label": "DNN-NMSE", "color": GREEN, "marker": "s", "markerfacecolor": GREEN},
        ),
        (
            {"label": "LDGEC-SURE", "model": "ldgec", "loss": "sure", "data": "deepmimo_o1"},
            {"label": "LDGEC-SURE", "color": CYAN, "marker": "p", "markerfacecolor": CYAN, "linestyle": "-."},
        ),
        (
            {"label": "LDGEC-NMSE", "model": "ldgec", "loss": "nmse", "data": "deepmimo_o1"},
            {"label": "LDGEC-NMSE", "color": GRAY_TEAL, "marker": "p", "markerfacecolor": GRAY_TEAL},
        ),
        (
            {"label": "DEQ-GSURE", "model": "deq", "loss": "gsure", "data": "deepmimo_o1"},
            {"label": "DEQ-GSURE", "color": YELLOW, "marker": "o", "markerfacecolor": YELLOW},
        ),
        (
            {"label": "DEQ-NMSE", "model": "deq", "loss": "nmse", "data": "deepmimo_o1"},
            {"label": "DEQ-NMSE", "color": TEAL, "marker": "h", "markerfacecolor": TEAL},
        ),
    ]

    for spec, style in model_specs:
        values = evaluate_model_curve(spec, device)
        if values is None:
            continue
        plot_curve(ax, SNR_LIST, values, style)
        results[style["label"]] = values

    apply_snr_axes(ax, legend_fontsize=12)
    save_figure(fig, FIGURE_DIR / "scenario_o1_nmse_vs_snr.pdf")
    save_torch({"snr": list(SNR_LIST), "nmse_db": results}, RESULT_DIR / "scenario_o1_results.pth")


def run_scenario_o2(device):
    """Plot the DeepMIMO O2 generalization NMSE-vs-SNR comparison."""
    results = {}
    fig, ax = plt.subplots(figsize=(7.8, 5.4))

    classic_styles = [
        ("omp", {"label": "OMP on O2 scenario", "color": YELLOW, "marker": "o", "markerfacecolor": YELLOW}),
        ("amp", {"label": "AMP on O2 scenario", "color": TEAL, "marker": "h", "markerfacecolor": TEAL}),
        ("oamp", {"label": "OAMP on O2 scenario", "color": GRAY_TEAL, "marker": "^", "markerfacecolor": GRAY_TEAL}),
    ]
    add_classic_curves(ax, "deepmimo_o2", classic_styles, results)

    model_specs = [
        (
            {"label": "LDGEC-SURE test on O1 scenario", "model": "ldgec", "loss": "sure", "data": "deepmimo_o1"},
            {"label": "LDGEC-SURE test on O1 scenario", "color": RED, "marker": "d", "markerfacecolor": "none"},
        ),
        (
            {"label": "DEQ-GSURE test on O1 scenario", "model": "deq", "loss": "gsure", "data": "deepmimo_o1"},
            {"label": "DEQ-GSURE test on O1 scenario", "color": BLUE, "marker": "*", "markerfacecolor": BLUE},
        ),
        (
            {"label": "DNN-NMSE test on O1 scenario", "model": "dnn", "loss": "nmse", "data": "deepmimo_o1"},
            {"label": "DNN-NMSE test on O1 scenario", "color": GREEN, "marker": "s", "markerfacecolor": GREEN},
        ),
        (
            {"label": "LDGEC-SURE test on O2 scenario", "model": "ldgec", "loss": "sure", "data": "deepmimo_o2"},
            {"label": "LDGEC-SURE test on O2 scenario", "color": DARK_TEAL, "marker": "h", "markerfacecolor": DARK_TEAL},
        ),
        (
            {"label": "DEQ-GSURE test on O2 scenario", "model": "deq", "loss": "gsure", "data": "deepmimo_o2"},
            {"label": "DEQ-GSURE test on O2 scenario", "color": PURPLE, "marker": "^", "markerfacecolor": PURPLE},
        ),
        (
            {"label": "DNN-NMSE  test on O2 scenario", "model": "dnn", "loss": "nmse", "data": "deepmimo_o2"},
            {"label": "DNN-NMSE  test on O2 scenario", "color": CYAN, "marker": "p", "markerfacecolor": CYAN, "linestyle": "-."},
        ),
    ]

    for spec, style in model_specs:
        values = evaluate_model_curve(spec, device)
        if values is None:
            continue
        plot_curve(ax, SNR_LIST, values, style)
        results[style["label"]] = values

    apply_snr_axes(ax, legend_fontsize=10)
    save_figure(fig, FIGURE_DIR / "scenario_o2_nmse_vs_snr.pdf")
    save_torch({"snr": list(SNR_LIST), "nmse_db": results}, RESULT_DIR / "scenario_o2_results.pth")


def compute_layerwise_values(out):
    """Compute layerwise residual and NMSE arrays from an evaluated DEQ output."""
    net = out["net"]
    channels = out["channels"].detach().cpu()
    layerwise_h = [value.detach().cpu() for value in net.layerwise_h]
    h_star = layerwise_h[-1]
    nmse_den = torch.sum(channels ** 2).clamp_min(1e-12)
    residual_den = torch.linalg.vector_norm(h_star, ord=2, dim=1).sum().clamp_min(1e-12)

    nmse_values = []
    residual_values = []
    for ht in layerwise_h:
        nmse_num = torch.sum((ht - channels) ** 2).clamp_min(1e-30)
        residual_num = torch.linalg.vector_norm(ht - h_star, ord=2, dim=1).sum()
        nmse_values.append(float((10.0 * torch.log10(nmse_num / nmse_den)).item()))
        residual_values.append(float((residual_num / residual_den).item()))
    return np.asarray(residual_values), np.asarray(nmse_values), np.arange(1, len(nmse_values) + 1)


def run_nfes(device):
    """Plot layerwise residual and NMSE for DEQ-GSURE over SNRs."""
    spec = {"label": "DEQ-GSURE nFEs", "model": "deq", "loss": "gsure", "data": "deepmimo_o1"}
    if not model_checkpoint_complete(spec):
        return

    train_iters = 15
    markers = ["o", "s", "+", "d", "^"]
    residuals = {}
    nmses = {}
    iters = {}
    all_positive_residuals = []
    all_nmses = []
    tmax = 0

    for snr in SNR_LIST:
        out = evaluate_deq("gsure", snr, device, A_type=UNITARY, data="deepmimo_o1", record_layerwise=True)
        if out is None:
            print(f"Skip SNR = {snr} dB in nFEs plot: evaluation returned no result.")
            continue
        residual, nmse, itr = compute_layerwise_values(out)
        residuals[snr] = residual
        nmses[snr] = nmse
        iters[snr] = itr
        all_positive_residuals.extend(residual[residual > 0].tolist())
        all_nmses.extend(nmse.tolist())
        tmax = max(tmax, int(itr[-1]))

    if not residuals:
        print("No nFEs data was available to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3))
    ax_res, ax_nmse = axes

    for idx, snr in enumerate(SNR_LIST):
        if snr not in residuals:
            continue
        ax_res.plot(
            iters[snr],
            residuals[snr],
            linewidth=2,
            marker=markers[idx],
            markersize=7,
            markerfacecolor="none",
            label=f"SNR = {snr} dB",
        )
    ax_res.set_yscale("log")
    ax_res.set_xlim(0, tmax)
    if all_positive_residuals:
        ax_res.set_ylim(min(all_positive_residuals) * 0.8, max(all_positive_residuals) * 1.2)
    ax_res.set_xlabel(r"Iteration $d$")
    ax_res.set_ylabel(r"$\mathrm{E}\{\|\mathbf{h}^{(d)}-\mathbf{h}^{\star}\|_2\}/\mathrm{E}\{\|\mathbf{h}^{\star}\|_2\}$")
    ax_res.grid(True)
    ax_res.axvline(train_iters, linestyle="--", linewidth=2, color=(0.3, 0.3, 0.3), label="iterations at training")
    ax_res.legend(loc="upper left")

    for idx, snr in enumerate(SNR_LIST):
        if snr not in nmses:
            continue
        ax_nmse.plot(
            iters[snr],
            nmses[snr],
            linewidth=2,
            marker=markers[idx],
            markersize=7,
            markerfacecolor="none",
        )
    ax_nmse.set_xlim(0, tmax)
    if all_nmses:
        ax_nmse.set_ylim(min(all_nmses) - 1, max(all_nmses) + 1)
    ax_nmse.set_xlabel(r"Iteration $d$")
    ax_nmse.set_ylabel("NMSE (dB)")
    ax_nmse.grid(True)
    ax_nmse.axvline(train_iters, linestyle="--", linewidth=2, color=(0.3, 0.3, 0.3))

    save_figure(fig, FIGURE_DIR / "nfes_layerwise_residual_nmse.pdf")
    save_torch(
        {"snr": list(SNR_LIST), "iters": iters, "residual": residuals, "nmse_db": nmses},
        RESULT_DIR / "nfes_layerwise_results.pth",
    )


def run_beta_omega():
    """Plot beta and omega/NMSE diagnostics from the supplied figure data."""
    snrs = np.asarray(SNR_LIST)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    ax_beta, ax_nmse = axes

    beta_gsure = [0.1708, 0.0884, 0.0467, 0.0271, 0.0157]
    beta_nmse = [0.1453, 0.0855, 0.0461, 0.0269, 0.0152]
    ax_beta.plot(snrs, beta_gsure, "d-", color=RED, markerfacecolor=RED, markersize=MARKER_SIZE, linewidth=1.5, label="DEQ-GSURE")
    ax_beta.plot(snrs, beta_nmse, "*-", color=BLUE, markerfacecolor=BLUE, markersize=MARKER_SIZE, linewidth=1.5, label="DEQ-NMSE")
    ax_beta.set_xlabel("SNR (dB)")
    ax_beta.set_ylabel(r"$\beta$")
    ax_beta.set_xticks(SNR_LIST)
    ax_beta.grid(True)
    ax_beta.legend(loc="best")

    nmse_curves = [
        ([-14.9300, -20.6243, -26.1386, -31.2773, -35.5151], "*-", BLUE, "DEQ-GSURE, bias=True"),
        ([-14.7072, -20.5251, -26.1043, -30.8559, -35.5367], "d-", RED, "DEQ-GSURE, bias=False"),
        ([-15.5952, -21.2711, -26.2475, -31.1131, -35.1983], "h-", DARK_TEAL, "DEQ-NMSE,  bias=True"),
        ([-16.0120, -20.8152, -26.1919, -30.9449, -35.8336], "^-", PURPLE, "DEQ-NMSE,  bias=False"),
    ]
    for values, line_marker, color, label in nmse_curves:
        ax_nmse.plot(snrs, values, line_marker, color=color, markerfacecolor=color, markersize=MARKER_SIZE, linewidth=1.5, label=label)
    ax_nmse.set_xlabel("SNR (dB)")
    ax_nmse.set_ylabel("NMSE (dB)")
    ax_nmse.set_xticks(SNR_LIST)
    ax_nmse.grid(True)
    ax_nmse.legend(loc="best")

    right_axis = ax_nmse.twinx()
    nmse_ticks = ax_nmse.get_yticks()
    omega_ticks = np.round(np.sqrt(10.0 ** (nmse_ticks / 10.0)), 2)
    right_axis.set_ylim(ax_nmse.get_ylim())
    right_axis.set_yticks(nmse_ticks)
    right_axis.set_yticklabels([f"{tick:g}" for tick in omega_ticks])
    right_axis.set_ylabel(r"$\omega$")
    right_axis.tick_params(axis="y", colors="black")
    right_axis.spines["right"].set_color("black")

    save_figure(fig, FIGURE_DIR / "beta_omega.pdf")
    save_torch(
        {
            "snr": list(SNR_LIST),
            "beta_gsure": beta_gsure,
            "beta_nmse": beta_nmse,
            "nmse_curves": {label: values for values, _, _, label in nmse_curves},
        },
        RESULT_DIR / "beta_omega_results.pth",
    )


def run_nmse_vs_A_type(device):
    """Plot DEQ NMSE against sensing-matrix type."""
    results = {}
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    curve_specs = [
        (
            {"label": r"Loss = GSURE, $\mathbf{A}=\mathrm{unitary}$", "model": "deq", "loss": "gsure", "A_type": "unitary"},
            {"label": r"Loss = GSURE, $\mathbf{A}=\mathrm{unitary}$", "marker": "o"},
        ),
        (
            {"label": r"Loss = GSURE, $\mathbf{A}=\mathrm{gaussian}$", "model": "deq", "loss": "gsure", "A_type": "gaussian"},
            {"label": r"Loss = GSURE, $\mathbf{A}=\mathrm{gaussian}$", "marker": "s"},
        ),
        (
            {"label": r"Loss = GSURE, $\mathbf{A}=\mathrm{bernoulli}$", "model": "deq", "loss": "gsure", "A_type": "bernoulli"},
            {"label": r"Loss = GSURE, $\mathbf{A}=\mathrm{bernoulli}$", "marker": "d"},
        ),
        (
            {"label": r"Loss = NMSE,  $\mathbf{A}=\mathrm{unitary}$", "model": "deq", "loss": "nmse", "A_type": "unitary"},
            {"label": r"Loss = NMSE,  $\mathbf{A}=\mathrm{unitary}$", "marker": "^"},
        ),
        (
            {"label": r"Loss = NMSE,  $\mathbf{A}=\mathrm{gaussian}$", "model": "deq", "loss": "nmse", "A_type": "gaussian"},
            {"label": r"Loss = NMSE,  $\mathbf{A}=\mathrm{gaussian}$", "marker": "v"},
        ),
        (
            {"label": r"Loss = NMSE,  $\mathbf{A}=\mathrm{bernoulli}$", "model": "deq", "loss": "nmse", "A_type": "bernoulli"},
            {"label": r"Loss = NMSE,  $\mathbf{A}=\mathrm{bernoulli}$", "marker": "x"},
        ),
    ]

    for spec, style in curve_specs:
        values = evaluate_model_curve(spec, device)
        if values is None:
            continue
        plot_curve(ax, SNR_LIST, values, {**style, "linewidth": 1.5})
        results[style["label"]] = values

    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("NMSE (dB)", fontsize=12)
    ax.set_xticks(SNR_LIST)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)
    save_figure(fig, FIGURE_DIR / "nmse_vs_A_type.pdf")
    save_torch({"snr": list(SNR_LIST), "nmse_db": results}, RESULT_DIR / "nmse_vs_A_type_results.pth")


def available_sigma_values(diagonal):
    """Return available covariance-perturbation checkpoint sigma values."""
    tag = "noisy_diagC" if diagonal else "noisy_C"
    root = Path("trained network") / UNITARY
    values = []
    pattern = re.compile(rf"^deq_gsure_{tag}([0-9.]+)_10dB\.pth$")
    for path in root.glob(f"deq_gsure_{tag}*_10dB.pth"):
        match = pattern.match(path.name)
        if match:
            values.append(float(match.group(1)))
    return sorted(set(values))


def evaluate_sigma_curve(device, diagonal):
    """Evaluate one sigma_e curve for full or diagonal perturbations."""
    values = []
    for sigma in available_sigma_values(diagonal):
        checkpoint = find_checkpoint("deq", "gsure", 10, A_type=UNITARY, data="deepmimo", finite_snapshot=sigma, diagonal=diagonal)
        if checkpoint is None:
            continue
        out = evaluate_deq("gsure", 10, device, A_type=UNITARY, data="deepmimo_o1", finite_snapshot=sigma, diagonal=diagonal)
        if out is None:
            continue
        values.append((sigma, out["nmse"]))
    if not values:
        return np.asarray([]), np.asarray([])
    values = sorted(values, key=lambda item: item[0])
    return np.asarray([item[0] for item in values]), np.asarray([item[1] for item in values])


def run_nmse_vs_sigma_e(device):
    """Plot NMSE versus covariance perturbation strength."""
    exact_out = evaluate_deq("gsure", 10, device, A_type=UNITARY, data="deepmimo_o1")
    full_sigma, full_nmse = evaluate_sigma_curve(device, diagonal=False)
    diag_sigma, diag_nmse = evaluate_sigma_curve(device, diagonal=True)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    if exact_out is not None:
        exact_nmse = exact_out["nmse"]
        x_ref = np.logspace(np.log10(1e-3), np.log10(5e-1), 300)
        ax.semilogx(x_ref, exact_nmse * np.ones_like(x_ref), "--", linewidth=2.0, color=DARK_GRAY, label="Infinite snapshot")
    else:
        print("Skip Infinite snapshot reference: missing baseline DEQ-GSURE checkpoint at 10 dB.")

    if full_sigma.size:
        ax.semilogx(
            full_sigma,
            full_nmse,
            "-o",
            linewidth=2.0,
            markersize=7,
            color=(0.0, 0.4470, 0.7410),
            markerfacecolor="white",
            label="Full perturbation",
        )
    else:
        print("Skip Full perturbation: no noisy_C checkpoints were available.")

    if diag_sigma.size:
        ax.semilogx(
            diag_sigma,
            diag_nmse,
            "-s",
            linewidth=2.0,
            markersize=6.5,
            color=(0.8500, 0.3250, 0.0980),
            markerfacecolor="white",
            label="Diagonal perturbation",
        )
    else:
        print("Skip Diagonal perturbation: no noisy_diagC checkpoints were available.")

    ax.grid(True, which="both")
    ax.set_xlabel(r"$\sigma_{e}^{2}$", fontsize=12)
    ax.set_ylabel("NMSE (dB)", fontsize=12)
    ax.set_xscale("log")
    ticks = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0.001", "0.002", "0.005", "0.01", "0.02", "0.05", "0.1", "0.2", "0.5"])
    ax.set_xlim(1e-3, 5e-1)
    ax.set_ylim(-21, 5)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(loc="upper left")

    save_figure(fig, FIGURE_DIR / "nmse_vs_sigma_e.pdf")
    save_torch(
        {
            "full_sigma": full_sigma,
            "full_nmse_db": full_nmse,
            "diag_sigma": diag_sigma,
            "diag_nmse_db": diag_nmse,
            "infinite_snapshot_nmse_db": None if exact_out is None else exact_out["nmse"],
        },
        RESULT_DIR / "nmse_vs_sigma_e_results.pth",
    )


def main():
    """Parse CLI arguments and dispatch one plotting task."""
    args = build_parser().parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_all_devices(device)
    RESULT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    print(f"Using device: {device}")

    if args.scenario_o1:
        run_scenario_o1(device)
    elif args.scenario_o2:
        run_scenario_o2(device)
    elif args.nFEs:
        run_nfes(device)
    elif args.beta_omega:
        run_beta_omega()
    elif args.nmse_vs_A_type:
        run_nmse_vs_A_type(device)
    elif args.nmse_vs_sigma_e:
        run_nmse_vs_sigma_e(device)


if __name__ == "__main__":
    main()
