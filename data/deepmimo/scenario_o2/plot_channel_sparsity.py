#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def project_root():
    """Return the project root directory.

    :return: Path to the tsp_deq_gsure root.
    """
    return Path(__file__).resolve().parents[3]


def channel_dir():
    """Return the DeepMIMO O2 channel directory.

    :return: Path to scenario_o2/channels.
    """
    return Path(__file__).resolve().parent / "channels"


def load_channel(path):
    """Load a batch-first O2 channel matrix.

    :param path: Path to an O2 channel .mat file.
    :return: Complex array with shape (batch, 256).
    """
    data = loadmat(path)
    if "h" not in data:
        raise KeyError(f"{path} must contain variable 'h'.")
    return data["h"]


def select_channel(scene=None, sample=None, seed=None):
    """Select one O2 channel sample for inspection.

    :param scene: Optional substring of the scene file name.
    :param sample: Optional sample row index.
    :param seed: Optional random seed.
    :return: Tuple (path, sample_index, channel_vector).
    """
    paths = sorted(channel_dir().glob("*.mat"))
    if not paths:
        raise FileNotFoundError(f"No .mat channel file was found in {channel_dir()}.")

    rng = np.random.default_rng(seed)
    if scene is None:
        path = paths[int(rng.integers(len(paths)))]
    else:
        matches = [p for p in paths if scene in p.stem]
        if not matches:
            names = "\n".join(p.name for p in paths)
            raise FileNotFoundError(f"No scene matched '{scene}'. Available files:\n{names}")
        path = matches[0]

    x = load_channel(path)
    if x.ndim != 2:
        raise ValueError(f"Expected channel matrix with shape (batch, N), got {x.shape}.")
    if sample is None:
        sample = int(rng.integers(x.shape[0]))
    if sample < 0 or sample >= x.shape[0]:
        raise ValueError(f"sample must be in [0, {x.shape[0] - 1}], got {sample}.")
    return path, sample, x[sample, :]


def sparsity_metrics(h):
    """Compute simple sparsity and power diagnostics.

    :param h: Complex channel vector with shape (256,).
    :return: Dictionary of power and energy concentration metrics.
    """
    magnitude = np.abs(h)
    energy = magnitude ** 2
    power = float(np.sum(energy))
    mean_power = float(np.mean(energy))
    max_magnitude = float(np.max(magnitude))
    threshold = 0.01 * max_magnitude
    support_1pct = int(np.sum(magnitude >= threshold))
    sorted_energy = np.sort(energy)[::-1]
    cumulative = np.cumsum(sorted_energy) / max(power, 1e-12)

    def top_ratio(k):
        """Compute the top-k energy ratio.

        :param k: Number of largest coefficients.
        :return: Energy ratio in [0, 1].
        """
        k = min(k, energy.size)
        return float(np.sum(sorted_energy[:k]) / max(power, 1e-12))

    return {
        "power": power,
        "mean_power": mean_power,
        "max_magnitude": max_magnitude,
        "support_1pct": support_1pct,
        "top5_energy": top_ratio(5),
        "top10_energy": top_ratio(10),
        "top20_energy": top_ratio(20),
        "cumulative_energy": cumulative,
    }


def plot_channel(path, sample, h, output):
    """Plot channel magnitude and cumulative energy.

    :param path: Source channel file path.
    :param sample: Sample row index.
    :param h: Complex channel vector.
    :param output: Output figure path.
    :return: Tuple (output_path, metrics).
    """
    magnitude = np.abs(h)
    metrics = sparsity_metrics(h)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), constrained_layout=True)

    axes[0].stem(np.arange(h.size), magnitude, basefmt=" ", linefmt="C0-", markerfmt="C0o")
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("|h|")
    axes[0].set_title(f"{path.name}, sample={sample}")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(np.arange(1, h.size + 1), metrics["cumulative_energy"], color="C1")
    for k in (5, 10, 20):
        axes[1].axvline(k, color="0.65", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Largest coefficients kept")
    axes[1].set_ylabel("Cumulative energy ratio")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(True, alpha=0.3)

    text = (
        f"||h||_2^2 = {metrics['power']:.6g}\n"
        f"mean |h|^2 = {metrics['mean_power']:.6g}\n"
        f">=1% peak count = {metrics['support_1pct']}/{h.size}\n"
        f"top-5/10/20 energy = "
        f"{metrics['top5_energy']:.3f} / {metrics['top10_energy']:.3f} / {metrics['top20_energy']:.3f}"
    )
    axes[1].text(
        0.98,
        0.08,
        text,
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75"},
    )

    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output, metrics


def main():
    """Parse arguments, draw one O2 channel and print diagnostics.

    :return: None.
    """
    parser = argparse.ArgumentParser(description="Plot one random DeepMIMO O2 channel and report its power.")
    parser.add_argument("--scene", default=None, help="Substring of the O2 scene file name.")
    parser.add_argument("--sample", type=int, default=None, help="Column index in the selected channel matrix.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for scene/sample selection.")
    parser.add_argument(
        "--output",
        default=str(project_root() / "figures" / "scenario_o2_channel_sparsity.png"),
    )
    args = parser.parse_args()

    path, sample, h = select_channel(scene=args.scene, sample=args.sample, seed=args.seed)
    output, metrics = plot_channel(path, sample, h, args.output)

    print(f"Scene file: {path}")
    print(f"Sample index: {sample}")
    print(f"Channel dimension: {h.size}")
    print(f"||h||_2^2: {metrics['power']:.8g}")
    print(f"mean |h|^2: {metrics['mean_power']:.8g}")
    print(f">=1% peak count: {metrics['support_1pct']}/{h.size}")
    print(f"top-5 energy ratio: {metrics['top5_energy']:.6f}")
    print(f"top-10 energy ratio: {metrics['top10_energy']:.6f}")
    print(f"top-20 energy ratio: {metrics['top20_energy']:.6f}")
    print(f"Saved figure: {output}")


if __name__ == "__main__":
    main()
