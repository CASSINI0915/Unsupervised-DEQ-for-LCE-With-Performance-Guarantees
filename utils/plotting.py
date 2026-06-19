#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_nmse_curve(results, snrs, title, output_path):
    """Save an NMSE-versus-SNR curve.
    
    :param results: Mapping from curve label to NMSE values in dB.
    :param snrs: SNR values used as the x-axis.
    :param title: Figure title.
    :param output_path: Output figure path.
    :return: Path to the saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    for label, values in results.items():
        if values is None:
            continue
        plt.plot(snrs, values, marker="o", label=label)
    plt.xlabel("SNR (dB)")
    plt.ylabel("NMSE (dB)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def save_layerwise_curve(values, ylabel, output_path):
    """Save a layerwise diagnostic curve indexed by nFE.
    
    :param values: Sequence of y-axis values, one per nFE.
    :param ylabel: Y-axis label.
    :param output_path: Output figure path.
    :return: Path to the saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(np.arange(1, len(values) + 1), values, marker="o")
    plt.xlabel("nFEs")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path
