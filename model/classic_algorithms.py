#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.data_utils import SNR_LIST, load_complex_A, load_raw_o1_data, load_raw_o2_data, load_raw_synthetic_data
from utils.metrics import nmse_db_numpy

np.seterr(all="ignore")


def complex_soft_threshold(r, rvar, theta):
    """Apply complex soft-thresholding used by AMP.

    :param r: Current pseudo-data matrix with shape (N, batch).
    :param rvar: Per-sample variance estimate.
    :param theta: Threshold parameters.
    :return: Tuple (xhat, dxdr, dxdr_h).
    """
    lam = theta[0] * np.sqrt(rvar)
    lam = np.maximum(lam, np.zeros_like(lam))
    abs_r = np.maximum(np.abs(r), 1e-12)
    arml = abs_r - lam
    xhat = r * (np.maximum(arml, np.zeros_like(arml)) / abs_r).astype(complex)
    dxdr = np.mean(((abs_r - lam) > 0).astype(complex) * (1 - lam / (2 * abs_r)).astype(complex), 0)
    dxdr_h = np.zeros_like(dxdr)
    if len(theta) == 2:
        xhat = xhat * theta[1]
        dxdr = dxdr * theta[1]
        dxdr_h = dxdr_h * theta[1]
    return xhat, dxdr, dxdr_h


def sparsity_combiner(r, rvar, sigma1):
    """Apply the Bernoulli-Gaussian sparsity combiner.

    :param r: Current pseudo-data matrix with shape (N, batch).
    :param rvar: Per-sample variance estimate.
    :param sigma1: Active-entry variance.
    :return: Tuple (xhat, dxdr, dxdr_h, xmmse).
    """
    lam = 3 / 256
    sigma2 = sigma1 + rvar
    rho = lam / sigma2
    abs2 = np.conj(r) * r
    N = rho * np.exp(-abs2 / sigma2) * r / (1 + rvar / sigma1)
    D1 = rho * np.exp(-abs2 / sigma2)
    D2 = ((1 - lam) / rvar) * np.exp(-abs2 / rvar)
    D = D1 + D2
    xhat = N / np.maximum(D, 1e-12)
    miu = abs2 / (1 + rvar / sigma1) ** 2
    posterior_variance = sigma1 * rvar / (sigma1 + rvar)
    ex2 = (miu + posterior_variance) * (D1 / np.maximum(D, 1e-12))
    xmmse = ex2 - np.conj(xhat) * xhat
    return xhat, None, None, xmmse


class ClassicAlgorithms:
    """Classic sparse recovery baselines for complex batch-first test data."""

    def __init__(self, A_type="unitary", data="deepmimo_o1", omp_iters=24):
        """Initialize classic sparse recovery algorithms.

        :param A_type: Matrix type. Only unitary is supported for these baselines.
        :param data: Test dataset in deepmimo_o1, deepmimo_o2 or synthetic.
        :param omp_iters: Number of OMP iterations.
        :return: None.
        """
        if A_type != "unitary":
            raise ValueError("Classic AMP/OAMP baselines only support A_type='unitary'.")
        if data not in {"deepmimo_o1", "deepmimo_o2", "synthetic"}:
            raise ValueError("data must be one of: deepmimo_o1, deepmimo_o2, synthetic.")
        self.A_type = A_type
        self.data = data
        self.omp_iters = omp_iters
        self.A = load_complex_A(A_type)

    def load_test_data(self, snr):
        """Load batch-first complex test data for one SNR.

        :param snr: Fixed SNR in dB.
        :return: Tuple (channels, measurements, sigma_squared) as complex numpy arrays.
        """
        if self.data == "deepmimo_o1":
            return load_raw_o1_data(self.A_type, snr=snr, split="test")
        if self.data == "deepmimo_o2":
            return load_raw_o2_data(snr=snr)
        if self.data == "synthetic":
            return load_raw_synthetic_data(snr=snr, split="test")
        raise ValueError("data must be one of: deepmimo_o1, deepmimo_o2, synthetic.")

    def omp(self, y):
        """Run OMP for a single measurement vector.

        :param y: Complex measurement vector with shape (M, 1).
        :return: Estimated channel vector with shape (N, 1).
        """
        M, N = self.A.shape
        xhat = np.zeros((N, 1), dtype=complex)
        pos = []
        for _ in range(self.omp_iters):
            residual = self.A @ xhat - y
            product = self.A.conj().T @ residual
            idx = int(np.argmax(np.abs(product)))
            if idx not in pos:
                pos.append(idx)
            As = self.A[:, pos]
            xhat[pos] = np.linalg.pinv(As) @ y
        return xhat

    def amp(self, y):
        """Run AMP for a batch represented column-wise.

        :param y: Complex measurement matrix with shape (M, batch).
        :return: Estimated channel matrix with shape (N, batch).
        """
        M, N = self.A.shape
        S = y.shape[1]
        T = 30
        alf = [1.1402]
        Bmf = self.A.conj().T
        xhat = np.zeros((N, S), dtype=complex)
        u = Bmf @ y
        z = u
        for _ in range(T):
            rhat = xhat + z
            rvar = np.sum(np.square(np.abs(z)), 0) / (N - 1)
            xhat, dxdr, dxdr_h = complex_soft_threshold(rhat, rvar, alf)
            b0 = dxdr * (N / M)
            b1 = dxdr_h * (N / M)
            z = u - (Bmf @ self.A) @ xhat + b0 * z + b1 * np.conj(z)
        return xhat

    def oamp(self, y, sigma_squared):
        """Run OAMP for a batch represented column-wise.

        :param y: Complex measurement matrix with shape (M, batch).
        :param sigma_squared: Noise variance with shape (1, batch).
        :return: Estimated channel matrix with shape (N, batch).
        """
        M, N = self.A.shape
        rho = M / N
        S = y.shape[1]
        T = 30
        Ah = self.A.conj().T
        u = Ah @ y
        B = Ah @ self.A
        x_apri = np.zeros((N, S), dtype=complex)
        v_apri = np.ones((1, S))
        x_bpost = x_apri
        for _ in range(T):
            x_apost = x_apri + (u - B @ x_apri) * v_apri / (v_apri + sigma_squared)
            v_apost = v_apri - rho * v_apri ** 2 / (v_apri + sigma_squared)
            vext = 1 / (1 / v_apost - 1 / v_apri)
            x_bpri = vext * (x_apost / v_apost - x_apri / v_apri)
            v_bpri = vext
            x_bpost, _, _, v_bpost = sparsity_combiner(x_bpri, v_bpri, N / 3)
            v_bpost = np.mean(v_bpost)
            vext = 1 / (1 / v_bpost - 1 / v_bpri)
            x_apri = vext * (x_bpost / v_bpost - x_bpri / v_bpri)
            v_apri = vext
        return x_bpost

    def evaluate(self, algorithms, snrs=SNR_LIST):
        """Evaluate selected classic algorithms over SNR values.

        :param algorithms: Iterable containing omp, amp and/or oamp.
        :param snrs: Iterable of fixed SNR values.
        :return: Dictionary with NMSE curves and runtime values.
        """
        results = {}
        runtimes = {}
        print(f"tested on: {self.data}")
        for alg in algorithms:
            nmse = []
            elapsed = []
            for snr in snrs:
                h_batch, y_batch, sigma_squared_batch = self.load_test_data(snr)
                h = h_batch.T
                y = y_batch.T
                sigma_squared = sigma_squared_batch.T
                start = time.perf_counter()
                if alg == "omp":
                    h_hat = np.zeros_like(h)
                    for i in range(y.shape[1]):
                        h_hat[:, i] = self.omp(y[:, i].reshape(-1, 1)).squeeze()
                elif alg == "amp":
                    h_hat = self.amp(y)
                elif alg == "oamp":
                    h_hat = self.oamp(y, sigma_squared)
                else:
                    raise ValueError("alg must be one of: omp, amp, oamp")
                elapsed.append(time.perf_counter() - start)
                nmse.append(float(nmse_db_numpy(h_hat, h)))
                print(f"{alg.upper()} | SNR={snr} dB | NMSE={nmse[-1]:.4f} dB | time={elapsed[-1]:.3f}s")
            results[alg] = nmse
            runtimes[alg] = float(np.mean(elapsed))
        print("\naverage time cost(all SNR):")
        for alg in algorithms:
            print(f"  {alg.upper()}: {runtimes[alg]:.3f}s")
        return {"data": self.data, "snr": list(snrs), "nmse_db": results, "runtime_s": runtimes}


def main():
    """Parse CLI arguments and run selected classic baselines.

    :return: None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", nargs="+", choices=["omp", "amp", "oamp"], default=["omp", "amp", "oamp"])
    parser.add_argument("--data", choices=["deepmimo_o1", "deepmimo_o2", "synthetic"], default="deepmimo_o1")
    args = parser.parse_args()
    runner = ClassicAlgorithms(A_type="unitary", data=args.data)
    state = runner.evaluate(args.alg)
    out = Path(__file__).resolve().parent / f"classic_algorithms_results_{args.data}.pth"
    torch.save(state, out)
    print(f"Saved classic algorithm results to {out}")


if __name__ == "__main__":
    main()
