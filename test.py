#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import numpy as np
import torch

from utils.checkpoint import find_checkpoint, infer_training_data_from_checkpoint
from utils.data_utils import SNR_LIST
from utils.evaluation import checkpoint_data_for_test_data, evaluate_deq, evaluate_model, set_all_devices, validate_eval_config


def build_parser():
    """Build the command-line parser for test-set NMSE evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate one trained model on all released test SNRs.")
    parser.add_argument("--A_type", choices=["gaussian", "bernoulli", "unitary"], default="unitary")
    parser.add_argument("--model", choices=["deq", "dnn", "ldgec"], default="deq")
    parser.add_argument("--data", choices=["deepmimo_o1", "deepmimo_o2", "synthetic"], default="deepmimo_o1")
    parser.add_argument("--loss", choices=["sure", "gsure", "nmse"], default="gsure")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--finite_snapshot", nargs="+", default=["0.0"], help="One or more sigma_e^2 values for covariance perturbation. 0 disables this branch.")
    parser.add_argument("--diagonal", action="store_true", help="When finite_snapshot > 0, select the diagonal covariance-perturbation checkpoint.")
    parser.add_argument("--estimate_lip", action="store_true", help="Estimate empirical Lipschitz constants for DEQ-GSURE and DEQ-NMSE.")
    return parser


def parse_finite_snapshot_values(values):
    """Parse one or more finite-snapshot values from CLI tokens."""
    tokens = []
    for value in values:
        cleaned = value.replace("[", "").replace("]", "").replace(",", " ")
        tokens.extend(part for part in cleaned.split() if part)
    return [float(token) for token in tokens] if tokens else [0.0]


def normalize_args(args):
    """Apply the same finite-snapshot normalization used by training."""
    args.finite_snapshot = parse_finite_snapshot_values(args.finite_snapshot)
    if args.estimate_lip:
        args.A_type = "unitary"
        args.data = "deepmimo_o1"
        args.model = "deq"
        args.loss = "gsure"
        args.finite_snapshot = [0.0]
        args.diagonal = False
        return args
    uses_finite_snapshot = any(value > 0 for value in args.finite_snapshot)
    if uses_finite_snapshot:
        if args.A_type != "unitary" or args.model != "deq" or args.loss != "gsure":
            print("finite_snapshot is enabled: forcing A_type=unitary, model=deq, loss=gsure.")
        args.A_type = "unitary"
        args.model = "deq"
        args.loss = "gsure"
    elif args.diagonal:
        print("--diagonal is ignored because finite_snapshot is 0.")
        args.diagonal = False
    if args.data == "synthetic":
        if args.A_type != "unitary" or args.model != "deq":
            print("synthetic data is enabled: forcing A_type=unitary and model=deq.")
        args.A_type = "unitary"
        args.model = "deq"
    if args.data == "deepmimo_o2" and args.A_type != "unitary":
        print("deepmimo_o2 data is enabled: forcing A_type=unitary.")
        args.A_type = "unitary"
    return args


def find_reference_checkpoint(args, finite_snapshot):
    """Find one checkpoint for printing the training/test dataset summary."""
    ckpt_data = checkpoint_data_for_test_data(args.data)
    for snr in SNR_LIST:
        ckpt = find_checkpoint(
            args.model,
            args.loss,
            snr,
            A_type=args.A_type,
            data=ckpt_data,
            finite_snapshot=finite_snapshot,
            diagonal=args.diagonal,
        )
        if ckpt is not None:
            return ckpt
    return None


def run_estimate_lip(device):
    """Print empirical Lipschitz estimates for released DEQ checkpoints."""
    for loss in ["gsure", "nmse"]:
        print(f"The empirical Lipschitz constants of DEQ-{loss.upper()}:")
        for snr in SNR_LIST:
            out = evaluate_deq(loss, snr, device, A_type="unitary", data="deepmimo_o1")
            if out is not None:
                print(f"SNR={snr} dB | estimated L1*L2 = {out['lip']}")
        if loss != "nmse":
            print()


def main():
    """Parse CLI arguments and evaluate one configuration over all SNRs."""
    parser = build_parser()
    args = normalize_args(parser.parse_args())
    try:
        validate_eval_config(args.model, args.loss, data=args.data, A_type=args.A_type, enforce_release_matrix=True)
    except ValueError as exc:
        parser.error(str(exc))

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    set_all_devices(device)

    if args.estimate_lip:
        run_estimate_lip(device)
        return

    for index, finite_snapshot in enumerate(args.finite_snapshot):
        if index > 0:
            print()
        ckpt = find_reference_checkpoint(args, finite_snapshot)
        print(f"trained on: {infer_training_data_from_checkpoint(ckpt)} | tested on: {args.data}")
        print(f"sigma_e^2: {finite_snapshot:g}")

        runtimes = []
        label = None
        for snr in SNR_LIST:
            out = evaluate_model(
                args.model,
                args.loss,
                snr,
                device,
                A_type=args.A_type,
                data=args.data,
                finite_snapshot=finite_snapshot,
                diagonal=args.diagonal,
            )
            if out is None:
                continue
            label = out["label"]
            runtimes.append(out["runtime"])
            print(f"{out['label']} | sigma_e^2={finite_snapshot:g} | SNR={snr} dB | NMSE={out['nmse']:.4f} dB | time={out['runtime']:.3f}s")

        print("\naverage time cost(all SNR):")
        if runtimes and label is not None:
            print(f"  {label}: {float(np.mean(runtimes)):.3f}s")
        else:
            print(f"  {args.model.upper()}-{args.loss.upper()}: nan")


if __name__ == "__main__":
    main()
