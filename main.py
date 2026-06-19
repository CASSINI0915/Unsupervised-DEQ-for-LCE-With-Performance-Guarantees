#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import datetime as _dt
import os

import numpy as np
import torch

import trainer as deq_trainer_module
import utils.utils as legacy_utils
from model import deq as deq_module
from model import dnn as dnn_module
from model import ldgec as ldgec_module
from model.deq import DEQ
from model.dnn import DNN
from trainer import Trainer
from utils.checkpoint import checkpoint_dir, checkpoint_name
from utils.data_utils import SNR_LIST, load_cs_matrix, prepare_deq_training_data, prepare_raw_o1_loaders


def parse_snr_values(values):
    """Parse one or more SNR values from CLI tokens.

    :param values: Raw CLI token list.
    :return: List of integer SNR values.
    """
    if values is None:
        return [20, 15, 10, 5, 0]
    tokens = []
    for value in values:
        cleaned = value.replace("[", "").replace("]", "").replace(",", " ")
        tokens.extend(part for part in cleaned.split() if part)
    snrs = [int(v) for v in tokens]
    illegal = [v for v in snrs if v not in SNR_LIST]
    if illegal:
        raise argparse.ArgumentTypeError(f"SNR only allows {SNR_LIST}; got {illegal}")
    return snrs


def build_parser():
    """Build the command-line parser for training.

    :return: Configured argparse parser.
    """
    parser = argparse.ArgumentParser(description="Train models for the TSP paper release.")
    parser.add_argument("--A_type", choices=["gaussian", "bernoulli", "unitary"], default="unitary")
    parser.add_argument("--model", choices=["deq", "dnn", "ldgec"], default="deq")
    parser.add_argument("--loss", choices=["sure", "gsure", "nmse"], default="gsure")
    parser.add_argument("--data", choices=["deepmimo", "synthetic"], default="deepmimo")
    parser.add_argument("--snr", nargs="+", default=["20", "15", "10", "5", "0"], help="One or more SNR values from [0,5,10,15,20]. Default: 20 15 10 5 0. Examples: --snr 20, --snr 0 5, --snr [0,5]")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--finite_snapshot", type=float, default=0.0, help="Variance sigma_e^2 of covariance perturbation. 0 disables this branch.")
    parser.add_argument("--diagonal", action="store_true", help="When finite_snapshot > 0, perturb only the covariance diagonal.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    return parser


def set_all_devices(device):
    """Set the torch device for copied modules.

    :param device: Torch device used for training.
    :return: None.
    """
    deq_trainer_module.set_device(device)
    legacy_utils.set_device(device)
    deq_module.set_device(device)
    dnn_module.set_device(device)
    ldgec_module.set_device(device)


def validate_model_loss(parser, args):
    """Validate model/loss combinations after argument normalization."""
    valid_losses = {
        "deq": {"gsure", "nmse"},
        "dnn": {"gsure", "nmse"},
        "ldgec": {"sure", "nmse"},
    }
    if args.loss not in valid_losses[args.model]:
        allowed = ", ".join(sorted(valid_losses[args.model]))
        parser.error(f"--model {args.model} supports --loss in {{{allowed}}}; got {args.loss}.")


def save_checkpoint_versions(state, A_type, model, loss, snr, timestamp):
    """Save timestamped and latest checkpoint versions."""
    out_dir = checkpoint_dir(A_type)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp_path = out_dir / checkpoint_name(model, loss, snr, timestamp=timestamp)
    latest_path = out_dir / checkpoint_name(model, loss, snr)
    torch.save(state, timestamp_path)
    torch.save(state, latest_path)
    print(f"Saved: {timestamp_path}")
    print(f"Saved: {latest_path}")


def train_deq_or_dnn(args, snr, device):
    """Train or initialize a DEQ/DNN model for one SNR.

    :param args: Parsed command-line arguments.
    :param snr: Fixed SNR in dB.
    :param device: Torch device used for training.
    :return: None.
    """
    A, Sigma, (train_loader, validation_loader) = prepare_deq_training_data(args, snr, batch_size=args.batch_size)
    if args.model == "deq":
        net = DEQ(
            A=A,
            Sigma=Sigma,
            lat_layers=4,
            contraction_factor=0.99,
            eps=1e-3,
            max_depth=15,
            num_channels=32,
            bias=False, 
        ).to(device)
    else:
        net = DNN(dim_in=256, dim_out=256).to(device)

    trainer = Trainer(net, train_loader, validation_loader, A=A, Sigma=Sigma)
    if args.epochs <= 0:
        best_model = net.state_dict()
    elif args.loss == "gsure":
        _, _, best_model = trainer.train_by_gsure(epochs=args.epochs)
    else:
        _, _, best_model = trainer.train_by_nmse(epochs=args.epochs)

    timestamp = _dt.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    state = best_model if best_model is not None else net.state_dict()
    save_checkpoint_versions(state, args.A_type, args.model, args.loss, snr, timestamp)


def train_ldgec(args, snr, device):
    """Train or initialize the LDGEC baseline for one SNR.

    :param args: Parsed command-line arguments.
    :param snr: Fixed SNR in dB.
    :param device: Torch device used for training.
    :return: None.
    """
    if args.data != "deepmimo":
        raise NotImplementedError("LDGEC training in this release uses DeepMIMO O1 raw measurements.")

    from utils import ldgec_utils

    ldgec_utils.set_device(device)
    ldgec_module.set_device(device)
    A, _ = load_cs_matrix(args.A_type)
    train_loader, validation_loader = prepare_raw_o1_loaders(args.A_type, snr=snr, batch_size=args.batch_size)
    model = ldgec_module.LDGEC(num_layers=8, A=A).to(device)
    runner = ldgec_utils.Trainer(model, train_loader, validation_loader, learning_rate=1e-3, weight_decay=0)
    runner.supervision = args.loss == "nmse"
    runner.using_fine_tune = False
    if args.epochs > 0: 
        model = runner.layer_By_layer(epochs=args.epochs) # epochs per layer, not cumulated epochs
    timestamp = _dt.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    save_checkpoint_versions(model.state_dict(), args.A_type, "ldgec", args.loss, snr, timestamp)


def main():
    """Parse CLI arguments and run training jobs in order.

    :return: None.
    """
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    parser = build_parser()
    args = parser.parse_args()
    args.snr = parse_snr_values(args.snr)

    if args.finite_snapshot > 0:
        if args.A_type != "unitary" or args.model != "deq" or args.loss != "gsure":
            print("finite_snapshot is enabled: forcing A_type=unitary, model=deq, loss=gsure.")
        args.A_type = "unitary"
        args.model = "deq"
        args.loss = "gsure"
    elif args.diagonal:
        print("--diagonal is ignored because finite_snapshot is 0.")
        args.diagonal = False

    validate_model_loss(parser, args)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_all_devices(device)

    seed = np.random.randint(1, 100)
    print(f"Seed: {seed}")
    torch.manual_seed(seed)

    for snr in args.snr:
        print(f"\n=== Training {args.model.upper()} | loss={args.loss} | data={args.data} | A={args.A_type} | SNR={snr} dB ===")
        if args.model in {"deq", "dnn"}:
            train_deq_or_dnn(args, snr, device)
        elif args.model == "ldgec":
            train_ldgec(args, snr, device)


if __name__ == "__main__":
    main()
