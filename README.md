# TSP DEQ-GSURE

This repository contains PyTorch code for selected experiments in:

**Unsupervised Deep Equilibrium Model Learning for Large-Scale Channel Estimation with Performance Guarantees**

The release supports DEQ-GSURE, DEQ-NMSE, DNN, LDGEC, and classic OMP/AMP/OAMP baselines for large-scale channel estimation. The provided scripts cover DeepMIMO O1, DeepMIMO O2 generalization, and synthetic sparse-channel experiments, including the results used for Fig. 4, Fig. 7-Fig. 11, Table 3, and Table 4.

## Environment

The code was run on the following local environment:

- OS: Windows 10
- GPU: 4 x NVIDIA GeForce RTX 3090, 24 GB each
- NVIDIA driver: 551.78
- Python: 3.10.6
- PyTorch: 2.1.2+cu121
- CUDA: 12.1
- DeepMIMO: required only when regenerating DeepMIMO O1 channel files

Install a PyTorch build matching your CUDA/CPU environment, then install the remaining packages:

```bash
pip install numpy scipy matplotlib tqdm prettytable
```

## Quick Start

We provide the following minimal workflow to reproduce DEQ-GSURE training on DeepMIMO O1.

First, download the DeepMIMO `O1_28` scenario from the official DeepMIMO O1_28 scenario page:

```text
https://www.deepmimo.net/scenarios/v4/o1_28
```

Place the extracted scenario files under:

```text
data/deepmimo/O1_28/
```

The directory should contain files such as `O1_28.params.mat`, `O1_28.1.CIR.mat`, and `O1_28.1.DoA.mat`. Then generate the DeepMIMO O1 channels and measurements, and run the default training command:

```bash
python data/deepmimo/scenario_o1/generate_channels.py
python data/deepmimo/scenario_o1/generate_measurements.py
python main.py
```

By default, `main.py` trains DEQ-GSURE on DeepMIMO O1 with unitary sensing in the SNR order `20, 15, 10, 5, 0` dB.

## Repository Structure

```text
main.py              Training entry point
test.py              Evaluation entry point
plot.py              Figure and result aggregation
trainer.py           DEQ/DNN training routines
model/               DEQ, DNN, LDGEC, and classic baselines
utils/               Data loading, metrics, checkpoints, evaluation
data/                Sensing matrices, DeepMIMO data, synthetic data
trained network/     Saved checkpoints
results/             Saved numerical results
figures/             Saved figures
```

## Data

Released SNR values are `[0, 5, 10, 15, 20]` dB. DeepMIMO O1 includes Gaussian, Bernoulli, and unitary measurements; DeepMIMO O2 and synthetic experiments use unitary measurements.

The raw DeepMIMO `O1_28` scenario is not included in this repository due to its file size. Please download it from the official DeepMIMO O1_28 scenario page and place it at `data/deepmimo/O1_28/` before generating O1 channels.

To regenerate data, run:

```bash
python data/sensing_matrix/generate_A.py
python data/deepmimo/scenario_o1/generate_channels.py
python data/deepmimo/scenario_o1/generate_measurements.py
python data/deepmimo/scenario_o2/generate_measurements.py
python data/synthetic/generate_channels.py
python data/synthetic/generate_measurements.py
```

`data/deepmimo/scenario_o2/generate_measurements.py` assumes that O2 channel files already exist in `data/deepmimo/scenario_o2/channels`.

## Training

Default training runs DEQ-GSURE on DeepMIMO O1 with unitary sensing for SNR values `20, 15, 10, 5, 0` dB:

```bash
python main.py
```

Common examples:

```bash
python main.py --model deq --loss gsure --A_type unitary --snr 0 5 10 15 20
python main.py --model dnn --loss nmse --A_type unitary --snr 20
python main.py --model ldgec --loss sure --A_type unitary --snr 20
python main.py --data synthetic --model deq --loss gsure --A_type unitary --snr 20
python main.py --finite_snapshot 0.001 --diagonal --snr 20
```

Supported training losses are `gsure`/`nmse` for DEQ and DNN, and `sure`/`nmse` for LDGEC. Checkpoints are saved under `trained network/<A_type>/`. For LDGEC, `--epochs` means epochs per layer, and the final NMSE is sensitive to hyperparameters such as `beta`, epoch count, and step size.

## Evaluation

Evaluate one trained configuration over all released SNR values:

```bash
python test.py --model deq --loss gsure --data deepmimo_o1 --A_type unitary
python test.py --model ldgec --loss sure --data deepmimo_o1 --A_type unitary
python test.py --model deq --loss gsure --data deepmimo_o2 --A_type unitary
python test.py --data synthetic --model deq --loss gsure --A_type unitary
python test.py --finite_snapshot 0.001 0.002 --diagonal
```

Classic baselines can be run directly:

```bash
python model/classic_algorithms.py --data deepmimo_o1 --alg omp amp oamp
python model/classic_algorithms.py --data deepmimo_o2 --alg omp amp oamp
python model/classic_algorithms.py --data synthetic --alg omp amp oamp
```

## Plotting

```bash
python plot.py --scenario_o1
python plot.py --scenario_o2
python plot.py --nFEs
python plot.py --beta_omega
python plot.py --nmse_vs_A_type
python plot.py --nmse_vs_sigma_e
```

Figures are saved in `figures/`, and numerical plotting results are saved in `results/`.

## Acknowledgements

We gratefully acknowledge the authors of the following work for making their code publicly available. Parts of this repository were implemented with reference to their released code.

[7] W. Yu, Y. Shen, H. He, X. Yu, S. Song, J. Zhang, and K. B. Letaief, "An adaptive and robust deep learning framework for THz ultra-massive MIMO channel estimation," IEEE Journal of Selected Topics in Signal Processing, vol. 17, no. 4, pp. 761-776, 2023.
