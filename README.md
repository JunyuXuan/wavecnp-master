# WaveCNP

Research code for WaveCNP experiments with conditional neural processes, attentive neural processes, convolutional CNPs, transformer neural process baselines, and wavelet-based variants for 1D regression and 2D image completion tasks.

The repository contains the model code, experiment scripts, and lightweight helper utilities. Large datasets, generated results, IDE metadata, and experiment outputs are intentionally ignored.

## Repository Layout

```text
wavecnp/             1D CNP/ANP/ConvCNP/WaveCNP models and GP regression data utilities
wavecnp2/            2D image completion models and WaveCNP2 image-completion variants
tnp/                 Transformer Neural Process baseline modules
tetnp/               Translation-equivariant TNP components and data utilities
pytorch_wavelets/    Vendored wavelet transforms used by WaveCNP models
data/image.py        Image-to-task helper utilities used by 2D training scripts
figures/             Figure drawing and visualisation scripts
configs/             Per-method JSON hyperparameter configs
main_1d.py           Unified 1D Gaussian-process regression entry point
main_2d.py           Unified 2D MNIST/CIFAR10/SVHN/CelebA/X-ray entry point
```

## Requirements

This codebase is Python/PyTorch research code. The main dependencies used across the scripts are:

```text
torch
torchvision
numpy
matplotlib
scikit-learn
gpytorch
pywavelets
addict
einops
check-shapes
python-slugify
lab
stheno
tqdm
```

Install the dependencies in your preferred environment, for example:

```bash
pip install torch torchvision numpy matplotlib scikit-learn gpytorch PyWavelets addict einops check-shapes python-slugify lab stheno tqdm
```

Depending on your CUDA/PyTorch setup, install `torch` and `torchvision` using the command recommended by the PyTorch website.

## Data

Datasets are not committed to Git. Place local datasets in the ignored root-level folders expected by the scripts:

```text
data/
MNIST/
```

The ignored folders can contain files such as SVHN `.mat` files, MNIST downloads, X-ray images, and generated experiment outputs without polluting the Git history.

## Running Experiments

Use `train.py` as the unified entry point for datasets and methods. It validates the requested combination, routes to `main_1d.py` or `main_2d.py`, and forwards common options.

List supported options:

```bash
python train.py --list
```

Current supported model names are:

```text
1D regression: wavecnp, convcnp, cnp, anp, tnp, liecnp, tetnp
2D image:      wavecnp2, convcnp2, cnp2, anp2, tnp, liecnp2, tetnp
```

`wavecnp` is the merged 1D WaveCNP implementation with the improved residual wavelet path. `wavecnp2` is the merged 2D WaveCNP implementation with the same high-level idea for image tasks. The older separate `wavecnpnew`, `wavecnpimproved`, and `wavecnp2_improved` entry points are no longer supported.

Example 1D regression experiment:

```bash
python train.py --task regression --data matern --model wavecnp --epochs 200
```

By default, the training scripts use GPU when CUDA is available and fall back to CPU otherwise. Select a device explicitly with `--device`:

```bash
python train.py --task regression --data matern --model wavecnp --device cuda:0
python train.py --task regression --data matern --model wavecnp --device cpu
```

Run every 1D method on the Matern, RBF, Polynomial, and RBF+Linear GP kernels:

```bash
bash scripts/run_1d_all_methods.sh --epochs 200
```

The script uses seeds `1 2 3 4 5` and device `cuda:0` by default. Override the kernel list, seeds, or device with `KERNELS`, `SEEDS`, and `DEVICE`:

```bash
KERNELS="matern eq" SEEDS="1 2 3" DEVICE="cuda:1" bash scripts/run_1d_all_methods.sh --epochs 200
```

The result CSV contains one row per method and one column per kernel. Each kernel cell stores the mean and standard deviation of `test_log_likelihood` across the completed seeds.

Long all-method sweeps are resumable. If a run is interrupted, rerun it with the same `RUN_ROOT`; completed method/kernel/seed combinations are skipped and per-run logs are kept in `_experiments/<RUN_ROOT>/_logs/`:

```bash
EXPERIMENTS_LOG=0 RUN_ROOT=all_1d_gp_kernels_20260607_213023 KERNELS="matern eq polynomial linear" SEEDS="1 2 3 4 5" DEVICE="cuda:0" RESULTS_DIR="rsults" bash scripts/run_1d_all_methods.sh --epochs 20
```

Example 2D image experiment:

```bash
python train.py --task image --data mnist --model wavecnp2 --epochs 200 --batch-size 16
```

Run every supported 2D sweep method on MNIST:

```bash
bash scripts/run_2d_all_methods.sh --epochs 200 --batch-size 16
```

Override datasets, methods, seeds, or device and reuse `RUN_ROOT` to resume:

```bash
DATASETS="mnist cifar10" METHODS="wavecnp2_adapt wavecnp2_noadapt convcnp2 tnp tetnp" SEEDS="1 2 3" DEVICE="cuda:1" \
  bash scripts/run_2d_all_methods.sh --epochs 100 --batch-size 16
```

In the 2D sweep script, `wavecnp2_adapt` and `wavecnp2_noadapt` are runner aliases for `--model wavecnp2 --adapt true` and `--model wavecnp2 --adapt false`.

Validate generated commands without downloading data or starting training:

```bash
DRY_RUN=1 SEEDS="1" bash scripts/run_2d_all_methods.sh --epochs 1
```

The runner refreshes a status CSV and a method-by-dataset aggregate CSV in
`rsults/` after each attempted run.

Run a tiny offline CPU smoke test with generated images:

```bash
DATASETS="fake" METHODS="convcnp2" SEEDS="1" DEVICE="cpu" \
  bash scripts/run_2d_all_methods.sh \
  --epochs 1 --batch-size 2 \
  --num-train-images 2 --num-val-images 2 --num-test-images 2
```

Example TNP or TETNP runs:

```bash
python train.py --task regression --data matern --model tnp
python train.py --task regression --data matern --model tetnp
python train.py --task image --data svhn --model tnp
python train.py --task image --data celeba --model tetnp
```

Common forwarded options include `--root`, `--epochs`, `--learning-rate`, `--weight-decay`, `--seed`, `--device`, `--train`, `--test`, `--task-dwt`, `--adapt`, `--reslevel`, `--batch-size`, `--r-dim`, `--level`, and `--smooth`. Extra arguments are passed through to the selected main script.

Method hyperparameters are stored as one JSON file per method:

```text
configs/1d/<model>.json
configs/2d/<model>.json
configs/2d/datasets/<dataset>.json
```

The main scripts load these automatically. To run with a custom config:

```bash
python train.py --task regression --data matern --model tnp --config configs/1d/tnp.json
```

Use `--dry-run` to inspect the selected command without starting training:

```bash
python train.py --task image --data svhn --model tnp --dry-run
```

The canonical entry points can also be called directly:

```bash
python main_1d.py --data matern --model wavecnp
python main_2d.py --data mnist --model wavecnp2
```

The old 1D `train-table-gpy*.py` scripts have been consolidated into `main_1d.py`, and the old 2D `train_*.py` scripts have been consolidated into `main_2d.py`.

Direct 2D example:

```bash
python main_2d.py --data mnist --model wavecnp2 --epochs 200
```

`main_2d.py` covers fake, MNIST, CIFAR10, SVHN, CelebA, and X-ray tasks with the 2D model stack.

## Notes

- Results and checkpoints are written to local experiment folders and are ignored by Git.
- The `pytorch_wavelets` package is included directly in this repository because the model code imports it locally.
- This repository is currently organized as research code rather than a packaged Python library.
