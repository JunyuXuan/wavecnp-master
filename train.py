"""Unified experiment launcher for WaveCNP training scripts."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from config_utils import parse_bool


ROOT = Path(__file__).resolve().parent

REGRESSION_DATASETS = (
    "eq",
    "matern",
    "noisy-mixture",
    "weakly-periodic",
    "linear",
    "polynomial",
)
REGRESSION_MODELS = (
    "wavecnp",
    "convcnp",
    "cnp",
    "anp",
    "tnp",
    "liecnp",
    "tetnp",
)

IMAGE_DATASETS = ("fake", "mnist", "cifar10", "svhn", "celeba", "xray")
IMAGE_MODELS = (
    "wavecnp2",
    "convcnp2",
    "cnp2",
    "anp2",
    "tnp",
    "liecnp2",
    "tetnp",
)
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run WaveCNP experiments through one stable command.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=("regression", "image"),
        default="regression",
        help="Experiment family to run.",
    )
    parser.add_argument("--data", help="Dataset name.")
    parser.add_argument("--model", help="Model/method name.")
    parser.add_argument("--root", default=None, help="Experiment output root.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for image tasks.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--device", default=None, help="Runtime device: auto, cpu, cuda, cuda:0, cuda:1, etc.")
    parser.add_argument("--config", default=None, help="Path to a custom method JSON config.")
    parser.add_argument("--tnp-feedforward", type=int, default=None, help="Feed-forward dimension for 1D TNP.")
    parser.add_argument("--tnp-layers", type=int, default=None, help="Number of transformer layers for 1D TNP.")
    parser.add_argument("--num-train-tasks", type=int, default=None, help="Number of generated 1D training tasks.")
    parser.add_argument("--num-val-tasks", type=int, default=None, help="Number of generated 1D validation tasks.")
    parser.add_argument("--num-test-tasks", type=int, default=None, help="Number of generated 1D test tasks.")
    parser.add_argument("--num-train-images", type=int, default=None, help="Number of 2D training images.")
    parser.add_argument("--num-val-images", type=int, default=None, help="Number of 2D validation images.")
    parser.add_argument("--num-test-images", type=int, default=None, help="Number of 2D test images.")
    parser.add_argument("--r-dim", type=int, default=None, help="Representation dimension for image tasks.")
    parser.add_argument("--level", type=int, default=None, help="Wavelet level for image tasks.")
    parser.add_argument("--smooth", type=float, default=None, help="Softmax smoothness for image tasks.")
    parser.add_argument(
        "--reslevel",
        choices=("task", "sample", "NA"),
        default=None,
        help="Resolution level for image tasks.",
    )
    parser.add_argument("--task-dwt", type=parse_bool, default=None, help="Enable task-level DWT.")
    parser.add_argument("--adapt", type=parse_bool, default=None, help="Enable adaptive wavelet transform.")
    parser.add_argument("--train", type=parse_bool, default=None, help="Run training stage.")
    parser.add_argument("--test", type=parse_bool, default=None, help="Run testing stage.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected command without running it.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print supported datasets and models, then exit.",
    )
    return parser


def print_supported() -> None:
    print("Regression datasets:", ", ".join(REGRESSION_DATASETS))
    print("Regression models:  ", ", ".join(REGRESSION_MODELS))
    print("Image datasets:     ", ", ".join(IMAGE_DATASETS))
    print("Image models:       ", ", ".join(IMAGE_MODELS))
def append_option(cmd: list[str], name: str, value: object | None) -> None:
    if value is not None:
        cmd.extend([name, str(value)])


def build_command(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    if not args.data:
        raise ValueError("--data is required unless --list is used.")
    if not args.model:
        raise ValueError("--model is required unless --list is used.")

    if args.task == "regression":
        if args.data not in REGRESSION_DATASETS:
            raise ValueError(f"Unknown regression dataset {args.data!r}.")
        if args.model not in REGRESSION_MODELS:
            raise ValueError(f"Unknown regression model {args.model!r}.")
        script = "main_1d.py"
    else:
        if args.data not in IMAGE_DATASETS:
            raise ValueError(f"Unknown image dataset {args.data!r}.")
        if args.model not in IMAGE_MODELS:
            raise ValueError(f"Unknown image model {args.model!r}.")
        script = "main_2d.py"

    cmd = [sys.executable, str(ROOT / script), "--data", args.data, "--model", args.model]
    append_option(cmd, "--root", args.root)
    append_option(cmd, "--epochs", args.epochs)
    append_option(cmd, "--learning_rate", args.learning_rate)
    append_option(cmd, "--weight_decay", args.weight_decay)
    append_option(cmd, "--seed", args.seed)
    append_option(cmd, "--device", args.device)
    append_option(cmd, "--config", args.config)
    append_option(cmd, "--train", args.train)
    append_option(cmd, "--test", args.test)
    append_option(cmd, "--task_dwt", args.task_dwt)
    append_option(cmd, "--adapt", args.adapt)

    if args.task == "image":
        append_option(cmd, "--batch_size", args.batch_size)
        append_option(cmd, "--r_dim", args.r_dim)
        append_option(cmd, "--level", args.level)
        append_option(cmd, "--smooth", args.smooth)
        append_option(cmd, "--reslevel", args.reslevel)
        append_option(cmd, "--num_training_tasks", args.num_train_images)
        append_option(cmd, "--num_validation_tasks", args.num_val_images)
        append_option(cmd, "--num_test_tasks", args.num_test_images)
    else:
        append_option(cmd, "--tnp_feedforward", args.tnp_feedforward)
        append_option(cmd, "--tnp_layers", args.tnp_layers)
        append_option(cmd, "--num_train_tasks", args.num_train_tasks)
        append_option(cmd, "--num_val_tasks", args.num_val_tasks)
        append_option(cmd, "--num_test_tasks", args.num_test_tasks)

    cmd.extend(passthrough)
    return cmd


def shell_command(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def record_experiment(args: argparse.Namespace) -> None:
    if os.environ.get("EXPERIMENTS_LOG", "1").lower() in {"0", "false", "no", "off"}:
        return

    launcher_command = [Path(sys.executable).name, "train.py", *sys.argv[1:]]
    aim = os.environ.get(
        "EXPERIMENT_AIM",
        f"Run {args.task} {args.data} with {args.model}",
    )
    output_path = f"_experiments/{args.root}" if args.root else ""
    record_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "record_experiment.py"),
        "--aim",
        aim,
        "--command",
        shell_command(launcher_command),
        "--device",
        args.device or "",
        "--model",
        args.model or "",
        "--task",
        args.task or "",
        "--data",
        args.data or "",
    ]
    if args.root:
        record_cmd.extend(["--root", args.root, "--output-path", output_path])
    if os.environ.get("SERVER_ALIAS"):
        record_cmd.extend(["--server", os.environ["SERVER_ALIAS"]])

    completed = subprocess.run(record_cmd, cwd=ROOT)
    if completed.returncode != 0:
        print("Warning: failed to record experiment in EXPERIMENTS.md", file=sys.stderr)


def main() -> int:
    parser = build_parser()
    args, passthrough = parser.parse_known_args()

    if args.list:
        print_supported()
        return 0

    try:
        cmd = build_command(args, passthrough)
    except ValueError as exc:
        parser.error(str(exc))

    print("Selected command:")
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd))
    if args.dry_run:
        return 0

    record_experiment(args)
    completed = subprocess.run(cmd, cwd=ROOT)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
