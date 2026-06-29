#!/usr/bin/env python
"""Diagnose missing or failed jobs in a 1D multi-method experiment root."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


DEFAULT_DATASETS = ("matern", "eq", "polynomial", "linear")
DEFAULT_SEEDS = ("1", "2", "3", "4", "5")
DEFAULT_METHODS = (
    "convcnp",
    "cnp",
    "anp",
    "tnp",
    "liecnp",
    "tetnp",
    "wavecnp",
)
ERROR_MARKERS = ("Traceback", "Error", "Exception", "RuntimeError", "AssertionError", "CUDA")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report completed, missing, and failed jobs for a 1D experiment root."
    )
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS))
    parser.add_argument("--seeds", nargs="*", default=list(DEFAULT_SEEDS))
    parser.add_argument("--methods", nargs="*", default=list(DEFAULT_METHODS))
    parser.add_argument("--tail-lines", type=int, default=20)
    return parser.parse_args()


def numeric(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


def load_completed(root: Path) -> set[tuple[str, str, str]]:
    completed: set[tuple[str, str, str]] = set()
    for path in root.glob("*/summary.json"):
        try:
            with path.open("r", encoding="utf-8-sig") as f:
                summary = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        params = summary.get("parameters", {})
        value = summary.get("results", {}).get("test_log_likelihood")
        if numeric(value) is None:
            continue

        completed.add(
            (
                str(params.get("data", "")),
                str(params.get("seed", "")),
                str(summary.get("method_name", "")),
            )
        )
    return completed


def read_log_clues(path: Path, tail_lines: int) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return [f"Could not read log: {exc}"]

    clues = [line for line in lines if any(marker in line for marker in ERROR_MARKERS)]
    if clues:
        return clues[-tail_lines:]
    return lines[-tail_lines:]


def main() -> int:
    args = parse_args()
    root = Path("_experiments") / args.experiment_root
    log_dir = root / "_logs"

    if not root.exists():
        raise SystemExit(f"Missing experiment root: {root}")

    expected = [
        (dataset, str(seed), method)
        for dataset in args.datasets
        for seed in args.seeds
        for method in args.methods
    ]
    completed = load_completed(root)
    incomplete = [job for job in expected if job not in completed]
    completed_expected = len(expected) - len(incomplete)

    print(f"Experiment root: {root}")
    print(f"Completed expected summaries: {completed_expected} / {len(expected)}")

    if not incomplete:
        print("All expected jobs have test_log_likelihood summaries.")
        return 0

    print("\nIncomplete jobs:")
    for dataset, seed, method in incomplete:
        log_path = log_dir / f"{dataset}_{seed}_{method}.log"
        status = "log exists" if log_path.exists() else "log missing"
        print(f"- {dataset} seed={seed} method={method}: {status}")

    print("\nLog clues:")
    for dataset, seed, method in incomplete:
        log_path = log_dir / f"{dataset}_{seed}_{method}.log"
        if not log_path.exists():
            continue
        print(f"\n[{log_path}]")
        for line in read_log_clues(log_path, args.tail_lines):
            print(line)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
