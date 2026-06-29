#!/usr/bin/env python
"""Aggregate 2D experiment summaries into a compact test likelihood CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path


DATASET_LABELS = {
    "fake": "FakeData",
    "mnist": "MNIST",
    "cifar10": "CIFAR10",
    "svhn": "SVHN",
    "celeba": "CelebA",
    "xray": "X-ray",
}
DATASET_ORDER = ("fake", "mnist", "cifar10", "svhn", "celeba", "xray")
METHOD_ORDER = (
    "wavecnp2_adapt",
    "wavecnp2_noadapt",
    "convcnp2",
    "cnp2",
    "anp2",
    "tnp",
    "liecnp2",
    "tetnp",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect 2D test_log_likelihood means/stds into one method-by-dataset CSV."
    )
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--dataset", default=None, help="Alias for --datasets with one value.")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", default=None)
    parser.add_argument("--output-dir", default="rsults")
    parser.add_argument("--allow-empty", action="store_true")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "results"


def numeric(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


def population_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def selected_datasets(args: argparse.Namespace) -> list[str]:
    if args.datasets:
        return args.datasets
    if args.dataset:
        return [args.dataset]
    return list(DATASET_ORDER)


def method_sort_key(method: str) -> tuple[int, str]:
    try:
        return (METHOD_ORDER.index(method), method)
    except ValueError:
        return (len(METHOD_ORDER), method)


def display_method(method: str, params: dict[str, object]) -> str:
    if method == "wavecnp2":
        if params.get("adapt") is True:
            return "wavecnp2_adapt"
        if params.get("adapt") is False:
            return "wavecnp2_noadapt"
    return method


def format_progress(values: list[float], expected_count: int | None) -> str:
    if not values:
        return "pending (0/{})".format(expected_count) if expected_count else ""

    mean = sum(values) / len(values)
    std = population_std(values)
    summary = f"{mean:.6g} +/- {std:.6g}"
    if expected_count and len(values) < expected_count:
        return f"{summary} ({len(values)}/{expected_count})"
    return summary


def parse_test_log_likelihood(log_path: Path) -> float | None:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    matches = re.findall(
        r"Model averages a log-likelihood of\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
        text,
    )
    if not matches:
        return None
    return numeric(float(matches[-1]))


def load_runs(
    experiment_root: str,
    datasets: set[str],
    methods: set[str] | None = None,
    seeds: set[str] | None = None,
) -> list[dict[str, object]]:
    root = Path("_experiments").joinpath(experiment_root)
    summary_paths = sorted(root.glob("*/summary.json"))
    rows_by_key: dict[tuple[str, str, str], dict[str, object]] = {}

    for path in summary_paths:
        try:
            with path.open("r", encoding="utf-8-sig") as f:
                summary = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        params = summary.get("parameters", {})
        if params.get("task") not in (None, "image"):
            continue

        dataset = str(params.get("data", ""))
        if dataset not in datasets:
            continue

        method = display_method(str(summary.get("method_name", "")), params)
        if methods and method not in methods:
            continue

        seed = str(params.get("seed", ""))
        if seeds and seed not in seeds:
            continue

        test_log_likelihood = numeric(summary.get("results", {}).get("test_log_likelihood"))
        if test_log_likelihood is None:
            continue

        rows_by_key[(method, dataset, seed)] = {
            "method": method,
            "dataset": dataset,
            "seed": seed,
            "test_log_likelihood": test_log_likelihood,
        }

    for path in sorted(root.glob("_logs/*.log")):
        stem_parts = path.stem.split("_", 2)
        if len(stem_parts) != 3:
            continue

        dataset, seed, method = stem_parts
        if dataset not in datasets:
            continue
        if methods and method not in methods:
            continue
        if seeds and seed not in seeds:
            continue

        test_log_likelihood = parse_test_log_likelihood(path)
        if test_log_likelihood is None:
            continue

        rows_by_key[(method, dataset, seed)] = {
            "method": method,
            "dataset": dataset,
            "seed": seed,
            "test_log_likelihood": test_log_likelihood,
        }

    return list(rows_by_key.values())


def ordered_datasets(datasets: list[str]) -> list[str]:
    ordered = [dataset for dataset in DATASET_ORDER if dataset in datasets]
    ordered.extend(dataset for dataset in datasets if dataset not in ordered)
    return ordered


def build_table(
    rows: list[dict[str, object]],
    datasets: list[str],
    methods: list[str] | None,
    expected_count: int | None,
) -> list[dict[str, str]]:
    selected_methods = methods or sorted({str(row["method"]) for row in rows}, key=method_sort_key)
    table: list[dict[str, str]] = []

    for method in sorted(selected_methods, key=method_sort_key):
        output_row = {"method": method}
        for dataset in ordered_datasets(datasets):
            values = [
                float(row["test_log_likelihood"])
                for row in rows
                if row["method"] == method and row["dataset"] == dataset
            ]
            output_row[DATASET_LABELS.get(dataset, dataset)] = format_progress(
                values, expected_count
            )
        table.append(output_row)

    return table


def main() -> int:
    args = parse_args()
    datasets = selected_datasets(args)
    unknown_datasets = [dataset for dataset in datasets if dataset not in DATASET_LABELS]
    if unknown_datasets:
        known = ", ".join(DATASET_ORDER)
        raise SystemExit(
            f"Unknown 2D dataset(s): {', '.join(unknown_datasets)}. "
            f"Supported datasets: {known}"
        )

    rows = load_runs(
        args.experiment_root,
        set(datasets),
        set(args.methods) if args.methods else None,
        set(args.seeds) if args.seeds else None,
    )
    if not rows and not args.allow_empty:
        raise SystemExit(
            f"No test_log_likelihood values found under _experiments/{args.experiment_root}"
        )

    expected_count = len(args.seeds) if args.seeds else None
    table = build_table(rows, datasets, args.methods, expected_count)
    fieldnames = [
        "method",
        *(DATASET_LABELS.get(dataset, dataset) for dataset in ordered_datasets(datasets)),
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"2d_all_methods_image_datasets_{safe_name(args.experiment_root)}.csv"

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table)

    print(f"Wrote {len(table)} method summaries from {len(rows)} runs to {output_path}")
    print(f"Generated at {datetime.now().isoformat(timespec='seconds')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
