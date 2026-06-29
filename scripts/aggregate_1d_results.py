#!/usr/bin/env python
"""Aggregate 1D experiment summaries into a compact test likelihood CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect 1D test_log_likelihood means/stds into one method-by-kernel CSV."
    )
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--dataset", default=None, help="Deprecated alias for --datasets with one value.")
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


KERNEL_LABELS = {
    "matern": "Matern",
    "eq": "RBF",
    "polynomial": "Polynomial",
    "linear": "RBF+Linear",
}
KERNEL_ORDER = ("matern", "eq", "polynomial", "linear")
METHOD_ORDER = (
    "convcnp",
    "cnp",
    "anp",
    "tnp",
    "liecnp",
    "tetnp",
    "wavecnp",
)


def selected_datasets(args: argparse.Namespace) -> list[str]:
    if args.datasets:
        return args.datasets
    if args.dataset:
        return [args.dataset]
    return list(KERNEL_ORDER)


def method_sort_key(method: str) -> tuple[int, str]:
    try:
        return (METHOD_ORDER.index(method), method)
    except ValueError:
        return (len(METHOD_ORDER), method)


def format_mean_std(values: list[float]) -> str:
    if not values:
        return ""
    mean = sum(values) / len(values)
    std = population_std(values)
    return f"{mean:.6g} +/- {std:.6g}"


def format_progress(values: list[float], expected_count: int | None) -> str:
    if not values:
        return "pending (0/{})".format(expected_count) if expected_count else ""

    summary = format_mean_std(values)
    if expected_count and len(values) < expected_count:
        return f"{summary} ({len(values)}/{expected_count})"
    return summary


def load_runs(experiment_root: str, datasets: set[str]) -> list[dict[str, object]]:
    summary_paths = sorted(Path("_experiments").joinpath(experiment_root).glob("*/summary.json"))
    rows: list[dict[str, object]] = []

    for path in summary_paths:
        with path.open("r", encoding="utf-8-sig") as f:
            summary = json.load(f)

        params = summary.get("parameters", {})
        results = summary.get("results", {})
        dataset = str(params.get("data", ""))
        if dataset not in datasets:
            continue

        test_log_likelihood = numeric(results.get("test_log_likelihood"))
        if test_log_likelihood is None:
            continue

        rows.append(
            {
                "method": summary.get("method_name", ""),
                "dataset": dataset,
                "seed": params.get("seed", ""),
                "test_log_likelihood": test_log_likelihood,
            }
        )

    return rows


def build_table(
    rows: list[dict[str, object]],
    datasets: list[str],
    methods: list[str] | None,
    expected_count: int | None,
) -> list[dict[str, str]]:
    selected_methods = methods or sorted({str(row["method"]) for row in rows}, key=method_sort_key)
    field_datasets = [dataset for dataset in KERNEL_ORDER if dataset in datasets]
    field_datasets.extend(dataset for dataset in datasets if dataset not in field_datasets)

    table: list[dict[str, str]] = []
    for method in sorted(selected_methods, key=method_sort_key):
        output_row = {"method": method}
        for dataset in field_datasets:
            values = [
                float(row["test_log_likelihood"])
                for row in rows
                if row["method"] == method and row["dataset"] == dataset
            ]
            output_row[KERNEL_LABELS.get(dataset, dataset)] = format_progress(values, expected_count)
        table.append(output_row)

    return table


def main() -> int:
    args = parse_args()
    datasets = selected_datasets(args)
    unknown_datasets = [dataset for dataset in datasets if dataset not in KERNEL_LABELS]
    if unknown_datasets:
        known = ", ".join(KERNEL_ORDER)
        raise SystemExit(f"Unknown 1D GP kernel(s): {', '.join(unknown_datasets)}. Supported kernels: {known}")

    rows = load_runs(args.experiment_root, set(datasets))
    if not rows and not args.allow_empty:
        raise SystemExit(
            f"No test_log_likelihood values found under _experiments/{args.experiment_root}"
        )

    expected_count = len(args.seeds) if args.seeds else None
    table = build_table(rows, datasets, args.methods, expected_count)
    field_datasets = [dataset for dataset in KERNEL_ORDER if dataset in datasets]
    field_datasets.extend(dataset for dataset in datasets if dataset not in field_datasets)
    fieldnames = ["method", *(KERNEL_LABELS.get(dataset, dataset) for dataset in field_datasets)]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"1d_all_methods_gp_kernels_{safe_name(args.experiment_root)}.csv"

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table)

    print(f"Wrote {len(table)} method summaries from {len(rows)} runs to {output_path}")
    print(f"Generated at {datetime.now().isoformat(timespec='seconds')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
