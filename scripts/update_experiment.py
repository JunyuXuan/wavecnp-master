#!/usr/bin/env python
"""Move an experiment record out of the running table and update its log."""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENTS_PATH = ROOT / "EXPERIMENTS.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--status", required=True, choices=("Completed", "Failed", "Stopped"))
    parser.add_argument("--notes", default="")
    parser.add_argument("--result-summary", default="")
    parser.add_argument("--artifacts", default="")
    parser.add_argument("--file", type=Path, default=DEFAULT_EXPERIMENTS_PATH)
    return parser.parse_args()


def insert_table_row(text: str, section: str, row: str) -> str:
    section_index = text.find(section)
    if section_index == -1:
        raise ValueError(f"Missing section {section!r}")
    separator_index = text.find("|---", section_index)
    if separator_index == -1:
        raise ValueError(f"Missing table in section {section!r}")
    line_end = text.find("\n", separator_index)
    return f"{text[:line_end + 1]}{row}\n{text[line_end + 1:]}"


def markdown_cell(value: object | None) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", r"\|").replace("\n", " ")


def detail_value(text: str, experiment_id: str, label: str) -> str:
    heading = re.search(
        rf"^### {re.escape(experiment_id)}\b.*?(?=^### |\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if not heading:
        return ""
    match = re.search(rf"^- {re.escape(label)}:\s*(.*)$", heading.group(), re.MULTILINE)
    return match.group(1).strip() if match else ""


def update_detail(text: str, experiment_id: str, status: str, notes: str) -> str:
    pattern = rf"(^### {re.escape(experiment_id)}\b.*?)(?=^### |\Z)"
    match = re.search(pattern, text, flags=re.MULTILINE | re.DOTALL)
    if not match:
        raise ValueError(f"Missing detailed entry for {experiment_id}")

    entry = re.sub(
        r"^\*\*Status:\*\* .*$",
        f"**Status:** {status}",
        match.group(),
        count=1,
        flags=re.MULTILINE,
    )
    if notes:
        entry = re.sub(
            r"^- Notes:.*$",
            f"- Notes: {notes}",
            entry,
            count=1,
            flags=re.MULTILINE,
        )
    return f"{text[:match.start()]}{entry}{text[match.end():]}"


def main() -> int:
    args = parse_args()
    text = args.file.read_text(encoding="utf-8")
    experiment_id = args.experiment_id.upper()
    row_pattern = rf"^\| `{re.escape(experiment_id)}` \|.*\|\n?"
    row_match = re.search(row_pattern, text, flags=re.MULTILINE)
    if not row_match:
        raise SystemExit(f"{experiment_id} is not in Running Experiments")

    cells = [cell.strip().strip("`") for cell in row_match.group().strip().strip("|").split("|")]
    if len(cells) != 9:
        raise SystemExit(f"Cannot parse running row for {experiment_id}")
    _, started, server, _device, aim, command, output_path, _status, old_notes = cells

    ended = datetime.now().strftime("%Y-%m-%d %H:%M")
    notes = args.notes or old_notes
    artifacts = args.artifacts or output_path
    text = f"{text[:row_match.start()]}{text[row_match.end():]}"

    if args.status == "Completed":
        model = detail_value(text, experiment_id, "Model")
        task = detail_value(text, experiment_id, "Task")
        data = detail_value(text, experiment_id, "Dataset / kernel")
        task_data = " / ".join(value for value in (task, data) if value)
        row = (
            f"| `{experiment_id}` | `{ended}` | `{server}` | `{markdown_cell(model)}` | "
            f"`{markdown_cell(task_data)}` | `{markdown_cell(aim)}` | `{markdown_cell(command)}` | "
            f"`{markdown_cell(args.result_summary)}` | `{markdown_cell(artifacts)}` | `{markdown_cell(notes)}` |"
        )
        text = insert_table_row(text, "## Completed Experiments", row)
    else:
        row = (
            f"| `{experiment_id}` | `{started}` | `{ended}` | `{server}` | {aim} | "
            f"{args.status} | `{artifacts}` | {notes} |"
        )
        text = insert_table_row(text, "## Incomplete Experiments", row)

    text = update_detail(text, experiment_id, args.status, notes)
    args.file.write_text(text, encoding="utf-8")
    print(f"Updated {experiment_id} to {args.status} in {args.file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
