#!/usr/bin/env python
"""Append experiment run records to EXPERIMENTS.md."""

from __future__ import annotations

import argparse
import re
import socket
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_PATH = ROOT / "EXPERIMENTS.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a launched experiment in EXPERIMENTS.md.")
    parser.add_argument("--aim", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--server", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--status", default="Running")
    parser.add_argument("--notes", default="")
    parser.add_argument("--extra", action="append", default=[])
    return parser.parse_args()


def git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return completed.stdout.strip()


def next_experiment_id(text: str) -> str:
    row_numbers = re.findall(r"^\| `E(\d{3,})` \|", text, flags=re.MULTILINE)
    heading_numbers = re.findall(r"^### E(\d{3,})\b", text, flags=re.MULTILINE)
    numbers = [int(match) for match in [*row_numbers, *heading_numbers]]
    next_number = max(numbers, default=0) + 1
    return f"E{next_number:03d}"


def markdown_cell(value: object | None) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", r"\|").replace("\n", " ")


def insert_running_row(text: str, row: str) -> str:
    marker = "|---|---|---|---|---|---|---|---|---|"
    index = text.find(marker)
    if index == -1:
        return text
    line_end = text.find("\n", index)
    if line_end == -1:
        return f"{text}\n{row}\n"
    return f"{text[:line_end + 1]}{row}\n{text[line_end + 1:]}"


def insert_log_entry(text: str, entry: str) -> str:
    section = "## Experiment Log"
    section_index = text.find(section)
    if section_index == -1:
        return f"{text.rstrip()}\n\n## Experiment Log\n\n{entry}\n"

    first_entry_index = text.find("\n### ", section_index)
    if first_entry_index != -1:
        return f"{text[:first_entry_index + 1]}{entry}\n{text[first_entry_index + 1:]}"

    next_section_index = text.find("\n## ", section_index + len(section))
    if next_section_index == -1:
        return f"{text.rstrip()}\n\n{entry}\n"
    return f"{text[:next_section_index].rstrip()}\n\n{entry}\n{text[next_section_index:]}"


def build_log_entry(args: argparse.Namespace, exp_id: str, started: str, command: str) -> str:
    server = args.server or socket.gethostname()
    output_path = args.output_path or (f"_experiments/{args.root}" if args.root else "")
    model = args.model or ""
    task = args.task or ""
    data = args.data or ""
    commit = git_commit()
    extra_lines = "\n".join(f"- {item}" for item in args.extra)
    if extra_lines:
        extra_lines = f"\n{extra_lines}"

    return f"""### {exp_id} - {args.aim}

**Status:** {args.status}

**Aim**

- {args.aim}

**Server and environment**

- Server alias: {server}
- Hostname: {socket.gethostname()}
- Device(s): {args.device or ""}
- Workspace path: {ROOT}
- Git commit: {commit}

**Experiment setup**

- Model: {model}
- Task: {task}
- Dataset / kernel: {data}
- Experiment root: {args.root or ""}
{extra_lines}

**Command**

```bash
{command}
```

**Outputs**

- Output path: {output_path}

**Results**

- Main metric:
- Secondary metrics:
- Runtime:
- Peak GPU memory:

**Observations**

- Notes: {args.notes}
"""


def main() -> int:
    args = parse_args()
    if not EXPERIMENTS_PATH.exists():
        raise SystemExit(f"Missing {EXPERIMENTS_PATH}")

    text = EXPERIMENTS_PATH.read_text(encoding="utf-8")
    exp_id = next_experiment_id(text)
    started = datetime.now().strftime("%Y-%m-%d %H:%M")
    server = args.server or socket.gethostname()
    command = args.command
    output_path = args.output_path or (f"_experiments/{args.root}" if args.root else "")

    row = (
        f"| `{exp_id}` | `{markdown_cell(started)}` | `{markdown_cell(server)}` | "
        f"`{markdown_cell(args.device)}` | {markdown_cell(args.aim)} | "
        f"`{markdown_cell(command)}` | `{markdown_cell(output_path)}` | "
        f"{markdown_cell(args.status)} | {markdown_cell(args.notes)} |"
    )
    entry = build_log_entry(args, exp_id, started, command)

    text = insert_running_row(text, row)
    text = insert_log_entry(text, entry)
    EXPERIMENTS_PATH.write_text(text, encoding="utf-8")
    print(f"Recorded {exp_id} in {EXPERIMENTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
