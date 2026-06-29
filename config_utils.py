import argparse
import json
from pathlib import Path


def parse_bool(value):
    """Parse common command-line boolean spellings."""
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def _json_safe(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(val) for val in value]
    return value


def resolve_method_config_path(config_dir, model_name, config_path=None):
    repo_root = Path(__file__).resolve().parent
    return Path(config_path) if config_path else repo_root / config_dir / f"{model_name}.json"


def load_method_config(config_dir, model_name, config_path=None):
    """Load the per-method JSON config for a training entry point."""
    path = resolve_method_config_path(config_dir, model_name, config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset_config(config_dir, dataset_name):
    """Load dataset-specific model hyperparameters."""
    repo_root = Path(__file__).resolve().parent
    path = repo_root / config_dir / f"{dataset_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_run_summary(path, summary):
    """Save a compact machine-readable and human-readable run summary."""
    path = Path(path)
    summary = _json_safe(summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    text_path = path.with_suffix(".txt")
    with text_path.open("w", encoding="utf-8") as f:
        f.write(f"method_name: {summary['method_name']}\n")
        f.write(f"config: {summary['config_path']}\n")
        f.write("parameters:\n")
        for key, value in summary["parameters"].items():
            f.write(f"  {key}: {value}\n")
        f.write("results:\n")
        for key, value in summary["results"].items():
            f.write(f"  {key}: {value}\n")
