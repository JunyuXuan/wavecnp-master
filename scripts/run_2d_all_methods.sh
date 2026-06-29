#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Run supported 2D methods on selected image datasets for multiple random seeds.

Usage:
  bash scripts/run_2d_all_methods.sh [extra train.py args...]

Examples:
  bash scripts/run_2d_all_methods.sh --epochs 200 --batch-size 16
  DATASETS="mnist cifar10" SEEDS="1 2 3" bash scripts/run_2d_all_methods.sh --epochs 50
  METHODS="convcnp2 wavecnp2_adapt wavecnp2_noadapt tnp" DEVICE="cuda:1" bash scripts/run_2d_all_methods.sh --epochs 100
  RUN_ROOT=all_2d_methods_20260612_120000 bash scripts/run_2d_all_methods.sh --epochs 20
  DRY_RUN=1 bash scripts/run_2d_all_methods.sh --epochs 1

Defaults:
  methods:  wavecnp2_adapt wavecnp2_noadapt convcnp2 cnp2 anp2 tnp liecnp2 tetnp
  datasets: mnist
  seeds:    1 2 3 4 5
  device:   cuda:0
  results:  rsults/

Environment:
  METHODS      Space-separated method list.
  DATASETS     Space-separated dataset list: fake mnist cifar10 svhn celeba xray.
  SEEDS        Space-separated random seeds.
  DEVICE       Runtime device accepted by train.py.
  RUN_ROOT     Reuse an experiment root to resume completed runs.
  RESULTS_DIR  Directory for aggregate and status CSV files.
  PYTHON       Python executable.
  DRY_RUN      Set to 1/true/yes/on to print and validate commands only.
  EXPERIMENTS_LOG
               Set to 0/false/no/off to disable EXPERIMENTS.md tracking.

Notes:
  MNIST, CIFAR10, SVHN, and CelebA may be downloaded by torchvision.
  X-ray data must already exist under data/xray/.
USAGE
  exit 0
fi

DEFAULT_METHODS=(
  wavecnp2_adapt
  wavecnp2_noadapt
  convcnp2
  cnp2
  anp2
  tnp
  liecnp2
  tetnp
)
SUPPORTED_METHODS=(
  "${DEFAULT_METHODS[@]}"
)
SUPPORTED_DATASETS=(fake mnist cifar10 svhn celeba xray)

read -r -a METHODS <<< "${METHODS:-${DEFAULT_METHODS[*]}}"
read -r -a DATASET_LIST <<< "${DATASETS:-mnist}"
read -r -a SEED_LIST <<< "${SEEDS:-1 2 3 4 5}"
DEVICE="${DEVICE:-cuda:0}"
RESULTS_DIR="${RESULTS_DIR:-rsults}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON_BIN="${PYTHON:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

contains() {
  local candidate="$1"
  shift
  local value
  for value in "$@"; do
    if [[ "$candidate" == "$value" ]]; then
      return 0
    fi
  done
  return 1
}

is_true() {
  case "$1" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

quote_words() {
  local quoted=""
  local word
  for word in "$@"; do
    printf -v word "%q" "$word"
    quoted+="${word} "
  done
  printf "%s" "${quoted% }"
}

method_model() {
  case "$1" in
    wavecnp2_adapt|wavecnp2_noadapt) printf "wavecnp2" ;;
    *) printf "%s" "$1" ;;
  esac
}

method_adapt() {
  case "$1" in
    wavecnp2_adapt) printf "true" ;;
    wavecnp2_noadapt) printf "false" ;;
  esac
}

if [[ "${1:-}" != "" && "${1:-}" != --* ]]; then
  if contains "$1" "${SUPPORTED_DATASETS[@]}"; then
    DATASET_LIST=("$1")
    shift
  else
    echo "Unknown 2D dataset '$1'. Supported datasets: ${SUPPORTED_DATASETS[*]}" >&2
    exit 2
  fi
fi

for dataset in "${DATASET_LIST[@]}"; do
  if ! contains "$dataset" "${SUPPORTED_DATASETS[@]}"; then
    echo "Unknown 2D dataset '$dataset'. Supported datasets: ${SUPPORTED_DATASETS[*]}" >&2
    exit 2
  fi
done

for method in "${METHODS[@]}"; do
  if ! contains "$method" "${SUPPORTED_METHODS[@]}"; then
    echo "Unknown or unsupported 2D method '$method'. Supported methods: ${SUPPORTED_METHODS[*]}" >&2
    exit 2
  fi
done

RUN_ROOT="${RUN_ROOT:-}"
if [[ -z "$RUN_ROOT" ]]; then
  RUN_ROOT="all_2d_methods_$(date +%Y%m%d_%H%M%S)"
fi

EXTRA_ARGS=("$@")
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
  case "${EXTRA_ARGS[$i]}" in
    --root)
      if ((i + 1 < ${#EXTRA_ARGS[@]})); then
        RUN_ROOT="${EXTRA_ARGS[$((i + 1))]}"
      fi
      ;;
    --root=*)
      RUN_ROOT="${EXTRA_ARGS[$i]#--root=}"
      ;;
  esac
done

run_completed() {
  local dataset="$1"
  local seed="$2"
  local method="$3"
  local model="$4"
  local adapt="$5"
  "$PYTHON_BIN" - "$RUN_ROOT" "$dataset" "$seed" "$method" "$model" "$adapt" <<'PY'
import json
import math
import sys
from pathlib import Path

root, dataset, seed, method, model, adapt = sys.argv[1:]
expected_adapt = None if adapt == "" else adapt.lower() == "true"

for path in Path("_experiments").joinpath(root).glob("*/summary.json"):
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            summary = json.load(f)
    except (OSError, json.JSONDecodeError):
        continue

    params = summary.get("parameters", {})
    value = summary.get("results", {}).get("test_log_likelihood")
    has_result = (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
    if (
        summary.get("method_name") == model
        and str(params.get("data")) == dataset
        and str(params.get("seed")) == seed
        and (expected_adapt is None or params.get("adapt") is expected_adapt)
        and has_result
    ):
        sys.exit(0)

sys.exit(1)
PY
}

csv_field() {
  local value="${1//\"/\"\"}"
  printf '"%s"' "$value"
}

append_status_row() {
  local dataset="$1"
  local method="$2"
  local seed="$3"
  local status="$4"
  local exit_code="$5"
  local timestamp
  timestamp="$(date +%Y-%m-%dT%H:%M:%S)"

  {
    csv_field "$timestamp"; printf ","
    csv_field "$RUN_ROOT"; printf ","
    csv_field "$dataset"; printf ","
    csv_field "$method"; printf ","
    csv_field "$seed"; printf ","
    csv_field "$DEVICE"; printf ","
    csv_field "$status"; printf ","
    csv_field "$exit_code"; printf "\n"
  } >> "$STATUS_CSV"
}

aggregate_progress() {
  "$PYTHON_BIN" scripts/aggregate_2d_results.py \
    --experiment-root "$RUN_ROOT" \
    --datasets "${DATASET_LIST[@]}" \
    --methods "${METHODS[@]}" \
    --seeds "${SEED_LIST[@]}" \
    --output-dir "$RESULTS_DIR" \
    --allow-empty
}

LOG_DIR="_experiments/$RUN_ROOT/_logs"
STATUS_CSV="$RESULTS_DIR/2d_all_methods_${RUN_ROOT}_run_status.csv"
if ! is_true "$DRY_RUN"; then
  mkdir -p "$LOG_DIR" "$RESULTS_DIR"
  if [[ ! -f "$STATUS_CSV" ]]; then
    printf 'timestamp,experiment_root,dataset,method,seed,device,status,exit_code\n' > "$STATUS_CSV"
  fi
  aggregate_progress
fi

LAUNCH_COMMAND="RUN_ROOT=\"$RUN_ROOT\" METHODS=\"${METHODS[*]}\" DATASETS=\"${DATASET_LIST[*]}\" SEEDS=\"${SEED_LIST[*]}\" DEVICE=\"${DEVICE}\" RESULTS_DIR=\"${RESULTS_DIR}\" bash scripts/run_2d_all_methods.sh"
if [[ "$#" -gt 0 ]]; then
  LAUNCH_COMMAND+=" $(quote_words "$@")"
fi

EXPERIMENT_ID=""
EXPERIMENT_FINALIZED=0
if ! is_true "$DRY_RUN" && ! is_true "${EXPERIMENTS_LOG:-1}"; then
  :
elif ! is_true "$DRY_RUN"; then
  record_output="$("$PYTHON_BIN" scripts/record_experiment.py \
    --aim "${EXPERIMENT_AIM:-Run all 2D methods on selected image datasets}" \
    --command "$LAUNCH_COMMAND" \
    --server "${SERVER_ALIAS:-$(hostname)}" \
    --device "$DEVICE" \
    --model "${METHODS[*]}" \
    --task "image" \
    --data "${DATASET_LIST[*]}" \
    --root "$RUN_ROOT" \
    --output-path "_experiments/$RUN_ROOT; aggregate CSV in $RESULTS_DIR/" \
    --extra "Datasets: ${DATASET_LIST[*]}" \
    --extra "Seeds: ${SEED_LIST[*]}" \
    --extra "Methods: ${METHODS[*]}" \
    --extra "Results directory: $RESULTS_DIR")"
  echo "$record_output"
  EXPERIMENT_ID="$(sed -n 's/^Recorded \(E[0-9][0-9]*\) .*/\1/p' <<< "$record_output")"
fi

finalize_experiment() {
  local status="$1"
  local notes="$2"
  if [[ -n "$EXPERIMENT_ID" && "$EXPERIMENT_FINALIZED" -eq 0 ]]; then
    "$PYTHON_BIN" scripts/update_experiment.py \
      --experiment-id "$EXPERIMENT_ID" \
      --status "$status" \
      --notes "$notes" \
      --artifacts "_experiments/$RUN_ROOT; $RESULTS_DIR/2d_all_methods_image_datasets_${RUN_ROOT}.csv"
    EXPERIMENT_FINALIZED=1
  fi
}

finalize_on_exit() {
  local exit_code="$?"
  if [[ "$exit_code" -ne 0 && "$EXPERIMENT_FINALIZED" -eq 0 ]]; then
    if [[ "$exit_code" -eq 130 || "$exit_code" -eq 143 ]]; then
      finalize_experiment "Stopped" "Runner was interrupted; inspect the status CSV and logs before resuming."
    else
      finalize_experiment "Failed" "Runner exited with code $exit_code; inspect the status CSV and logs."
    fi
  fi
}

finalize_on_signal() {
  local exit_code="$1"
  finalize_experiment "Stopped" "Runner was interrupted; inspect the status CSV and logs before resuming."
  trap - EXIT INT TERM
  exit "$exit_code"
}

trap finalize_on_exit EXIT
trap 'finalize_on_signal 130' INT
trap 'finalize_on_signal 143' TERM

FAILED_RUNS=0
for dataset in "${DATASET_LIST[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    for method in "${METHODS[@]}"; do
      model="$(method_model "$method")"
      adapt="$(method_adapt "$method")"
      log_file="${LOG_DIR}/${dataset}_${seed}_${method}.log"
      if ! is_true "$DRY_RUN" && run_completed "$dataset" "$seed" "$method" "$model" "$adapt"; then
        echo "Skipping completed 2D dataset=${dataset} method=${method} seed=${seed} root=${RUN_ROOT}"
        continue
      fi

      if is_true "$DRY_RUN"; then
        echo "Validating 2D dataset=${dataset} method=${method} seed=${seed} device=${DEVICE} root=${RUN_ROOT}"
      else
        echo "Running 2D dataset=${dataset} method=${method} seed=${seed} device=${DEVICE} root=${RUN_ROOT} log=${log_file}"
      fi

      command=(
        "$PYTHON_BIN" train.py
        --task image
        --data "$dataset"
        --model "$model"
        --root "$RUN_ROOT"
        --seed "$seed"
        --device "$DEVICE"
      )
      if [[ -n "$adapt" ]]; then
        command+=(--adapt "$adapt")
      fi
      if is_true "$DRY_RUN"; then
        command+=(--dry-run)
      fi
      command+=("$@")

      if is_true "$DRY_RUN"; then
        if ! "${command[@]}"; then
          FAILED_RUNS=$((FAILED_RUNS + 1))
        fi
      elif EXPERIMENTS_LOG=0 "${command[@]}" > "$log_file" 2>&1; then
        echo "Completed 2D dataset=${dataset} method=${method} seed=${seed}"
        append_status_row "$dataset" "$method" "$seed" "completed" "0"
      else
        exit_code="$?"
        FAILED_RUNS=$((FAILED_RUNS + 1))
        append_status_row "$dataset" "$method" "$seed" "failed" "$exit_code"
        echo "Failed 2D dataset=${dataset} method=${method} seed=${seed}; see ${log_file}" >&2
        tail -n 40 "$log_file" >&2 || true
      fi

      if ! is_true "$DRY_RUN"; then
        aggregate_progress
      fi
    done
  done
done

if [[ "$FAILED_RUNS" -gt 0 ]]; then
  if is_true "$DRY_RUN"; then
    echo "$FAILED_RUNS dry-run command(s) failed." >&2
  else
    echo "$FAILED_RUNS run(s) failed. See $STATUS_CSV for details." >&2
  fi
  exit 1
fi

if is_true "$DRY_RUN"; then
  echo "Dry run completed; no training was started."
else
  aggregate_progress
  finalize_experiment \
    "Completed" \
    "All requested method, dataset, and seed combinations completed successfully."
fi
