#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Run all 1D methods on the selected 1D GP kernels for multiple random seeds.

Usage:
  bash scripts/run_1d_all_methods.sh [extra train.py args...]

Examples:
  bash scripts/run_1d_all_methods.sh --epochs 200
  KERNELS="matern eq" bash scripts/run_1d_all_methods.sh --epochs 50 --root table
  DEVICE="cuda:1" bash scripts/run_1d_all_methods.sh --epochs 200
  EXPERIMENTS_LOG=0 RUN_ROOT=all_1d_gp_kernels_20260607_213023 bash scripts/run_1d_all_methods.sh --epochs 20
  METHODS="convcnp wavecnp" KERNELS="matern" SEEDS="1" bash scripts/run_1d_all_methods.sh --epochs 1

Defaults:
  methods: convcnp cnp anp tnp liecnp tetnp wavecnp
  kernels: matern eq polynomial linear
           (Matern, RBF, Polynomial, RBF+Linear)
  seeds:   1 2 3 4 5
  device:  cuda:0
  results: rsults/

Progress files:
  The aggregate CSV is created when the script starts and refreshed after each
  method/seed run. A per-run status CSV is also written to help diagnose failed
  method/kernel/seed combinations.
USAGE
  exit 0
fi

DEFAULT_METHODS=(
  convcnp
  cnp
  anp
  tnp
  liecnp
  tetnp
  wavecnp
)
read -r -a METHODS <<< "${METHODS:-${DEFAULT_METHODS[*]}}"

SUPPORTED_KERNELS=(matern eq polynomial linear)
read -r -a DATASET_LIST <<< "${KERNELS:-matern eq polynomial linear}"
read -r -a SEED_LIST <<< "${SEEDS:-1 2 3 4 5}"
DEVICE="${DEVICE:-cuda:0}"
RESULTS_DIR="${RESULTS_DIR:-rsults}"
PYTHON_BIN="${PYTHON:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

is_supported_kernel() {
  local candidate="$1"
  local kernel
  for kernel in "${SUPPORTED_KERNELS[@]}"; do
    if [[ "$candidate" == "$kernel" ]]; then
      return 0
    fi
  done
  return 1
}

if [[ "${1:-}" != "" && "${1:-}" != --* ]]; then
  if is_supported_kernel "$1"; then
    DATASET_LIST=("$1")
    shift
  else
    echo "Unknown 1D GP kernel '$1'. Supported kernels: ${SUPPORTED_KERNELS[*]}" >&2
    exit 2
  fi
fi

for dataset in "${DATASET_LIST[@]}"; do
  if ! is_supported_kernel "$dataset"; then
    echo "Unknown 1D GP kernel '$dataset'. Supported kernels: ${SUPPORTED_KERNELS[*]}" >&2
    exit 2
  fi
done

RUN_ROOT="${RUN_ROOT:-}"
if [[ -z "$RUN_ROOT" ]]; then
  RUN_ROOT="all_1d_gp_kernels_$(date +%Y%m%d_%H%M%S)"
fi

EXTRA_ARGS=("$@")
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
  case "${EXTRA_ARGS[$i]}" in
    --root)
      if (( i + 1 < ${#EXTRA_ARGS[@]} )); then
        RUN_ROOT="${EXTRA_ARGS[$((i + 1))]}"
      fi
      ;;
    --root=*)
      RUN_ROOT="${EXTRA_ARGS[$i]#--root=}"
      ;;
  esac
done

quote_words() {
  local quoted=""
  local word
  for word in "$@"; do
    printf -v word "%q" "$word"
    quoted+="${word} "
  done
  printf "%s" "${quoted% }"
}

run_completed() {
  local dataset="$1"
  local seed="$2"
  local method="$3"
  "$PYTHON_BIN" - "$RUN_ROOT" "$dataset" "$seed" "$method" <<'PY'
import json
import math
import sys
from pathlib import Path

root, dataset, seed, method = sys.argv[1:]

for path in Path("_experiments").joinpath(root).glob("*/summary.json"):
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            summary = json.load(f)
    except (OSError, json.JSONDecodeError):
        continue

    params = summary.get("parameters", {})
    results = summary.get("results", {})
    value = results.get("test_log_likelihood")
    has_result = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    if (
        summary.get("method_name") == method
        and str(params.get("data")) == dataset
        and str(params.get("seed")) == seed
        and has_result
    ):
        sys.exit(0)

sys.exit(1)
PY
}

LOG_DIR="_experiments/$RUN_ROOT/_logs"
mkdir -p "$LOG_DIR"

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
  "$PYTHON_BIN" scripts/aggregate_1d_results.py \
    --experiment-root "$RUN_ROOT" \
    --datasets "${DATASET_LIST[@]}" \
    --methods "${METHODS[@]}" \
    --seeds "${SEED_LIST[@]}" \
    --output-dir "$RESULTS_DIR" \
    --allow-empty
}

LAUNCH_COMMAND="RUN_ROOT=\"$RUN_ROOT\" METHODS=\"${METHODS[*]}\" KERNELS=\"${DATASET_LIST[*]}\" SEEDS=\"${SEED_LIST[*]}\" DEVICE=\"${DEVICE}\" RESULTS_DIR=\"${RESULTS_DIR}\" bash scripts/run_1d_all_methods.sh"
if [[ "$#" -gt 0 ]]; then
  LAUNCH_COMMAND+=" $(quote_words "$@")"
fi

mkdir -p "$RESULTS_DIR"
STATUS_CSV="$RESULTS_DIR/1d_all_methods_gp_kernels_${RUN_ROOT}_run_status.csv"
if [[ ! -f "$STATUS_CSV" ]]; then
  printf 'timestamp,experiment_root,dataset,method,seed,device,status,exit_code\n' > "$STATUS_CSV"
fi

EXPERIMENT_ID=""
EXPERIMENT_FINALIZED=0
if [[ "${EXPERIMENTS_LOG:-1}" != "0" && "${EXPERIMENTS_LOG:-1}" != "false" && "${EXPERIMENTS_LOG:-1}" != "no" && "${EXPERIMENTS_LOG:-1}" != "off" ]]; then
  record_output="$("$PYTHON_BIN" scripts/record_experiment.py \
    --aim "${EXPERIMENT_AIM:-Run all 1D methods on selected GP kernels}" \
    --command "$LAUNCH_COMMAND" \
    --server "${SERVER_ALIAS:-$(hostname)}" \
    --device "$DEVICE" \
    --model "${METHODS[*]}" \
    --task "regression" \
    --data "${DATASET_LIST[*]}" \
    --root "$RUN_ROOT" \
    --output-path "_experiments/$RUN_ROOT; aggregate CSV in $RESULTS_DIR/" \
    --extra "Kernels: ${DATASET_LIST[*]}" \
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
      --artifacts "_experiments/$RUN_ROOT; $RESULTS_DIR/1d_all_methods_gp_kernels_${RUN_ROOT}.csv"
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
trap finalize_on_exit EXIT

aggregate_progress

FAILED_RUNS=0
for dataset in "${DATASET_LIST[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    for method in "${METHODS[@]}"; do
      log_file="${LOG_DIR}/${dataset}_${seed}_${method}.log"
      if run_completed "$dataset" "$seed" "$method"; then
        echo "Skipping completed 1D kernel=${dataset} method=${method} seed=${seed} root=${RUN_ROOT}"
        continue
      fi

      echo "Running 1D kernel=${dataset} method=${method} seed=${seed} device=${DEVICE} root=${RUN_ROOT} log=${log_file}"
      if EXPERIMENTS_LOG=0 "$PYTHON_BIN" train.py \
        --task regression \
        --data "$dataset" \
        --model "$method" \
        --root "$RUN_ROOT" \
        --seed "$seed" \
        --device "$DEVICE" \
        "$@" > "$log_file" 2>&1; then
        echo "Completed 1D kernel=${dataset} method=${method} seed=${seed}"
        append_status_row "$dataset" "$method" "$seed" "completed" "0"
      else
        exit_code="$?"
        FAILED_RUNS=$((FAILED_RUNS + 1))
        append_status_row "$dataset" "$method" "$seed" "failed" "$exit_code"
        echo "Failed 1D kernel=${dataset} method=${method} seed=${seed}; see ${log_file}" >&2
        tail -n 40 "$log_file" >&2 || true
      fi

      aggregate_progress
    done
  done
done

aggregate_progress

if [[ "$FAILED_RUNS" -gt 0 ]]; then
  echo "$FAILED_RUNS run(s) failed. See $STATUS_CSV for details." >&2
  exit 1
fi

finalize_experiment \
  "Completed" \
  "All requested method, kernel, and seed combinations completed successfully."
