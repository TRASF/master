#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASET_DIR="${WINGBEAT_DATASET_DIR:?Set WINGBEAT_DATASET_DIR}"
BENCH_ROOT="${WINGBEAT_BENCHMARK_ROOT:-$REPO_ROOT/runtime/precision-benchmark}"
CACHE_DIR="${WINGBEAT_CACHE_DIR:-$BENCH_ROOT/shared-cache}"
TOLERANCE="${WINGBEAT_PARITY_TOLERANCE:-0.05}"

run_case() {
  local name="$1"
  local profile="$2"
  local runtime="$BENCH_ROOT/$name"
  WINGBEAT_RUNTIME_ROOT="$runtime" \
  WINGBEAT_CACHE_DIR="$CACHE_DIR" \
  WINGBEAT_DATASET_DIR="$DATASET_DIR" \
  WINGBEAT_PROFILE="$REPO_ROOT/configs/profiles/$profile" \
  WINGBEAT_ENABLE_WANDB=false \
  WINGBEAT_CAPTURE_CONSOLE_LOG=false \
    bash "$REPO_ROOT/ops/wandb/train-pretrain.sh"
}

# The first run owns graph/kernel initialization and fills the stable cache.
run_case prime-cache benchmark_fp32.yaml
run_case fp32 benchmark_fp32.yaml
run_case mixed benchmark_mixed.yaml

FP32_METRICS="$(find "$BENCH_ROOT/fp32" -name metrics.jsonl -type f -print -quit)"
MIXED_METRICS="$(find "$BENCH_ROOT/mixed" -name metrics.jsonl -type f -print -quit)"
REPORT="$BENCH_ROOT/precision-benchmark.json"

python - "$FP32_METRICS" "$MIXED_METRICS" "$REPORT" "$TOLERANCE" <<'PY'
import json
import statistics
import sys
from pathlib import Path


def summarize(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    steady = rows[1:] if len(rows) > 1 else rows
    throughput = [
        row["train_examples"] / row["train_duration_seconds"]
        for row in steady
        if row.get("train_duration_seconds", 0) > 0
    ]
    return {
        "epochs": len(rows),
        "steady_examples_per_second": statistics.median(throughput),
        "best_val_macro_f1": max(row.get("val_macro_f1", 0.0) for row in rows),
    }


fp32 = summarize(sys.argv[1])
mixed = summarize(sys.argv[2])
tolerance = float(sys.argv[4])
speedup = mixed["steady_examples_per_second"] / fp32["steady_examples_per_second"]
parity_delta = mixed["best_val_macro_f1"] - fp32["best_val_macro_f1"]
accepted = speedup >= 0.98 and parity_delta >= -tolerance
report = {
    "fp32": fp32,
    "mixed_float16": mixed,
    "speedup": speedup,
    "val_macro_f1_delta": parity_delta,
    "parity_tolerance": tolerance,
    "accepted": accepted,
    "recommended_precision": "auto" if accepted else "float32",
}
Path(sys.argv[3]).parent.mkdir(parents=True, exist_ok=True)
Path(sys.argv[3]).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
PY

echo "Precision benchmark: $REPORT"
