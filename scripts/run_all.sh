#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/qwen_a100.yaml}"
LOG_FILE="results/run.log"

mkdir -p results

python scripts/build_datasets.py --out data/ 2>&1 | tee -a "$LOG_FILE"
python scripts/benchmark_paper.py --config "$CONFIG" --out-dir results/ 2>&1 | tee -a "$LOG_FILE"

# benchmark_paper.py writes results/LATEST as soon as the run dir is created,
# so even partial-failure runs leave a usable pointer here.
RUN_DIR=$(cat results/LATEST)

python scripts/plot_paper.py --in "$RUN_DIR" --out "$RUN_DIR/plots/" 2>&1 | tee -a "$LOG_FILE"

echo "Done. Results in $RUN_DIR"
