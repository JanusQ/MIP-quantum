#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -d ".venv-classical-a100" ]; then
  # shellcheck disable=SC1091
  . ".venv-classical-a100/bin/activate"
elif [ -d ".venv-bench" ]; then
  # shellcheck disable=SC1091
  . ".venv-bench/bin/activate"
fi

OUT_DIR="data/miplib_benders/classical_results"
mkdir -p "$OUT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
PREFIX="classical_benchmark_a100_${STAMP}"
ENV_JSON="$OUT_DIR/${PREFIX}.environment.json"

python3 benchmark_classical_ilp.py \
  --timeout "${CLASSICAL_TIMEOUT:-60}" \
  --solvers "${CLASSICAL_SOLVERS:-exhaustive,gurobi,cplex,scip}" \
  --output-dir "$OUT_DIR" \
  --output-prefix "$PREFIX" \
  --env-output "$ENV_JSON"

echo "CSV prefix: $OUT_DIR/${PREFIX}_<solver>.csv"
echo "ENV: $ENV_JSON"
