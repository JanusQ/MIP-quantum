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

OUT_DIR="data/miplib_benders/qaoa_results"
mkdir -p "$OUT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
PREFIX="qaoa_feasible_benchmark_a100_${STAMP}"
ENV_JSON="$OUT_DIR/${PREFIX}.environment.json"
EXTRA_ARGS=()
if [ -n "${QAOA_MAX_STATE_SPACE:-}" ]; then
  EXTRA_ARGS+=(--max-state-space "$QAOA_MAX_STATE_SPACE")
fi
if [ -n "${QAOA_OUTPUT_PREFIX:-}" ]; then
  PREFIX="$QAOA_OUTPUT_PREFIX"
  ENV_JSON="$OUT_DIR/${PREFIX}.environment.json"
fi
if [ "${QAOA_RESUME:-0}" = "1" ]; then
  EXTRA_ARGS+=(--resume)
fi

python3 benchmark_qaoa_feasible.py \
  --timeout "${QAOA_TIMEOUT:-0}" \
  --p "${QAOA_P:-1}" \
  --maxiter "${QAOA_MAXITER:-5}" \
  --circuit-type "${QAOA_CIRCUIT_TYPE:-multi_beta}" \
  --output-dir "$OUT_DIR" \
  --output-prefix "$PREFIX" \
  --env-output "$ENV_JSON" \
  "${EXTRA_ARGS[@]}"

echo "CSV: $OUT_DIR/${PREFIX}_qaoa.csv"
echo "ENV: $ENV_JSON"
