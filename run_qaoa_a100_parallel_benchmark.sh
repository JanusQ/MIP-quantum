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

JOBS="${QAOA_JOBS:-2}"
if [ "$JOBS" -lt 1 ]; then
  echo "QAOA_JOBS must be >= 1" >&2
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
PREFIX="${QAOA_OUTPUT_PREFIX:-qaoa_feasible_benchmark_a100_parallel_${STAMP}}"
COMPLETED_CSV="${QAOA_COMPLETED_CSV:-}"
if [ -z "$COMPLETED_CSV" ]; then
  COMPLETED_CSV="$(find "$OUT_DIR" -maxdepth 1 -type f -name '*_qaoa.csv' -print | sort | tail -1 || true)"
fi

EXTRA_ARGS=()
if [ -n "${QAOA_MAX_STATE_SPACE:-}" ]; then
  EXTRA_ARGS+=(--max-state-space "$QAOA_MAX_STATE_SPACE")
fi
if [ -n "$COMPLETED_CSV" ]; then
  EXTRA_ARGS+=(--completed-csv "$COMPLETED_CSV")
fi

echo "Launching $JOBS QAOA queues"
if [ -n "$COMPLETED_CSV" ]; then
  echo "Completed CSV: $COMPLETED_CSV"
else
  echo "Completed CSV: none"
fi

PIDS=()
for shard in $(seq 0 $((JOBS - 1))); do
  SHARD_PREFIX="${PREFIX}_shard$(printf '%02d' "$shard")"
  LOG="$OUT_DIR/${SHARD_PREFIX}.log"
  ENV_JSON="$OUT_DIR/${SHARD_PREFIX}.environment.json"
  python3 benchmark_qaoa_feasible.py \
    --timeout "${QAOA_TIMEOUT:-0}" \
    --p "${QAOA_P:-1}" \
    --maxiter "${QAOA_MAXITER:-5}" \
    --circuit-type "${QAOA_CIRCUIT_TYPE:-multi_beta}" \
    --output-dir "$OUT_DIR" \
    --output-prefix "$SHARD_PREFIX" \
    --env-output "$ENV_JSON" \
    --shard-count "$JOBS" \
    --shard-index "$shard" \
    "${EXTRA_ARGS[@]}" \
    > "$LOG" 2>&1 &
  PIDS+=("$!")
  echo "Shard $shard PID ${PIDS[-1]} LOG $LOG CSV $OUT_DIR/${SHARD_PREFIX}_qaoa.csv"
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
