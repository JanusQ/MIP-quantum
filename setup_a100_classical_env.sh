#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv-classical-a100}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Missing python3. Install Python before running this script." >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
. "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install psutil gurobipy cplex pyscipopt qutip scipy matplotlib sympy

echo "Environment ready: $VENV_DIR"
python - <<'PY'
import importlib.util
for name in ["psutil", "gurobipy", "cplex", "pyscipopt", "qutip", "scipy", "matplotlib", "sympy"]:
    print(name, "available" if importlib.util.find_spec(name) else "missing")
PY

echo "Command-line solvers:"
command -v gurobi_cl || true
command -v cplex || true
command -v scip || true
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
