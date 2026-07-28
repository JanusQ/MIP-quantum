# Classical ILP Baseline Protocol

Goal: compare the benchmark instances against classical ILP solvers by measuring
time to the first feasible solution and peak memory.

Run on the A100 node from the repository root:

```bash
cd /path/to/MIP-quantum
./setup_a100_classical_env.sh
./run_classical_a100_benchmark.sh
```

Equivalent explicit command:

```bash
python3 benchmark_classical_ilp.py \
  --timeout 60 \
  --solvers exhaustive,gurobi,cplex,scip \
  --output data/miplib_benders/classical_results/classical_benchmark_a100.csv \
  --env-output data/miplib_benders/classical_results/classical_benchmark_a100.environment.json
```

The CSV records:

- problem path, variable count, constraint count, candidate state-space size;
- solver name and status;
- time to first feasible solution in seconds;
- wall time in seconds;
- peak resident memory in MB;
- exhaustive-search node count where applicable;
- first feasible solution.

The environment JSON records CPU, RAM, solver command paths, and `nvidia-smi`
GPU information.  Classical ILP solvers generally run on CPU; the A100 metadata
is recorded to document the machine used for the benchmark.

Expected solver commands on the A100 node:

```bash
which gurobi_cl
which cplex
which scip
nvidia-smi
```

If a solver command is missing, its CSV rows are marked `not_available`.
The setup script installs Python fallbacks for Gurobi, CPLEX, and SCIP
(`gurobipy`, `cplex`, `pyscipopt`) in `.venv-classical-a100`, so command-line
solver binaries are preferred but not required when the Python package and
license are usable.

Do not commit files under:

```text
data/miplib_benders/classical_results/
```
