# MIPLIB Benders QAOA Benchmarks

This directory contains the MIPLIB-derived benchmark data used by the local
`MIP-quantum` test scripts.

Sources:

- MIPLIB benchmark page: https://miplib.zib.de/tag_benchmark.html
- MIPLIB benchmark archive: https://miplib.zib.de/downloads/benchmark.zip
- MIPLIB benchmark list: https://miplib.zib.de/downloads/benchmark-v2.test

## Layout

```text
data/miplib_benders/
  processed/<instance>/
    <instance>.mps
    metadata.json
  qaoa_problems/
    manifest.json
    nvars_04/*.json
    ...
    nvars_10/*.json
  docs/
    benchmark_protocol.md
```

Related code in this repository:

```text
benchmark_scripts/build_miplib_exact_benchmarks.py
export_miplib_qaoa_problems.py
run_miplib_qaoa_benchmark.py
cvIP/qaoa.py
```

`cvIP/qaoa.py` is not modified by the benchmark generation code.

## Regenerate

From the repository root:

```bash
cd /Users/ghost/Downloads/MIP-quantum
python3 benchmark_scripts/build_miplib_exact_benchmarks.py --download --instances 240 --max-vars 18 --max-domain 8 --max-rows 40
python3 export_miplib_qaoa_problems.py --clean --min-modes 4 --max-modes 10 --per-mode 100 --N 8 --per-source-cap 120 --per-block-cap 50 --combos-per-row 200 --row-groups-per-block 80 --max-candidate-state-space 2000000000 --min-nonzeros-per-row 3 --min-feasible-states 2 --max-feasible-states 200000
```

The exporter reads:

```text
data/miplib_benders/processed
```

and writes:

```text
data/miplib_benders/qaoa_problems
```

## Current Export

The current export contains 700 QAOA-ready JSON problems:

```text
nvars_04: 100
nvars_05: 100
nvars_06: 100
nvars_07: 100
nvars_08: 100
nvars_09: 100
nvars_10: 100
```

Constraint count distribution:

```text
1 constraint : 198
2 constraints: 336
3 constraints: 166
```

Other checks:

```text
distinct processed source instances: 19
distinct Benders blocks: 20
duplicate mathematical signatures: 0
exact feasible-state counts: 700 / 700
feasible-state count range: 2..9838
```

Each exported problem:

- is sampled from one strict `metadata.json` Benders subproblem block;
- has 4 to 10 variables;
- has 1 to 3 equality constraints;
- uses integer coefficients;
- uses `N = 8`;
- normalizes benchmark variables to `0 <= x_i <= 7`;
- requires every constraint row to contain at least 3 nonzero source-variable coefficients;
- requires the variable-constraint bipartite graph to be connected;
- stores exact feasible-state counts in the QAOA truncation.

Rows with `<=` or `>=` are converted into equality form by adding a nonnegative
row slack, matching the `BosonicQAOAIPSolver(A, b, c, N=...)` interface.

## Smoke Test

From the repository root:

```bash
python3 run_miplib_qaoa_benchmark.py --problem data/miplib_benders/qaoa_problems/nvars_04/assign1-5-8_ds_qaoa_R0002_0001.json --smoke
```

This initializes `cvIP.qaoa.BosonicQAOAIPSolver` without running the expensive
optimization loop.
