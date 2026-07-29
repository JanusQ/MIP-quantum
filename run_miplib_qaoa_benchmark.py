#!/usr/bin/env python3
"""Run cvIP/qaoa.py's BosonicQAOAIPSolver on a MIPLIB-derived benchmark.

This script does not modify cvIP/qaoa.py. It adapts the integer-coefficient MPS
files generated under data/miplib_benders/processed into the small
linear-equality form expected by BosonicQAOAIPSolver:

    max c^T x  subject to A x = b, x >= 0 integer.

The adapter selects a few binary variables from one equality row and fixes all
other variables to a known feasible assignment, producing a tiny exact QAOA test
instance that still records its MIPLIB-derived provenance.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np

from cvIP.qaoa import BosonicQAOAIPSolver


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "miplib_benders"
DEFAULT_MIP_DIR = DEFAULT_DATA_DIR / "processed"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "qaoa_runs"


@dataclass
class Row:
    name: str
    sense: str
    rhs: int = 0
    coeffs: dict[str, int] = field(default_factory=dict)


@dataclass
class Var:
    name: str
    lb: int = 0
    ub: int = 1
    obj: int = 0


@dataclass
class MPSModel:
    name: str
    rows: dict[str, Row]
    vars: dict[str, Var]


def parse_mps(path: Path) -> MPSModel:
    rows: dict[str, Row] = {}
    vars_: dict[str, Var] = {}
    obj_name = "OBJ"
    section = None

    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("*"):
            continue
        marker = raw[:14].strip().upper()
        if marker in {"NAME", "OBJSENSE", "ROWS", "COLUMNS", "RHS", "BOUNDS", "ENDATA"}:
            section = marker
            if marker == "ENDATA":
                break
            continue
        if section == "OBJSENSE":
            continue

        parts = raw.split()
        if section == "ROWS":
            sense, row_name = parts[0], parts[1]
            if sense == "N":
                obj_name = row_name
            else:
                rows[row_name] = Row(row_name, sense)
        elif section == "COLUMNS":
            if len(parts) >= 3 and parts[1] == "'MARKER'":
                continue
            var_name = parts[0]
            var = vars_.setdefault(var_name, Var(var_name))
            for i in range(1, len(parts) - 1, 2):
                row_name, value = parts[i], int(float(parts[i + 1]))
                if row_name == obj_name:
                    var.obj += value
                elif row_name in rows:
                    rows[row_name].coeffs[var_name] = rows[row_name].coeffs.get(var_name, 0) + value
        elif section == "RHS":
            for i in range(1, len(parts) - 1, 2):
                row_name, value = parts[i], int(float(parts[i + 1]))
                if row_name in rows:
                    rows[row_name].rhs = value
        elif section == "BOUNDS":
            btype = parts[0]
            var_name = parts[2]
            var = vars_.setdefault(var_name, Var(var_name))
            value = int(float(parts[3])) if len(parts) > 3 else None
            if btype == "BV":
                var.lb, var.ub = 0, 1
            elif btype in {"LI", "LO"}:
                var.lb = int(value)
            elif btype in {"UI", "UP"}:
                var.ub = int(value)
            elif btype == "FX":
                var.lb = var.ub = int(value)

    return MPSModel(path.stem, rows, vars_)


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def binary_variables(model: MPSModel) -> list[str]:
    return [name for name, var in model.vars.items() if var.lb == 0 and var.ub == 1]


def feasible_binary_assignments(model: MPSModel, names: list[str]) -> Iterable[dict[str, int]]:
    fixed = {name: var.lb for name, var in model.vars.items() if name not in names}
    for values in product([0, 1], repeat=len(names)):
        assignment = dict(fixed)
        assignment.update(dict(zip(names, values)))
        ok = True
        for row in model.rows.values():
            lhs = sum(coef * assignment[var] for var, coef in row.coeffs.items())
            if row.sense == "E" and lhs != row.rhs:
                ok = False
                break
            if row.sense == "L" and lhs > row.rhs:
                ok = False
                break
            if row.sense == "G" and lhs < row.rhs:
                ok = False
                break
        if ok:
            yield assignment


def choose_qaoa_problem(model: MPSModel, modes: int) -> dict:
    bin_vars = binary_variables(model)
    if len(bin_vars) < modes:
        raise ValueError(f"{model.name}: fewer than {modes} binary variables")

    feasible = next(feasible_binary_assignments(model, bin_vars), None)
    if feasible is None:
        raise ValueError(f"{model.name}: no feasible binary assignment found")

    equality_rows = [row for row in model.rows.values() if row.sense == "E"]
    equality_rows.sort(key=lambda row: (-sum(1 for v in row.coeffs if v in bin_vars), row.name))

    for row in equality_rows:
        candidates = [v for v in sorted(row.coeffs) if v in bin_vars and row.coeffs[v] != 0]
        if len(candidates) < 2:
            continue
        selected = candidates[:modes]
        if len(selected) < modes:
            selected.extend(v for v in bin_vars if v not in selected)
            selected = selected[:modes]
        fixed_rhs = row.rhs - sum(coef * feasible[var] for var, coef in row.coeffs.items() if var not in selected)
        A = [[int(row.coeffs.get(var, 0)) for var in selected]]
        b = [int(fixed_rhs)]
        if sum(abs(x) for x in A[0]) == 0:
            continue
        feasible_count = sum(
            1
            for x in product([0, 1], repeat=modes)
            if np.array_equal(np.array(A, dtype=int) @ np.array(x, dtype=int), np.array(b, dtype=int))
        )
        if feasible_count == 0:
            continue
        c = [int(model.vars[var].obj) for var in selected]
        if all(v == 0 for v in c):
            c = [i + 1 for i in range(modes)]
        problem = {
            "name": f"{model.name}_qaoa_{row.name}",
            "source_mps": model.name,
            "source_row": row.name,
            "variables": selected,
            "fixed_variables_from_feasible_assignment": {
                name: int(value) for name, value in feasible.items() if name not in selected and value != model.vars[name].lb
            },
            "A": A,
            "b": b,
            "c": c,
            "N": 2,
            "feasible_states_in_qaoa_truncation": feasible_count,
        }
        if feasible_count >= 2:
            return problem

    raise ValueError(f"{model.name}: no suitable equality row for QAOA adapter")


def build_or_load_problem(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.problem:
        return json.loads(Path(args.problem).read_text(encoding="utf-8"))

    mps_files = sorted(Path(args.mip_dir).glob("*/*.mps"))
    errors = []
    for mps in mps_files:
        try:
            problem = choose_qaoa_problem(parse_mps(mps), args.modes)
            out = output_dir / f"{safe_filename(problem['name'])}.json"
            out.write_text(json.dumps(problem, indent=2), encoding="utf-8")
            print(f"Wrote QAOA problem: {out}")
            return problem
        except Exception as exc:
            errors.append(f"{mps.name}: {exc}")
    raise RuntimeError("Could not build a QAOA problem:\n" + "\n".join(errors))


def run_qaoa(problem: dict, args: argparse.Namespace) -> dict:
    summary_keys = ("name", "source_mps", "source_row", "source_rows", "decomposition", "benders_subproblem_id", "variables", "A", "b", "c", "N")
    print(json.dumps({k: problem[k] for k in summary_keys if k in problem}, indent=2))
    solver = BosonicQAOAIPSolver(
        problem["A"],
        problem["b"],
        problem["c"],
        N=int(problem.get("N", 2)),
        p=args.p,
        maxiter=args.maxiter,
        seed=args.seed,
        circuit_type=args.circuit_type,
    )
    if args.smoke:
        print(json.dumps({"initialized": True, "modes": solver.num_modes, "feasible_states": len(solver.feasible_states)}, indent=2))
        return {"obj": None, "params": []}
    result = solver.optimize()
    solver.print_summary()
    return {"obj": float(result["obj"]), "params": [float(x) for x in result["params"]]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mip-dir", default=str(DEFAULT_MIP_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--problem", help="Use an existing exported QAOA JSON problem.")
    parser.add_argument("--modes", type=int, default=4)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--circuit-type", default="multi_beta", choices=["beta_gamma", "multi_beta", "multi_beta_oneH"])
    parser.add_argument("--smoke", action="store_true", help="Only initialize the solver and skip optimization.")
    args = parser.parse_args()

    problem = build_or_load_problem(args)
    result = run_qaoa(problem, args)
    result_path = Path(args.output_dir) / f"{safe_filename(problem['name'])}_result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote QAOA result: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
