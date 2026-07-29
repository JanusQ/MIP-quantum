#!/usr/bin/env python3
"""Classical ILP baselines for exported QAOA benchmark JSON files.

Each selected solver gets its own CSV file. Every row records both:
- the first feasible solution found by that solver;
- the best proven optimal solution for max c^T x, when found before timeout.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shutil
import time
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import psutil


ROOT = Path(__file__).resolve().parent
DEFAULT_PROBLEM_DIR = ROOT / "data" / "miplib_benders" / "qaoa_problems"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "miplib_benders" / "classical_results"


@dataclass
class Problem:
    path: Path
    name: str
    variables: list[str]
    A: list[list[int]]
    b: list[int]
    c: list[float]
    bounds: list[tuple[int, int]]


def load_problem(path: Path) -> Problem:
    data = json.loads(path.read_text(encoding="utf-8"))
    variables = list(data["variables"])
    default_ub = int(data.get("N", 2)) - 1
    raw_bounds = data.get("variable_bounds", {})
    bounds = []
    for var in variables:
        bound = raw_bounds.get(var, {"lb": 0, "ub": default_ub})
        bounds.append((int(bound["lb"]), int(bound["ub"])))
    return Problem(
        path=path,
        name=str(data.get("name", path.stem)),
        variables=variables,
        A=[[int(v) for v in row] for row in data["A"]],
        b=[int(v) for v in data["b"]],
        c=[float(v) for v in data.get("c", [0] * len(variables))],
        bounds=bounds,
    )


def select_problem_files(root: Path, glob_pattern: str, limit: int | None, per_nvars: int | None) -> list[Path]:
    files = sorted(p for p in root.glob(glob_pattern) if p.name != "manifest.json")
    if per_nvars is not None:
        grouped: dict[str, list[Path]] = defaultdict(list)
        for path in files:
            grouped[path.parent.name].append(path)
        files = []
        for group in sorted(grouped):
            files.extend(grouped[group][:per_nvars])
    return files[:limit] if limit is not None else files


def env_info() -> dict[str, Any]:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 3),
        "python_executable": os.sys.executable,
        "gurobi_cl": shutil.which("gurobi_cl") or "",
        "cplex": shutil.which("cplex") or "",
        "scip": shutil.which("scip") or "",
    }
    for module in ["gurobipy", "cplex", "pyscipopt", "psutil"]:
        try:
            __import__(module)
            info[f"python_module_{module}"] = True
        except Exception:
            info[f"python_module_{module}"] = False
    try:
        import subprocess

        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
            timeout=10,
        )
        info["nvidia_smi"] = [line.strip() for line in out.splitlines() if line.strip()]
    except Exception:
        info["nvidia_smi"] = []
    return info


def base_row(problem: Problem, solver: str) -> dict[str, Any]:
    return {
        "problem": problem.name,
        "path": str(problem.path),
        "nvars": len(problem.variables),
        "nconstraints": len(problem.A),
        "candidate_state_space_size": math.prod(ub - lb + 1 for lb, ub in problem.bounds),
        "solver": solver,
        "first_feasible_status": "",
        "time_to_first_feasible_sec": "",
        "first_feasible_objective": "",
        "first_feasible_solution": "",
        "optimal_status": "",
        "time_to_optimal_sec": "",
        "optimal_objective": "",
        "optimal_solution": "",
        "wall_sec": "",
        "peak_memory_mb": "",
        "nodes": "",
        "error": "",
    }


def objective(problem: Problem, values: list[int]) -> float:
    return float(sum(coef * value for coef, value in zip(problem.c, values)))


def solution_json(problem: Problem, values: list[int]) -> str:
    return json.dumps(dict(zip(problem.variables, [int(v) for v in values])), sort_keys=True)


def unavailable(problem: Problem, solver: str) -> dict[str, Any]:
    row = base_row(problem, solver)
    row["first_feasible_status"] = "not_available"
    row["optimal_status"] = "not_available"
    return row


def exhaustive_solve(problem: Problem, timeout: float | None) -> dict[str, Any]:
    row = base_row(problem, "exhaustive")
    start = time.perf_counter()
    tracemalloc.start()
    nrows = len(problem.A)
    nvars = len(problem.variables)
    values = [0] * nvars
    partial = [0] * nrows
    nodes = 0
    first_values: list[int] | None = None
    first_time = None
    best_values: list[int] | None = None
    best_obj = -math.inf

    suffix_min = [[0] * nrows for _ in range(nvars + 1)]
    suffix_max = [[0] * nrows for _ in range(nvars + 1)]
    suffix_obj_max = [0.0] * (nvars + 1)
    for i in range(nvars - 1, -1, -1):
        lb, ub = problem.bounds[i]
        coef_obj = problem.c[i]
        suffix_obj_max[i] = suffix_obj_max[i + 1] + max(coef_obj * lb, coef_obj * ub)
        for r in range(nrows):
            coef = problem.A[r][i]
            suffix_min[i][r] = suffix_min[i + 1][r] + (coef * lb if coef >= 0 else coef * ub)
            suffix_max[i][r] = suffix_max[i + 1][r] + (coef * ub if coef >= 0 else coef * lb)

    def timed_out() -> bool:
        return timeout is not None and time.perf_counter() - start >= timeout

    def can_finish(depth: int) -> bool:
        for r in range(nrows):
            if partial[r] + suffix_min[depth][r] > problem.b[r]:
                return False
            if partial[r] + suffix_max[depth][r] < problem.b[r]:
                return False
        return True

    def dfs(depth: int, obj_so_far: float) -> None:
        nonlocal nodes, first_values, first_time, best_values, best_obj
        if timed_out():
            return
        if obj_so_far + suffix_obj_max[depth] < best_obj:
            return
        if not can_finish(depth):
            return
        if depth == nvars:
            nodes += 1
            if partial == problem.b:
                if first_values is None:
                    first_values = values.copy()
                    first_time = time.perf_counter() - start
                if obj_so_far > best_obj:
                    best_obj = obj_so_far
                    best_values = values.copy()
            return

        row_coeffs = [problem.A[r][depth] for r in range(nrows)]
        lb, ub = problem.bounds[depth]
        candidates = range(lb, ub + 1)
        if problem.c[depth] > 0:
            candidates = range(ub, lb - 1, -1)
        for value in candidates:
            values[depth] = value
            for r, coef in enumerate(row_coeffs):
                partial[r] += coef * value
            nodes += 1
            dfs(depth + 1, obj_so_far + problem.c[depth] * value)
            for r, coef in enumerate(row_coeffs):
                partial[r] -= coef * value
            if timed_out():
                return

    dfs(0, 0.0)

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed = time.perf_counter() - start
    row["wall_sec"] = elapsed
    row["peak_memory_mb"] = peak / (1024 * 1024)
    row["nodes"] = nodes
    if first_values is None:
        row["first_feasible_status"] = "timeout" if timed_out() else "infeasible"
    else:
        row["first_feasible_status"] = "feasible"
        row["time_to_first_feasible_sec"] = first_time
        row["first_feasible_objective"] = objective(problem, first_values)
        row["first_feasible_solution"] = solution_json(problem, first_values)
    if timed_out():
        row["optimal_status"] = "timeout"
    elif best_values is None:
        row["optimal_status"] = "infeasible"
    else:
        row["optimal_status"] = "optimal"
        row["time_to_optimal_sec"] = elapsed
        row["optimal_objective"] = best_obj
        row["optimal_solution"] = solution_json(problem, best_values)
    return row


def gurobi_solve(problem: Problem, timeout: float | None) -> dict[str, Any]:
    try:
        import gurobipy as gp
    except ImportError:
        return unavailable(problem, "gurobi")

    row = base_row(problem, "gurobi")
    proc = psutil.Process(os.getpid())
    start_rss = proc.memory_info().rss / (1024 * 1024)
    start = time.perf_counter()
    try:
        model = gp.Model()
        model.Params.OutputFlag = 0
        if timeout is not None:
            model.Params.TimeLimit = float(timeout)
        xvars = [model.addVar(lb=lb, ub=ub, vtype=gp.GRB.INTEGER, name=f"x{i}") for i, (lb, ub) in enumerate(problem.bounds)]
        model.setObjective(gp.quicksum(problem.c[i] * xvars[i] for i in range(len(xvars))), gp.GRB.MAXIMIZE)
        for r, row_a in enumerate(problem.A):
            model.addConstr(gp.quicksum(coef * xvars[i] for i, coef in enumerate(row_a)) == problem.b[r], name=f"c{r}")

        first_seen = {"time": None, "values": None}

        def callback(model_: gp.Model, where: int) -> None:
            if where == gp.GRB.Callback.MIPSOL and first_seen["values"] is None:
                vals = [int(round(model_.cbGetSolution(var))) for var in xvars]
                first_seen["time"] = time.perf_counter() - start
                first_seen["values"] = vals

        model.optimize(callback)
        elapsed = time.perf_counter() - start
        row["wall_sec"] = elapsed
        if first_seen["values"] is not None:
            vals = first_seen["values"]
            row["first_feasible_status"] = "feasible"
            row["time_to_first_feasible_sec"] = first_seen["time"]
            row["first_feasible_objective"] = objective(problem, vals)
            row["first_feasible_solution"] = solution_json(problem, vals)
        elif model.SolCount > 0:
            vals = [int(round(var.X)) for var in xvars]
            row["first_feasible_status"] = "feasible"
            row["time_to_first_feasible_sec"] = elapsed
            row["first_feasible_objective"] = objective(problem, vals)
            row["first_feasible_solution"] = solution_json(problem, vals)
        else:
            row["first_feasible_status"] = "timeout" if model.Status == gp.GRB.TIME_LIMIT else "infeasible"
        if model.Status == gp.GRB.OPTIMAL:
            vals = [int(round(var.X)) for var in xvars]
            row["optimal_status"] = "optimal"
            row["time_to_optimal_sec"] = elapsed
            row["optimal_objective"] = objective(problem, vals)
            row["optimal_solution"] = solution_json(problem, vals)
        elif model.Status == gp.GRB.TIME_LIMIT:
            row["optimal_status"] = "timeout"
        else:
            row["optimal_status"] = "infeasible" if model.Status in {gp.GRB.INFEASIBLE, gp.GRB.INF_OR_UNBD} else "error"
    except Exception as exc:
        row["first_feasible_status"] = "error"
        row["optimal_status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["wall_sec"] = time.perf_counter() - start
    row["peak_memory_mb"] = max(0.0, proc.memory_info().rss / (1024 * 1024) - start_rss)
    return row


def scip_solve(problem: Problem, timeout: float | None) -> dict[str, Any]:
    try:
        from pyscipopt import Model, quicksum
    except ImportError:
        return unavailable(problem, "scip")

    row = base_row(problem, "scip")
    proc = psutil.Process(os.getpid())
    start_rss = proc.memory_info().rss / (1024 * 1024)
    start = time.perf_counter()
    try:
        model = Model()
        model.hideOutput()
        if timeout is not None:
            model.setParam("limits/time", float(timeout))
        xvars = [model.addVar(vtype="I", lb=float(lb), ub=float(ub), name=f"x{i}") for i, (lb, ub) in enumerate(problem.bounds)]
        for r, row_a in enumerate(problem.A):
            model.addCons(quicksum(float(coef) * xvars[i] for i, coef in enumerate(row_a)) == float(problem.b[r]), name=f"c{r}")
        model.setObjective(quicksum(float(problem.c[i]) * xvars[i] for i in range(len(xvars))), "maximize")
        model.optimize()
        elapsed = time.perf_counter() - start
        row["wall_sec"] = elapsed
        sol = model.getBestSol()
        raw_status = str(model.getStatus()).lower()
        if sol is not None:
            vals = [int(round(model.getSolVal(sol, var))) for var in xvars]
            row["first_feasible_status"] = "feasible"
            row["time_to_first_feasible_sec"] = elapsed
            row["first_feasible_objective"] = objective(problem, vals)
            row["first_feasible_solution"] = solution_json(problem, vals)
            if "optimal" in raw_status:
                row["optimal_status"] = "optimal"
                row["time_to_optimal_sec"] = elapsed
                row["optimal_objective"] = objective(problem, vals)
                row["optimal_solution"] = solution_json(problem, vals)
            else:
                row["optimal_status"] = "timeout" if "time" in raw_status else raw_status
        else:
            row["first_feasible_status"] = "timeout" if "time" in raw_status else ("infeasible" if "infeasible" in raw_status else "error")
            row["optimal_status"] = row["first_feasible_status"]
    except Exception as exc:
        row["first_feasible_status"] = "error"
        row["optimal_status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["wall_sec"] = time.perf_counter() - start
    row["peak_memory_mb"] = max(0.0, proc.memory_info().rss / (1024 * 1024) - start_rss)
    return row


def cplex_solve(problem: Problem, timeout: float | None) -> dict[str, Any]:
    try:
        import cplex
    except ImportError:
        return unavailable(problem, "cplex")

    row = base_row(problem, "cplex")
    proc = psutil.Process(os.getpid())
    start_rss = proc.memory_info().rss / (1024 * 1024)
    start = time.perf_counter()
    try:
        model = cplex.Cplex()
        model.set_log_stream(None)
        model.set_error_stream(None)
        model.set_warning_stream(None)
        model.set_results_stream(None)
        names = [f"x{i}" for i in range(len(problem.variables))]
        model.objective.set_sense(model.objective.sense.maximize)
        model.variables.add(
            names=names,
            lb=[float(lb) for lb, _ in problem.bounds],
            ub=[float(ub) for _, ub in problem.bounds],
            obj=[float(v) for v in problem.c],
            types=[model.variables.type.integer] * len(problem.variables),
        )
        constraints = [cplex.SparsePair(ind=names, val=[float(v) for v in row_a]) for row_a in problem.A]
        model.linear_constraints.add(lin_expr=constraints, senses=["E"] * len(problem.A), rhs=[float(v) for v in problem.b])
        if timeout is not None:
            model.parameters.timelimit.set(float(timeout))
        model.solve()
        elapsed = time.perf_counter() - start
        row["wall_sec"] = elapsed
        status_text = model.solution.get_status_string().lower()
        try:
            vals = [int(round(v)) for v in model.solution.get_values()]
        except Exception:
            vals = []
        if vals:
            row["first_feasible_status"] = "feasible"
            row["time_to_first_feasible_sec"] = elapsed
            row["first_feasible_objective"] = objective(problem, vals)
            row["first_feasible_solution"] = solution_json(problem, vals)
            if "optimal" in status_text:
                row["optimal_status"] = "optimal"
                row["time_to_optimal_sec"] = elapsed
                row["optimal_objective"] = objective(problem, vals)
                row["optimal_solution"] = solution_json(problem, vals)
            else:
                row["optimal_status"] = "timeout" if "time" in status_text else status_text
        else:
            row["first_feasible_status"] = "timeout" if "time" in status_text else ("infeasible" if "infeasible" in status_text else "error")
            row["optimal_status"] = row["first_feasible_status"]
    except Exception as exc:
        row["first_feasible_status"] = "error"
        row["optimal_status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["wall_sec"] = time.perf_counter() - start
    row["peak_memory_mb"] = max(0.0, proc.memory_info().rss / (1024 * 1024) - start_rss)
    return row


def solve(problem: Problem, solver: str, timeout: float | None) -> dict[str, Any]:
    if solver == "exhaustive":
        return exhaustive_solve(problem, timeout)
    if solver == "gurobi":
        return gurobi_solve(problem, timeout)
    if solver == "cplex":
        return cplex_solve(problem, timeout)
    if solver == "scip":
        return scip_solve(problem, timeout)
    return unavailable(problem, solver)


def output_path_for_solver(output_dir: Path, prefix: str, solver: str) -> Path:
    return output_dir / f"{prefix}_{solver}.csv"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-dir", default=str(DEFAULT_PROBLEM_DIR))
    parser.add_argument("--glob", default="nvars_*/*.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--per-nvars", type=int)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--solvers", default="exhaustive,gurobi,cplex,scip")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-prefix", default="classical_benchmark_results")
    parser.add_argument("--env-output", help="Write machine/GPU environment JSON. Defaults in output-dir.")
    args = parser.parse_args()

    files = select_problem_files(Path(args.problem_dir), args.glob, args.limit, args.per_nvars)
    solvers = [item.strip() for item in args.solvers.split(",") if item.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env_path = Path(args.env_output) if args.env_output else output_dir / f"{args.output_prefix}.environment.json"
    env_path.write_text(json.dumps(env_info(), indent=2), encoding="utf-8")

    fields = list(base_row(load_problem(files[0]), solvers[0] if solvers else "solver").keys()) if files and solvers else []
    handles = {}
    writers = {}
    try:
        for solver in solvers:
            path = output_path_for_solver(output_dir, args.output_prefix, solver)
            f = path.open("w", newline="", encoding="utf-8")
            handles[solver] = f
            writers[solver] = csv.DictWriter(f, fieldnames=fields)
            writers[solver].writeheader()
        for i, path in enumerate(files, 1):
            problem = load_problem(path)
            for solver in solvers:
                row = solve(problem, solver, args.timeout)
                writers[solver].writerow({key: row.get(key, "") for key in fields})
                handles[solver].flush()
            print(f"[{i}/{len(files)}] {problem.name}")
    finally:
        for f in handles.values():
            f.close()
    for solver in solvers:
        print(f"Wrote {output_path_for_solver(output_dir, args.output_prefix, solver)}")
    print(f"Wrote {env_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
