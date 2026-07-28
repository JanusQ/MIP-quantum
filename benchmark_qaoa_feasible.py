#!/usr/bin/env python3
"""QAOA first-feasible benchmark for exported problem JSON files.

The current BosonicQAOAIPSolver discovers feasible Fock states during solver
initialization. This script records the first feasible Fock state in the
enumerated feasible list and peak memory usage.

Each problem runs in a child process so timeouts, import failures, and memory
pressure are recorded as benchmark rows instead of aborting the whole batch.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from threading import Event, Thread
from typing import Any

import psutil


ROOT = Path(__file__).resolve().parent
DEFAULT_PROBLEM_DIR = ROOT / "data" / "miplib_benders" / "qaoa_problems"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "miplib_benders" / "qaoa_results"


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
        "python_executable": sys.executable,
        "gurobi_cl": shutil.which("gurobi_cl") or "",
        "cplex": shutil.which("cplex") or "",
        "scip": shutil.which("scip") or "",
    }
    for module in ["numpy", "scipy", "qutip", "sympy", "psutil"]:
        try:
            __import__(module)
            info[f"python_module_{module}"] = True
        except Exception:
            info[f"python_module_{module}"] = False
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
            timeout=10,
        )
        info["nvidia_smi"] = [line.strip() for line in out.splitlines() if line.strip()]
    except Exception:
        info["nvidia_smi"] = []
    return info


def problem_metadata(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    variables = list(data["variables"])
    default_ub = int(data.get("N", 2)) - 1
    raw_bounds = data.get("variable_bounds", {})
    bounds = []
    for var in variables:
        bound = raw_bounds.get(var, {"lb": 0, "ub": default_ub})
        bounds.append((int(bound["lb"]), int(bound["ub"])))
    return {
        "problem": str(data.get("name", path.stem)),
        "path": str(path),
        "nvars": len(variables),
        "nconstraints": len(data["A"]),
        "N": int(data.get("N", 2)),
        "candidate_state_space_size": math.prod(ub - lb + 1 for lb, ub in bounds),
    }


def monitor_peak_rss(proc: subprocess.Popen, stop: Event) -> dict[str, float]:
    peak = 0
    try:
        root_proc = psutil.Process(proc.pid)
    except psutil.Error:
        return {"peak": 0.0}
    while not stop.is_set():
        try:
            rss = root_proc.memory_info().rss
            for child in root_proc.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except psutil.Error:
                    pass
            peak = max(peak, rss)
        except psutil.Error:
            break
        time.sleep(0.01)
    return {"peak": peak / (1024 * 1024)}


def run_child(problem_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="qaoa_feasible_") as tmp:
        result_path = Path(tmp) / "result.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-problem",
            str(problem_path),
            "--worker-result",
            str(result_path),
            "--p",
            str(args.p),
            "--maxiter",
            str(args.maxiter),
            "--seed",
            str(args.seed),
            "--circuit-type",
            args.circuit_type,
        ]
        start = time.perf_counter()
        child_env = os.environ.copy()
        child_env["CVIP_QAOA_PROBLEM_NAME"] = problem_path.stem
        child_env["CVIP_QAOA_PROBLEM_PATH"] = str(problem_path)
        child_env.setdefault(
            "CVIP_QAOA_FIRST_FEASIBLE_CSV",
            str(Path(args.output_dir) / f"{args.output_prefix}_cvIP_qaoa_first_feasible.csv"),
        )
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=child_env)
        stop = Event()
        peak = {"peak": 0.0}
        thread = Thread(target=lambda: peak.update(monitor_peak_rss(proc, stop)), daemon=True)
        thread.start()
        try:
            stdout, stderr = proc.communicate(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            status = "timeout"
        else:
            status = "completed" if proc.returncode == 0 else "error"
        stop.set()
        thread.join(timeout=1)
        wall_sec = time.perf_counter() - start

        result: dict[str, Any] = {}
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                result = {}
        if not result:
            result = {
                "solver": "qaoa",
                "first_feasible_status": status,
                "time_to_first_feasible_sec": "",
                "first_feasible_objective": "",
                "first_feasible_solution": "",
                "error": (stderr or stdout).strip().splitlines()[-1] if (stderr or stdout).strip() else "",
            }
        if status == "timeout":
            result["first_feasible_status"] = "timeout"
        result["wall_sec"] = wall_sec
        result["peak_memory_mb"] = peak["peak"]
        return result


def dimension_limit_result(limit: int, state_space: int) -> dict[str, Any]:
    return {
        "solver": "qaoa",
        "first_feasible_status": "dimension_limit",
        "time_to_first_feasible_sec": "",
        "first_feasible_objective": "",
        "first_feasible_solution": "",
        "wall_sec": "",
        "peak_memory_mb": "",
        "error": f"candidate_state_space_size {state_space} exceeds max_state_space {limit}",
    }


def completed_paths(output: Path) -> set[str]:
    if not output.exists():
        return set()
    with output.open(newline="", encoding="utf-8") as f:
        return {row["path"] for row in csv.DictReader(f) if row.get("path")}


def collect_completed_paths(paths: list[str], include_output: Path | None) -> set[str]:
    done: set[str] = set()
    if include_output is not None:
        done.update(completed_paths(include_output))
    for raw_path in paths:
        done.update(completed_paths(Path(raw_path)))
    return done


def shard_files(files: list[Path], shard_count: int, shard_index: int) -> list[Path]:
    if shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < shard-count")
    if shard_count == 1:
        return files
    return [path for i, path in enumerate(files) if i % shard_count == shard_index]


def worker(problem_path: Path, result_path: Path, args: argparse.Namespace) -> int:
    start = time.perf_counter()
    result: dict[str, Any] = {"solver": "qaoa"}
    try:
        import numpy as np

        from cvIP.qaoa import BosonicQAOAIPSolver

        data = json.loads(problem_path.read_text(encoding="utf-8"))
        solver = BosonicQAOAIPSolver(
            data["A"],
            data["b"],
            data["c"],
            N=int(data.get("N", 2)),
            p=args.p,
            maxiter=args.maxiter,
            seed=args.seed,
            circuit_type=args.circuit_type,
        )
        elapsed = time.perf_counter() - start
        first_values = [int(v) for v in solver.feasible_states[0]]
        first_objective = float(np.dot(np.array(data["c"], dtype=float), np.array(first_values, dtype=float)))
        result.update(
            {
                "first_feasible_status": "feasible",
                "time_to_first_feasible_sec": elapsed,
                "first_feasible_objective": first_objective,
                "first_feasible_solution": json.dumps(dict(zip(data["variables"], first_values)), sort_keys=True),
                "error": "",
            }
        )
    except MemoryError as exc:
        result.update({"first_feasible_status": "memory_error", "time_to_first_feasible_sec": "", "first_feasible_objective": "", "first_feasible_solution": "", "error": str(exc)})
    except Exception as exc:
        status = f"error:{type(exc).__name__}"
        result.update({"first_feasible_status": status, "time_to_first_feasible_sec": "", "first_feasible_objective": "", "first_feasible_solution": "", "error": str(exc)})
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result.get("first_feasible_status") == "feasible" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-dir", default=str(DEFAULT_PROBLEM_DIR))
    parser.add_argument("--glob", default="nvars_*/*.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--per-nvars", type=int)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-prefix", default="qaoa_feasible_benchmark_results")
    parser.add_argument("--output", help="Deprecated: exact CSV path. Prefer --output-dir and --output-prefix.")
    parser.add_argument("--env-output", help="Write machine/GPU environment JSON. Defaults next to CSV.")
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--circuit-type", default="multi_beta", choices=["beta_gamma", "multi_beta", "multi_beta_oneH"])
    parser.add_argument("--max-state-space", type=int, help="Record dimension_limit instead of running QAOA above this candidate state-space size.")
    parser.add_argument("--resume", action="store_true", help="Append to an existing output CSV and skip paths already present.")
    parser.add_argument("--completed-csv", action="append", default=[], help="Existing CSV whose path column marks problems already completed. Can be passed multiple times.")
    parser.add_argument("--shard-count", type=int, default=1, help="Split unfinished problems into this many queues.")
    parser.add_argument("--shard-index", type=int, default=0, help="Run only this zero-based queue index.")
    parser.add_argument("--worker-problem")
    parser.add_argument("--worker-result")
    args = parser.parse_args()

    if args.worker_problem:
        if not args.worker_result:
            raise SystemExit("--worker-result is required with --worker-problem")
        return worker(Path(args.worker_problem), Path(args.worker_result), args)

    if args.timeout is not None and args.timeout <= 0:
        args.timeout = None

    output = Path(args.output) if args.output else Path(args.output_dir) / f"{args.output_prefix}_qaoa.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    env_path = Path(args.env_output) if args.env_output else output.parent / f"{output.stem}.environment.json"
    env_path.write_text(json.dumps(env_info(), indent=2), encoding="utf-8")
    files = select_problem_files(Path(args.problem_dir), args.glob, args.limit, args.per_nvars)
    done_paths = collect_completed_paths(args.completed_csv, output if args.resume else None)
    unfinished_files = [path for path in files if str(path) not in done_paths]
    files = shard_files(unfinished_files, args.shard_count, args.shard_index)

    fields = [
        "problem",
        "path",
        "nvars",
        "nconstraints",
        "N",
        "candidate_state_space_size",
        "solver",
        "first_feasible_status",
        "time_to_first_feasible_sec",
        "first_feasible_objective",
        "first_feasible_solution",
        "wall_sec",
        "peak_memory_mb",
        "error",
    ]
    mode = "a" if args.resume and output.exists() else "w"
    with output.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if mode == "w":
            writer.writeheader()
        print(
            f"Selected {len(files)} unfinished problems for shard "
            f"{args.shard_index + 1}/{args.shard_count} "
            f"({len(done_paths)} completed paths detected)."
        )
        for i, path in enumerate(files, 1):
            row = problem_metadata(path)
            if args.max_state_space is not None and row["candidate_state_space_size"] > args.max_state_space:
                row.update(dimension_limit_result(args.max_state_space, row["candidate_state_space_size"]))
            else:
                row.update(run_child(path, args))
            writer.writerow({key: row.get(key, "") for key in fields})
            f.flush()
            print(f"[{i}/{len(files)}] {row['problem']} {row['first_feasible_status']}")
    print(f"Wrote {output}")
    print(f"Wrote {env_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
