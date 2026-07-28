#!/usr/bin/env python3
"""Run Choco-Q CPU on MIPLIB-derived integer problems with 3-bit encoding."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import psutil


FIELDS = [
    "problem",
    "path",
    "n_int_vars",
    "n_binary_vars",
    "n_constraints",
    "provider",
    "max_iter",
    "shots",
    "status",
    "wall_sec",
    "peak_rss_mb",
    "classical_optimal_objective",
    "classical_optimal_solution",
    "choco_best_feasible_objective",
    "choco_best_feasible_solution",
    "choco_best_feasible_probability",
    "choco_raw_states",
    "choco_raw_probabilities",
    "iteration_count",
    "circuit_depth",
    "circuit_width",
    "circuit_culled_depth",
    "circuit_num_one_qubit_gates",
    "error",
]


def bit_name(var: str, bit: int) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in var)
    return f"{safe}_b{bit}"


def decode_state(state: list[int], variables: list[str]) -> dict[str, int]:
    out = {}
    for i, var in enumerate(variables):
        b0, b1, b2 = state[3 * i : 3 * i + 3]
        out[var] = int(b0 + 2 * b1 + 4 * b2)
    return out


def feasible(data: dict, decoded: dict[str, int]) -> bool:
    variables = data["variables"]
    values = [decoded[var] for var in variables]
    return all(sum(row[i] * values[i] for i in range(len(variables))) == rhs for row, rhs in zip(data["A"], data["b"]))


def objective(data: dict, decoded: dict[str, int]) -> float:
    return float(sum(coef * decoded[var] for coef, var in zip(data["c"], data["variables"])))


def worker(problem_path: str, max_iter: int, shots: int, provider_name: str, out_q: mp.Queue) -> None:
    start = time.perf_counter()
    try:
        from chocoq.model import LinearConstrainedBinaryOptimization as LcboModel
        from chocoq.solvers.optimizers import CobylaOptimizer
        from chocoq.solvers.qiskit import AerProvider, ChocoSolver, DdsimProvider

        path = Path(problem_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        variables = data["variables"]

        print(f"[problem] {data['name']} path={path}", flush=True)
        print(f"[input] n_int={len(variables)} n_bin={3 * len(variables)} n_constr={len(data['A'])}", flush=True)
        print(f"[input] A={data['A']} b={data['b']} c={data['c']}", flush=True)

        m = LcboModel()
        bit_vars = {}
        for var in variables:
            bit_vars[var] = [m.addVar(name=bit_name(var, bit)) for bit in range(3)]

        print("[encoding]", flush=True)
        for var in variables:
            bits = bit_vars[var]
            print(f"  {var} = {bits[0].name} + 2*{bits[1].name} + 4*{bits[2].name}", flush=True)

        obj_expr = 0
        for coeff, var in zip(data["c"], variables):
            bits = bit_vars[var]
            obj_expr += int(coeff) * (bits[0] + 2 * bits[1] + 4 * bits[2])
        m.setObjective(obj_expr, "max")

        expanded_rows = []
        for row, rhs in zip(data["A"], data["b"]):
            lhs = 0
            expanded = []
            for coeff, var in zip(row, variables):
                bits = bit_vars[var]
                coeff = int(coeff)
                lhs += coeff * (bits[0] + 2 * bits[1] + 4 * bits[2])
                expanded.extend([coeff, 2 * coeff, 4 * coeff])
            m.addConstr(lhs == int(rhs))
            expanded_rows.append(expanded)
        print(f"[expanded] A_binary={expanded_rows} b={data['b']}", flush=True)
        print(f"[expanded] lin_constr_mtx={m.lin_constr_mtx.tolist()}", flush=True)

        classical_obj, classical_solution_bits = m.optimize()
        m._best_cost = classical_obj
        classical_state = [int(round(classical_solution_bits[bit_name(var, bit)])) for var in variables for bit in range(3)]
        classical_solution = decode_state(classical_state, variables)
        print(f"[classical] objective={classical_obj} solution={classical_solution}", flush=True)

        provider = DdsimProvider() if provider_name == "ddsim" else AerProvider()
        solver = ChocoSolver(
            prb_model=m,
            optimizer=CobylaOptimizer(max_iter=max_iter),
            provider=provider,
            num_layers=1,
            shots=shots,
        )
        metrics = solver.circuit_analyze(["depth", "width", "culled_depth", "num_one_qubit_gates"])
        print(f"[circuit] metrics={metrics}", flush=True)
        print("[solve] start", flush=True)
        states, probs, iter_count = solver.solve()
        print(f"[solve] states={states}", flush=True)
        print(f"[solve] probs={probs}", flush=True)

        best_obj = None
        best_sol = None
        best_prob = None
        for state, prob in zip(states, probs):
            decoded = decode_state([int(x) for x in state], variables)
            if feasible(data, decoded):
                obj = objective(data, decoded)
                if best_obj is None or obj > best_obj:
                    best_obj = obj
                    best_sol = decoded
                    best_prob = float(prob)
        print(f"[choco-best] objective={best_obj} probability={best_prob} solution={best_sol}", flush=True)

        out_q.put(
            {
                "problem": data["name"],
                "path": str(path),
                "n_int_vars": len(variables),
                "n_binary_vars": 3 * len(variables),
                "n_constraints": len(data["A"]),
                "provider": provider_name,
                "max_iter": max_iter,
                "shots": shots,
                "status": "ok",
                "wall_sec": time.perf_counter() - start,
                "classical_optimal_objective": classical_obj,
                "classical_optimal_solution": json.dumps(classical_solution, sort_keys=True),
                "choco_best_feasible_objective": best_obj if best_obj is not None else "",
                "choco_best_feasible_solution": json.dumps(best_sol, sort_keys=True) if best_sol is not None else "",
                "choco_best_feasible_probability": best_prob if best_prob is not None else "",
                "choco_raw_states": json.dumps(states),
                "choco_raw_probabilities": json.dumps(probs),
                "iteration_count": iter_count,
                "circuit_depth": metrics[0],
                "circuit_width": metrics[1],
                "circuit_culled_depth": metrics[2],
                "circuit_num_one_qubit_gates": metrics[3],
                "error": "",
            }
        )
    except Exception:
        out_q.put(
            {
                "problem": "",
                "path": problem_path,
                "status": "error",
                "wall_sec": time.perf_counter() - start,
                "error": traceback.format_exc(limit=20),
            }
        )


def run_one(path: Path, max_iter: int, shots: int, provider: str, timeout: float) -> dict:
    q: mp.Queue = mp.Queue()
    proc = mp.Process(target=worker, args=(str(path), max_iter, shots, provider, q))
    proc.start()
    ps_proc = psutil.Process(proc.pid)
    peak = 0
    start = time.perf_counter()
    while proc.is_alive():
        try:
            peak = max(peak, ps_proc.memory_info().rss)
            for child in ps_proc.children(recursive=True):
                try:
                    peak = max(peak, child.memory_info().rss)
                except psutil.Error:
                    pass
        except psutil.Error:
            pass
        if timeout and timeout > 0 and time.perf_counter() - start > timeout:
            proc.kill()
            proc.join(5)
            return {"path": str(path), "status": "timeout", "wall_sec": time.perf_counter() - start, "peak_rss_mb": peak / (1024 * 1024)}
        time.sleep(0.2)
    proc.join()
    try:
        row = q.get_nowait()
    except queue.Empty:
        row = {"path": str(path), "status": "error", "wall_sec": time.perf_counter() - start, "error": "worker exited without result"}
    row["peak_rss_mb"] = max(float(row.get("peak_rss_mb") or 0), peak / (1024 * 1024))
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem-dir", default="data/miplib_benders/qaoa_problems")
    ap.add_argument("--glob", default="nvars_*/*.json")
    ap.add_argument("--output", required=True)
    ap.add_argument("--provider", choices=["ddsim", "aer"], default="ddsim")
    ap.add_argument("--max-iter", type=int, default=30)
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--timeout-sec", type=float, default=300.0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()

    files = sorted(Path(args.problem_dir).glob(args.glob))
    if args.limit:
        files = files[: args.limit]
    done = set()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        with output.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("path"):
                    done.add(row["path"])

    write_header = not output.exists()
    pending = [path for path in files if str(path) not in done]

    with output.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
            f.flush()
        if args.jobs <= 1:
            for idx, path in enumerate(files, 1):
                if str(path) in done:
                    print(f"[{idx}/{len(files)}] skip done {path}", flush=True)
                    continue
                print(f"[{idx}/{len(files)}] run {path}", flush=True)
                row = run_one(path, args.max_iter, args.shots, args.provider, args.timeout_sec)
                writer.writerow({key: row.get(key, "") for key in FIELDS})
                f.flush()
                print(
                    f"[{idx}/{len(files)}] status={row.get('status')} wall={row.get('wall_sec')} "
                    f"peak_mb={row.get('peak_rss_mb')} choco_obj={row.get('choco_best_feasible_objective')}",
                    flush=True,
                )
        else:
            print(f"[parallel] total={len(files)} done={len(done)} pending={len(pending)} jobs={args.jobs}", flush=True)
            with ProcessPoolExecutor(max_workers=args.jobs) as ex:
                future_to_path = {
                    ex.submit(run_one, path, args.max_iter, args.shots, args.provider, args.timeout_sec): path
                    for path in pending
                }
                completed = 0
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    completed += 1
                    try:
                        row = future.result()
                    except Exception:
                        row = {
                            "path": str(path),
                            "status": "error",
                            "error": traceback.format_exc(limit=20),
                        }
                    writer.writerow({key: row.get(key, "") for key in FIELDS})
                    f.flush()
                    print(
                        f"[parallel {completed}/{len(pending)}] path={path} status={row.get('status')} "
                        f"wall={row.get('wall_sec')} peak_mb={row.get('peak_rss_mb')} "
                        f"choco_obj={row.get('choco_best_feasible_objective')}",
                        flush=True,
                    )
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn")
    raise SystemExit(main())
