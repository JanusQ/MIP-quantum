#!/usr/bin/env python3
"""Build small exact MILP benchmarks from MIPLIB 2017 benchmark instances.

The downsampling is deterministic: select finite-domain integer variables from
an official MPS instance, fix all other variables to zero, keep constraints with
nonzero selected coefficients, integerize coefficients row-wise, and write a
small derived MPS plus metadata.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from collections import defaultdict, deque
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT / "data" / "miplib_benders"
RAW = BENCHMARK_ROOT / "raw"
OUT = BENCHMARK_ROOT / "processed"
META = BENCHMARK_ROOT / "metadata"

BENCHMARK_ZIP_URL = "https://miplib.zib.de/downloads/benchmark.zip"
BENCHMARK_TEST_URL = "https://miplib.zib.de/downloads/benchmark-v2.test"


@dataclass
class Row:
    name: str
    sense: str
    rhs: float = 0.0
    coeffs: dict[str, float] = field(default_factory=dict)


@dataclass
class Var:
    name: str
    lb: float = 0.0
    ub: float = math.inf
    kind: str = "C"
    obj: float = 0.0


@dataclass
class Model:
    name: str
    obj_name: str | None
    rows: dict[str, Row]
    vars: dict[str, Var]
    obj_sense: str = "MIN"


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    with urllib.request.urlopen(url) as src, tmp.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    tmp.replace(path)


def tokens(line: str) -> list[str]:
    return line.strip().split()


def parse_mps_bytes(blob: bytes, name: str) -> Model:
    text = gzip.decompress(blob).decode("utf-8", errors="replace") if name.endswith(".gz") else blob.decode("utf-8", errors="replace")
    rows: dict[str, Row] = {}
    vars_: dict[str, Var] = {}
    obj_name = None
    section = None
    integer_mode = False
    ranges: dict[str, float] = {}
    obj_sense = "MIN"

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("*"):
            continue
        head = line[:14].strip().upper()
        if head in {"NAME", "ROWS", "COLUMNS", "RHS", "RANGES", "BOUNDS", "ENDATA"}:
            section = head
            if head == "ENDATA":
                break
            continue
        if head == "OBJSENSE":
            section = "OBJSENSE"
            continue
        if section == "OBJSENSE":
            val = line.strip().upper()
            if val in {"MIN", "MAX"}:
                obj_sense = val
            continue

        t = tokens(line)
        if not t:
            continue

        if section == "ROWS":
            sense, row_name = t[0].upper(), t[1]
            if sense == "N" and obj_name is None:
                obj_name = row_name
            else:
                rows[row_name] = Row(row_name, sense)

        elif section == "COLUMNS":
            if len(t) >= 3 and t[1].upper() == "'MARKER'":
                marker = t[2].upper().strip("'")
                integer_mode = marker == "INTORG"
                continue
            var_name = t[0]
            var = vars_.setdefault(var_name, Var(var_name))
            if integer_mode and var.kind == "C":
                var.kind = "I"
            for i in range(1, len(t) - 1, 2):
                row_name, value = t[i], float(t[i + 1])
                if row_name == obj_name:
                    var.obj += value
                elif row_name in rows:
                    rows[row_name].coeffs[var_name] = rows[row_name].coeffs.get(var_name, 0.0) + value

        elif section == "RHS":
            for i in range(1, len(t) - 1, 2):
                row_name, value = t[i], float(t[i + 1])
                if row_name in rows:
                    rows[row_name].rhs = value

        elif section == "RANGES":
            for i in range(1, len(t) - 1, 2):
                ranges[t[i]] = float(t[i + 1])

        elif section == "BOUNDS":
            btype = t[0].upper()
            var_name = t[2]
            val = float(t[3]) if len(t) > 3 else None
            var = vars_.setdefault(var_name, Var(var_name))
            if btype == "LO":
                var.lb = float(val)
            elif btype == "UP":
                var.ub = float(val)
            elif btype == "FX":
                var.lb = var.ub = float(val)
            elif btype == "FR":
                var.lb, var.ub = -math.inf, math.inf
            elif btype == "MI":
                var.lb = -math.inf
            elif btype == "PL":
                var.ub = math.inf
            elif btype == "BV":
                var.lb, var.ub, var.kind = 0.0, 1.0, "B"
            elif btype == "LI":
                var.lb, var.kind = float(val), "I"
            elif btype == "UI":
                var.ub, var.kind = float(val), "I"

    # Convert ranged rows into explicit upper/lower rows before sampling.
    for row_name, rng in list(ranges.items()):
        if row_name not in rows:
            continue
        row = rows[row_name]
        if row.sense == "E":
            lo, hi = (row.rhs, row.rhs + rng) if rng >= 0 else (row.rhs + rng, row.rhs)
            row.sense, row.rhs = "L", hi
            rows[row_name + "__range_lo"] = Row(row_name + "__range_lo", "G", lo, dict(row.coeffs))
        elif row.sense == "L":
            row.rhs = row.rhs + abs(rng) if rng >= 0 else row.rhs
            rows[row_name + "__range_lo"] = Row(row_name + "__range_lo", "G", row.rhs - abs(rng), dict(row.coeffs))
        elif row.sense == "G":
            row.rhs = row.rhs + abs(rng) if rng < 0 else row.rhs
            rows[row_name + "__range_hi"] = Row(row_name + "__range_hi", "L", row.rhs + abs(rng), dict(row.coeffs))

    return Model(Path(name).name.replace(".mps.gz", "").replace(".mps", ""), obj_name, rows, vars_, obj_sense)


def finite_integral_domain(v: Var, max_domain: int) -> bool:
    if v.kind not in {"B", "I"}:
        return False
    if not math.isfinite(v.lb) or not math.isfinite(v.ub):
        return False
    if abs(v.lb - round(v.lb)) > 1e-9 or abs(v.ub - round(v.ub)) > 1e-9:
        return False
    return 1 <= int(round(v.ub - v.lb + 1)) <= max_domain


def choose_vars(model: Model, max_vars: int, max_domain: int) -> list[str]:
    candidates = [v for v in model.vars.values() if finite_integral_domain(v, max_domain)]
    degree = defaultdict(int)
    for row in model.rows.values():
        for name, coef in row.coeffs.items():
            if abs(coef) > 1e-12:
                degree[name] += 1
    candidates.sort(
        key=lambda v: (
            int(round(v.ub - v.lb + 1)) <= 2,
            -int(round(v.ub - v.lb + 1)),
            -degree[v.name],
            v.name,
        )
    )
    return [v.name for v in candidates[:max_vars]]


def lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b) if a and b else max(abs(a), abs(b))


def intize(values: list[float], max_den: int, max_scale: int) -> tuple[list[int], int, float]:
    fracs = [Fraction(x).limit_denominator(max_den) for x in values]
    scale = 1
    for f in fracs:
        scale = lcm(scale, f.denominator)
        if scale > max_scale:
            scale = max_scale
            break
    ints = [int(round(x * scale)) for x in values]
    err = max((abs(i / scale - x) for i, x in zip(ints, values)), default=0.0)
    return ints, scale, err


def downsample(model: Model, max_vars: int, max_domain: int, max_rows: int, max_den: int, max_scale: int) -> tuple[Model, dict]:
    selected = choose_vars(model, max_vars, max_domain)
    selected_set = set(selected)
    rows = []
    for row in model.rows.values():
        coeffs = {v: c for v, c in row.coeffs.items() if v in selected_set and abs(c) > 1e-12}
        if coeffs:
            rows.append(Row(row.name, row.sense, row.rhs, coeffs))
    rows.sort(key=lambda r: (-len(r.coeffs), r.name))
    rows = rows[:max_rows]

    new_vars = {name: Var(name, model.vars[name].lb, model.vars[name].ub, model.vars[name].kind, model.vars[name].obj) for name in selected}
    new_rows: dict[str, Row] = {}
    max_error = 0.0
    scales = {}
    for row in rows:
        names = sorted(row.coeffs)
        vals = [row.coeffs[n] for n in names] + [row.rhs]
        ints, scale, err = intize(vals, max_den, max_scale)
        max_error = max(max_error, err)
        scales[row.name] = scale
        coeffs = {n: ints[i] for i, n in enumerate(names) if ints[i] != 0}
        if coeffs:
            new_rows[row.name] = Row(row.name, row.sense, ints[-1], coeffs)

    obj_vals = [new_vars[n].obj for n in selected]
    obj_ints, obj_scale, obj_err = intize(obj_vals, max_den, max_scale)
    max_error = max(max_error, obj_err)
    for i, name in enumerate(selected):
        new_vars[name].obj = obj_ints[i]

    sampled = Model(model.name + "_ds", "OBJ", new_rows, new_vars, model.obj_sense)
    feasible_states, candidate_states = count_feasible_states(sampled)
    meta = {
        "source_instance": model.name,
        "downsampled_instance": sampled.name,
        "procedure": "finite-domain integer variable subset; all unselected variables fixed to 0; retain constraints with selected nonzero coefficients; row-wise rational approximation and integer scaling",
        "n_variables": len(new_vars),
        "n_constraints": len(new_rows),
        "n_binary": sum(1 for v in new_vars.values() if int(v.lb) == 0 and int(v.ub) == 1),
        "n_integer": len(new_vars),
        "variable_bounds": {n: {"lb": int(round(v.lb)), "ub": int(round(v.ub)), "type": v.kind} for n, v in new_vars.items()},
        "candidate_state_space_size": candidate_states,
        "number_of_feasible_states": feasible_states,
        "feasible_state_count_exact": feasible_states is not None,
        "integerization": {"row_scales": scales, "objective_scale": obj_scale, "max_abs_approx_error": max_error},
    }
    meta["benders_blocks"] = benders_blocks(sampled)
    return sampled, meta


def benders_blocks(model: Model, master_fraction: float = 0.2) -> dict:
    degree = defaultdict(int)
    for row in model.rows.values():
        for v in row.coeffs:
            degree[v] += 1
    n_master = max(1, min(len(model.vars), round(len(model.vars) * master_fraction))) if model.vars else 0
    master = set(sorted(model.vars, key=lambda v: (-degree[v], v))[:n_master])
    graph = defaultdict(set)
    row_vars = {}
    for row in model.rows.values():
        local = [v for v in row.coeffs if v not in master]
        row_vars[row.name] = local
        for i, a in enumerate(local):
            for b in local[i + 1:]:
                graph[a].add(b)
                graph[b].add(a)
    seen = set()
    comps = []
    for v in sorted(set(model.vars) - master):
        if v in seen:
            continue
        q, comp = deque([v]), []
        seen.add(v)
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nb in graph[cur]:
                if nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        rows = [r for r, vs in row_vars.items() if any(x in comp for x in vs)]
        comps.append({"id": len(comps), "variables": sorted(comp), "constraints": sorted(rows)})
    return {"master_variables": sorted(master), "subproblems": comps}


def row_satisfied(lhs: int | float, sense: str, rhs: int | float) -> bool:
    if sense == "L":
        return lhs <= rhs + 1e-9
    if sense == "G":
        return lhs >= rhs - 1e-9
    if sense == "E":
        return abs(lhs - rhs) <= 1e-9
    raise ValueError(f"unsupported row sense: {sense}")


def count_feasible_states(model: Model, cap: int = 1_000_000) -> tuple[int | None, int]:
    domains = []
    candidate_states = 1
    for var in model.vars.values():
        lo, hi = int(round(var.lb)), int(round(var.ub))
        dom_size = hi - lo + 1
        candidate_states *= dom_size
        domains.append(range(lo, hi + 1))
    if candidate_states > cap:
        return None, candidate_states
    names = list(model.vars)
    feasible = 0
    for values in product(*domains):
        sol = dict(zip(names, values))
        ok = True
        for row in model.rows.values():
            lhs = sum(coef * sol[var] for var, coef in row.coeffs.items())
            if not row_satisfied(lhs, row.sense, row.rhs):
                ok = False
                break
        feasible += int(ok)
    return feasible, candidate_states


def write_mps(model: Model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(f"NAME          {model.name}\n")
        f.write("OBJSENSE\n")
        f.write(f" {model.obj_sense}\n")
        f.write("ROWS\n")
        f.write(" N  OBJ\n")
        for row in model.rows.values():
            f.write(f" {row.sense}  {row.name}\n")
        f.write("COLUMNS\n")
        f.write("    MARK0000  'MARKER'                 'INTORG'\n")
        for var in model.vars.values():
            entries = [("OBJ", var.obj)] + [(r.name, r.coeffs[var.name]) for r in model.rows.values() if var.name in r.coeffs]
            for i in range(0, len(entries), 2):
                chunk = entries[i : i + 2]
                parts = [f"    {var.name:<16}"]
                for rn, val in chunk:
                    parts.append(f"{rn:<16} {val:>16g}")
                f.write(" ".join(parts) + "\n")
        f.write("    MARK0001  'MARKER'                 'INTEND'\n")
        f.write("RHS\n")
        for row in model.rows.values():
            f.write(f"    RHS1              {row.name:<16} {row.rhs:>16g}\n")
        f.write("BOUNDS\n")
        for var in model.vars.values():
            if int(var.lb) == 0 and int(var.ub) == 1:
                f.write(f" BV BND1              {var.name}\n")
            else:
                f.write(f" LI BND1              {var.name:<16} {int(var.lb):>16d}\n")
                f.write(f" UI BND1              {var.name:<16} {int(var.ub):>16d}\n")
        f.write("ENDATA\n")


def iter_zip_instances(zip_path: Path, names: set[str]):
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            base = Path(info.filename).name
            if base in names:
                yield base, zf.read(info)


def run_gurobi_check(mps_path: Path, timelimit: int) -> dict:
    if shutil.which("gurobi_cl") is None:
        return {"available": False}
    res = subprocess.run(
        ["gurobi_cl", f"TimeLimit={timelimit}", f"LogFile={mps_path.with_suffix('.gurobi.log')}", str(mps_path)],
        cwd=mps_path.parent,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timelimit + 20,
    )
    return {"available": True, "returncode": res.returncode, "log": str(mps_path.with_suffix(".gurobi.log"))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="download official benchmark.zip")
    ap.add_argument("--instances", type=int, default=8, help="number of derived instances to create")
    ap.add_argument("--max-vars", type=int, default=18)
    ap.add_argument("--max-domain", type=int, default=8)
    ap.add_argument("--max-rows", type=int, default=40)
    ap.add_argument("--max-denominator", type=int, default=1000)
    ap.add_argument("--max-scale", type=int, default=1000000)
    ap.add_argument("--gurobi-check", action="store_true")
    ap.add_argument("--gurobi-timelimit", type=int, default=30)
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    META.mkdir(parents=True, exist_ok=True)
    download(BENCHMARK_TEST_URL, RAW / "benchmark-v2.test")
    if args.download:
        download(BENCHMARK_ZIP_URL, RAW / "benchmark.zip")

    manifest = {
        "source": {
            "miplib_benchmark_page": "https://miplib.zib.de/tag_benchmark.html",
            "benchmark_zip": BENCHMARK_ZIP_URL,
            "benchmark_test": BENCHMARK_TEST_URL,
        },
        "parameters": vars(args),
        "instances": [],
    }

    zip_path = RAW / "benchmark.zip"
    if not zip_path.exists():
        print(f"metadata downloaded to {RAW / 'benchmark-v2.test'}")
        print("run again with --download to fetch benchmark.zip and build processed instances")
        (META / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return 0

    names = [x.strip() for x in (RAW / "benchmark-v2.test").read_text().splitlines() if x.strip()]
    wanted = set(names)
    made = 0
    for base, blob in iter_zip_instances(zip_path, wanted):
        try:
            model = parse_mps_bytes(blob, base)
            sampled, meta = downsample(model, args.max_vars, args.max_domain, args.max_rows, args.max_denominator, args.max_scale)
            if meta["n_variables"] == 0 or meta["n_constraints"] == 0:
                continue
            inst_dir = OUT / sampled.name
            write_mps(sampled, inst_dir / f"{sampled.name}.mps")
            (inst_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            if args.gurobi_check:
                meta["gurobi_check"] = run_gurobi_check(inst_dir / f"{sampled.name}.mps", args.gurobi_timelimit)
                (inst_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            manifest["instances"].append(meta)
            made += 1
            print(f"created {sampled.name}: vars={meta['n_variables']} rows={meta['n_constraints']} states={meta['number_of_feasible_states']}")
            if made >= args.instances:
                break
        except Exception as exc:
            print(f"skip {base}: {exc}", file=sys.stderr)
    (META / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {made} processed instances under {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
