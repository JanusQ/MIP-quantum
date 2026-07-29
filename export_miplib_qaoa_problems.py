#!/usr/bin/env python3
"""Export many QAOA-ready equality IP subproblems from MIPLIB-derived MPS files.

This is intentionally an export-only script: it does not import or run
cvIP/qaoa.py. Each output JSON contains A, b, c, N and provenance, so it can be
loaded later by a runner that calls BosonicQAOAIPSolver(A, b, c, ...).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, field
from collections import deque
from itertools import combinations, product
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "miplib_benders"
DEFAULT_MIP_DIR = DEFAULT_DATA_DIR / "processed"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "qaoa_problems"


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
    benders_subproblems: list[dict] = field(default_factory=list)


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


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

    metadata_path = path.parent / "metadata.json"
    benders_subproblems = []
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        benders_subproblems = metadata.get("benders_blocks", {}).get("subproblems", [])

    return MPSModel(path.stem, rows, vars_, benders_subproblems)


def bounded_integer_variables(model: MPSModel, fock_N: int) -> list[str]:
    return [
        name
        for name, var in model.vars.items()
        if var.lb <= var.ub and (var.ub - var.lb + 1) <= fock_N
    ]


def count_feasible(A: list[list[int]], b: list[int], variables: list[str], bounds: dict[str, dict[str, int]]) -> int:
    max_states = 200000
    counts = {tuple(0 for _ in A): 1}
    for i, var in enumerate(variables):
        next_counts = {}
        for partial, partial_count in counts.items():
            for value in range(bounds[var]["lb"], bounds[var]["ub"] + 1):
                new_sum = tuple(partial[j] + A[j][i] * value for j in range(len(A)))
                next_counts[new_sum] = next_counts.get(new_sum, 0) + partial_count
                if len(next_counts) > max_states:
                    return -1
        counts = next_counts
    return counts.get(tuple(b), 0)


def candidate_state_space_size(variables: list[str], bounds: dict[str, dict[str, int]]) -> int:
    size = 1
    for var in variables:
        size *= bounds[var]["ub"] - bounds[var]["lb"] + 1
    return size


def passes_hardness_filters(A: list[list[int]], b: list[int], source_var_count: int, min_nonzeros_per_row: int) -> bool:
    if all(rhs == 0 for rhs in b):
        return False
    for row in A:
        if sum(1 for value in row[:source_var_count] if value != 0) < min_nonzeros_per_row:
            return False
    return True


def is_bipartite_connected(A: list[list[int]]) -> bool:
    if not A:
        return False
    nrows = len(A)
    nvars = len(A[0])
    row_to_vars = [set(i for i, value in enumerate(row) if value != 0) for row in A]
    if any(not s for s in row_to_vars):
        return False
    var_to_rows = [set() for _ in range(nvars)]
    for r, vars_ in enumerate(row_to_vars):
        for v in vars_:
            var_to_rows[v].add(r)
    if any(not rs for rs in var_to_rows):
        return False
    start = ("r", 0)
    seen = {start}
    q = deque([start])
    while q:
        kind, idx = q.popleft()
        if kind == "r":
            neighbors = [("v", v) for v in row_to_vars[idx]]
        else:
            neighbors = [("r", r) for r in var_to_rows[idx]]
        for nb in neighbors:
            if nb not in seen:
                seen.add(nb)
                q.append(nb)
    return len(seen) == nrows + nvars


def count_primary_feasible(coeffs: list[int], rhs: int, selected: list[str], model: MPSModel) -> int:
    counts = {0: 1}
    for coef, var in zip(coeffs, selected):
        next_counts = {}
        for partial, partial_count in counts.items():
            for value in range(model.vars[var].lb, model.vars[var].ub + 1):
                new_sum = partial + coef * value
                next_counts[new_sum] = next_counts.get(new_sum, 0) + partial_count
        counts = next_counts
    return counts.get(rhs, 0)


def count_primary_with_row_slack(coeffs: list[int], rhs: int, selected: list[str], model: MPSModel, fock_N: int) -> int:
    counts = {0: 1}
    for coef, var in zip(coeffs, selected):
        next_counts = {}
        for partial, partial_count in counts.items():
            for value in range(model.vars[var].lb, model.vars[var].ub + 1):
                new_sum = partial + coef * value
                next_counts[new_sum] = next_counts.get(new_sum, 0) + partial_count
        counts = next_counts
    return sum(count for total, count in counts.items() if 0 <= rhs - total < fock_N)


def objective_for(model: MPSModel, variables: list[str]) -> list[int]:
    c = [int(model.vars[var].obj) for var in variables]
    if all(v == 0 for v in c):
        c = [i + 1 for i in range(len(variables))]
    return c


def domain_size(model: MPSModel, name: str) -> int:
    return model.vars[name].ub - model.vars[name].lb + 1


def nonbinary_source_var_count(model: MPSModel, fock_N: int) -> int:
    return sum(
        1
        for name in bounded_integer_variables(model, fock_N)
        if domain_size(model, name) > 2
    )


def primitive_signature(values: list[int], rhs: int) -> tuple[int, ...]:
    data = list(values) + [rhs]
    gcd = 0
    for value in data:
        gcd = abs(value) if gcd == 0 else __import__("math").gcd(gcd, abs(value))
    if gcd > 1:
        data = [value // gcd for value in data]
    first = next((value for value in data if value != 0), 0)
    if first < 0:
        data = [-value for value in data]
    return tuple(data)


def diverse_constraints(A: list[list[int]], b: list[int], source_var_count: int) -> bool:
    full_signatures = set()
    source_signatures = set()
    supports = set()
    for row, rhs in zip(A, b):
        source_coeffs = row[:source_var_count]
        if not any(source_coeffs):
            return False
        full_sig = primitive_signature(row, rhs)
        source_sig = primitive_signature(source_coeffs, rhs)
        support = tuple(i for i, value in enumerate(source_coeffs) if value != 0)
        if full_sig in full_signatures or source_sig in source_signatures or support in supports:
            return False
        full_signatures.add(full_sig)
        source_signatures.add(source_sig)
        supports.add(support)
    return True


def row_problem_candidates(
    model: MPSModel,
    modes: int,
    combos_per_row: int,
    fock_N: int,
    only_subproblem_id=None,
    row_groups_per_block: int = 20,
    args=None,
):
    int_vars = bounded_integer_variables(model, fock_N)
    if len(int_vars) < modes:
        return

    filler = sorted(int_vars, key=lambda name: (-domain_size(model, name), name))
    for subproblem in model.benders_subproblems:
        if only_subproblem_id is not None and subproblem.get("id") != only_subproblem_id:
            continue
        block_vars = [v for v in subproblem.get("variables", []) if v in int_vars]
        block_var_set = set(block_vars)
        if not block_vars:
            continue
        candidate_rows = [
            model.rows[name]
            for name in subproblem.get("constraints", [])
            if name in model.rows and model.rows[name].sense in {"E", "L", "G"}
            and any(v in block_var_set and model.rows[name].coeffs.get(v, 0) != 0 for v in block_vars)
        ]
        candidate_rows.sort(key=lambda row: (row.sense != "E", row.name, -len(row.coeffs)))
        row_groups = []
        for group_size in range(min(3, len(candidate_rows)), 0, -1):
            for start in range(max(1, len(candidate_rows) - group_size + 1)):
                row_groups.append(tuple(candidate_rows[start : start + group_size]))
                if len(row_groups) >= row_groups_per_block:
                    break
            if len(row_groups) >= row_groups_per_block:
                break

        local_filler = sorted(block_vars, key=lambda name: (-domain_size(model, name), name))

        for rows in row_groups:
            yield from benders_row_group_candidates(
                model,
                subproblem,
                rows,
                local_filler,
                modes,
                combos_per_row,
                fock_N,
                args,
            )


def benders_row_group_candidates(
    model: MPSModel,
    subproblem: dict,
    rows: tuple[Row, ...],
    filler: list[str],
    modes: int,
    combos_per_row: int,
    fock_N: int,
    args,
):
        slack_count = sum(1 for row in rows if row.sense in {"L", "G"})
        variable_slots = modes - slack_count
        if variable_slots < 1:
            return
        active_set = set()
        for row in rows:
            active_set.update(v for v in row.coeffs if v in filler and row.coeffs[v] != 0)
        active = sorted(
            active_set,
            key=lambda name: (-domain_size(model, name), name),
        )
        if not active:
            return

        emitted_for_row = 0
        if len(active) >= variable_slots:
            starts = range(len(active) - variable_slots + 1)
        else:
            starts = range(1)
        seen = set()
        for start in starts:
            if emitted_for_row >= combos_per_row:
                break
            selected = list(active[start : start + variable_slots])
            selected.extend(v for v in filler if v not in selected)
            selected = selected[:variable_slots]
            if len(selected) != variable_slots:
                continue
            key = tuple(selected)
            if key in seen:
                continue
            seen.add(key)

            variables = list(selected)
            c = objective_for(model, selected)
            variable_bounds = {var: {"lb": 0, "ub": fock_N - 1} for var in variables}
            A = []
            b = []
            slack_variables = []
            for row in rows:
                coeffs = [int(row.coeffs.get(var, 0)) for var in selected]
                rhs = int(row.rhs)
                if row.sense == "G":
                    coeffs = [-v for v in coeffs]
                    rhs = -rhs
                coeffs.extend([0] * (len(variables) - len(selected)))
                if row.sense in {"L", "G"}:
                    slack_name = f"row_slack_{safe_filename(row.name)}"
                    for existing_row in A:
                        existing_row.append(0)
                    variables.append(slack_name)
                    c.append(0)
                    variable_bounds[slack_name] = {"lb": 0, "ub": fock_N - 1}
                    slack_variables.append(
                        {
                            "name": slack_name,
                            "source_row": row.name,
                            "type": "row_slack_for_le" if row.sense == "L" else "row_slack_for_ge",
                        }
                    )
                    coeffs.append(1)
                while len(coeffs) < len(variables):
                    coeffs.append(0)
                A.append(coeffs)
                b.append(rhs)

            if len(variables) != modes:
                continue
            if len(A) > 1 and not diverse_constraints(A, b, len(selected)):
                continue
            min_nonzeros = getattr(args, "min_nonzeros_per_row", 3) if args is not None else 3
            if not passes_hardness_filters(A, b, len(selected), min_nonzeros):
                continue
            if not is_bipartite_connected(A):
                continue

            state_space = candidate_state_space_size(variables, variable_bounds)
            max_candidate_space = getattr(args, "max_candidate_state_space", 300000) if args is not None else 300000
            if state_space > max_candidate_space:
                continue
            feasible_count = count_feasible(A, b, variables, variable_bounds)
            min_feasible = getattr(args, "min_feasible_states", 10) if args is not None else 10
            max_feasible = getattr(args, "max_feasible_states", 200000) if args is not None else 200000
            if feasible_count < min_feasible or feasible_count > max_feasible:
                continue

            row_part = "_".join(safe_filename(row.name) for row in rows[:3])
            name = f"{model.name}_qaoa_{row_part}_{len(seen):04d}"
            emitted_for_row += 1
            problem = {
                "name": name,
                "source_mps": model.name,
                "decomposition": "benders",
                "benders_subproblem_id": subproblem.get("id"),
                "benders_subproblem_variables": subproblem.get("variables", []),
                "benders_subproblem_constraints": subproblem.get("constraints", []),
                "source_rows": [{"name": row.name, "sense": row.sense} for row in rows],
                "variables": variables,
                "A": A,
                "b": b,
                "c": c,
                "N": fock_N,
                "variable_bounds": variable_bounds,
                "encoding": "Exported benchmark variables are normalized experimental variables with domain 0..N-1. Coefficients and row provenance come from the source MIP row.",
                "candidate_state_space_size": state_space,
                "feasible_states_in_qaoa_truncation": feasible_count,
                "feasible_state_count_exact": True,
                "adapter": "strict Benders subproblem export; variables and source rows are selected only from one benders_blocks.subproblems entry; exported variables use domain 0..N-1; <=/>= rows use one nonnegative row slack each",
            }
            if slack_variables:
                problem["slack_variables"] = slack_variables
            yield problem


def export_problem(problem: dict, output_dir: Path) -> Path:
    filename = safe_filename(problem["name"]) + ".json"
    path = output_dir / filename
    path.write_text(json.dumps(problem, indent=2), encoding="utf-8")
    return path


def problem_signature(problem: dict, include_source: bool = False) -> str:
    payload = {
        "N": problem["N"],
        "n_variables": len(problem["variables"]),
        "A": problem["A"],
        "b": problem["b"],
        "c": problem["c"],
        "variable_bounds": [problem["variable_bounds"][var] for var in problem["variables"]],
    }
    if include_source:
        payload.update(
            {
                "source_mps": problem["source_mps"],
                "benders_subproblem_id": problem["benders_subproblem_id"],
                "source_rows": problem["source_rows"],
            }
        )
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def export_for_mode(args: argparse.Namespace, modes: int, output_dir: Path) -> list[dict]:
    mode_dir = output_dir / f"nvars_{modes:02d}"
    mode_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    seen_signatures = set()
    models = [parse_mps(mps) for mps in sorted(Path(args.mip_dir).glob("*/*.mps"))]
    models.sort(key=lambda model: (-nonbinary_source_var_count(model, args.N), model.name))
    per_block_cap = max(1, getattr(args, "per_block_cap", 5))
    per_source_cap = max(1, getattr(args, "per_source_cap", 20))
    source_counts = {}
    for model in models:
        for subproblem in model.benders_subproblems:
            if source_counts.get(model.name, 0) >= per_source_cap:
                continue
            block_count = 0
            for problem in row_problem_candidates(
                model,
                modes,
                args.combos_per_row,
                args.N,
                only_subproblem_id=subproblem.get("id"),
                row_groups_per_block=args.row_groups_per_block,
                args=args,
            ):
                if source_counts.get(model.name, 0) >= per_source_cap:
                    break
                if block_count >= per_block_cap:
                    break
                signature = problem_signature(problem, include_source=args.dedupe_include_source)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                block_count += 1
                source_counts[model.name] = source_counts.get(model.name, 0) + 1
                candidates.append((len(problem["A"]), problem["feasible_states_in_qaoa_truncation"] if isinstance(problem["feasible_states_in_qaoa_truncation"], int) else 10**18, problem))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]["source_mps"], item[2]["name"]))
    exported = []
    for _, _, problem in candidates[: args.per_mode]:
        path = export_problem(problem, mode_dir)
        exported.append(
            {
                "path": str(path),
                "name": problem["name"],
                "source_mps": problem["source_mps"],
                "source_rows": problem["source_rows"],
                "n_constraints": len(problem["A"]),
                "n_variables": len(problem["variables"]),
                "N": problem["N"],
                "signature": problem_signature(problem, include_source=args.dedupe_include_source),
                "feasible_states_in_qaoa_truncation": problem["feasible_states_in_qaoa_truncation"],
            }
        )
    return exported


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mip-dir", default=str(DEFAULT_MIP_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--per-mode", type=int, default=100)
    parser.add_argument("--min-modes", type=int, default=4)
    parser.add_argument("--max-modes", type=int, default=10)
    parser.add_argument("--modes", type=int, help="Export only this variable count.")
    parser.add_argument("--combos-per-row", type=int, default=30)
    parser.add_argument("--row-groups-per-block", type=int, default=20)
    parser.add_argument("--per-block-cap", type=int, default=5)
    parser.add_argument("--per-source-cap", type=int, default=20)
    parser.add_argument("--min-nonzeros-per-row", type=int, default=3)
    parser.add_argument("--min-feasible-states", type=int, default=10)
    parser.add_argument("--max-feasible-states", type=int, default=200000)
    parser.add_argument("--max-candidate-state-space", type=int, default=300000)
    parser.add_argument("--dedupe-include-source", action="store_true")
    parser.add_argument("--N", type=int, default=8, help="Fock truncation N stored for cvIP/qaoa.py.")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for path in output_dir.glob("**/*.json"):
            path.unlink()

    mode_values = [args.modes] if args.modes else list(range(args.min_modes, args.max_modes + 1))
    manifest = {
        "mip_dir": str(Path(args.mip_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "per_mode": args.per_mode,
        "mode_values": mode_values,
        "N": args.N,
        "problems": [],
        "counts_by_n_variables": {},
    }

    missing = []
    for modes in mode_values:
        exported = export_for_mode(args, modes, output_dir)
        manifest["problems"].extend(exported)
        manifest["counts_by_n_variables"][str(modes)] = len(exported)
        print(f"nvars={modes}: exported {len(exported)}")
        if len(exported) < args.per_mode:
            missing.append((modes, len(exported)))

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    total = len(manifest["problems"])
    print(f"Exported {total} QAOA problems to {output_dir}")
    print(f"Wrote manifest: {manifest_path}")
    if missing:
        details = ", ".join(f"{modes}: {count}" for modes, count in missing)
        raise SystemExit(f"Some variable counts have fewer than {args.per_mode} problems: {details}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
