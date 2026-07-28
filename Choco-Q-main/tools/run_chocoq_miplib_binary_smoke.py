#!/usr/bin/env python3
"""Smoke-test Choco-Q on one MIPLIB-derived integer problem via 3-bit encoding."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from chocoq.model import LinearConstrainedBinaryOptimization as LcboModel
from chocoq.solvers.optimizers import CobylaOptimizer
from chocoq.solvers.qiskit import AerGpuProvider, ChocoSolver


def bit_name(var: str, bit: int) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in var)
    return f"{safe}_b{bit}"


def expr_for(bits):
    return bits[0] + 2 * bits[1] + 4 * bits[2]


def brute_force_best(data: dict) -> tuple[float | None, tuple[int, ...] | None]:
    n = len(data["variables"])
    best = None
    best_values = None
    for values in itertools.product(range(8), repeat=n):
        if all(sum(row[i] * values[i] for i in range(n)) == rhs for row, rhs in zip(data["A"], data["b"])):
            obj = sum(data["c"][i] * values[i] for i in range(n))
            if best is None or obj > best:
                best = float(obj)
                best_values = values
    return best, best_values


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("problem")
    ap.add_argument("--max-iter", type=int, default=12)
    ap.add_argument("--shots", type=int, default=256)
    args = ap.parse_args()

    path = Path(args.problem)
    data = json.loads(path.read_text(encoding="utf-8"))
    variables = data["variables"]

    print(f"[input] file={path}")
    print(f"[input] name={data['name']} n_int_vars={len(variables)} n_constraints={len(data['A'])} N={data.get('N')}")
    print(f"[input] integer variables={variables}")
    print(f"[input] A={data['A']}")
    print(f"[input] b={data['b']}")
    print(f"[input] c={data['c']}")

    best, best_values = brute_force_best(data)
    print(f"[classical-check] brute_force_best={best} values={best_values}")

    m = LcboModel()
    bit_vars = {}
    flat_bits = []
    for var in variables:
        bit_vars[var] = [m.addVar(name=bit_name(var, bit)) for bit in range(3)]
        flat_bits.extend(bit_vars[var])

    print("[encoding] 3-bit expansion:")
    for var in variables:
        names = [b.name for b in bit_vars[var]]
        print(f"  {var} = {names[0]} + 2*{names[1]} + 4*{names[2]}")

    objective = 0
    for coeff, var in zip(data["c"], variables):
        objective += int(coeff) * expr_for(bit_vars[var])
    m.setObjective(objective, "max")

    expanded_rows = []
    for row, rhs in zip(data["A"], data["b"]):
        lhs = 0
        expanded = []
        for coeff, var in zip(row, variables):
            coeff = int(coeff)
            lhs += coeff * expr_for(bit_vars[var])
            expanded.extend([coeff, 2 * coeff, 4 * coeff])
        m.addConstr(lhs == int(rhs))
        expanded_rows.append(expanded)

    print(f"[expanded] binary_vars={len(flat_bits)}")
    print(f"[expanded] A_binary={expanded_rows}")
    print(f"[expanded] b={data['b']}")
    print(f"[expanded] lin_constr_mtx=\n{m.lin_constr_mtx}")

    # Avoid a second exhaustive pass inside Choco-Q's evaluation setup.
    if best is not None:
        m._best_cost = best

    print("[gurobi-check] optimizing expanded binary model...")
    print(f"[gurobi-check] optimize={m.optimize()}")
    if best is not None:
        m._best_cost = best

    provider = AerGpuProvider()
    print(f"[provider] backend={provider.backend}")
    print(f"[provider] devices={provider.backend.available_devices()}")

    solver = ChocoSolver(
        prb_model=m,
        optimizer=CobylaOptimizer(max_iter=args.max_iter),
        provider=provider,
        num_layers=1,
        shots=args.shots,
    )
    print(f"[circuit] metrics={solver.circuit_analyze(['depth', 'width', 'culled_depth', 'num_one_qubit_gates'])}")
    print("[solve] starting Choco-Q solve...")
    result = solver.solve()
    print(f"[solve] result={result}")
    print(f"[eval] {solver.evaluation()}")
    print("[done] CHOCOQ_MIPLIB_BINARY_SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
