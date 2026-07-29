# Benchmark Protocol

## Source instances

The source set is the official MIPLIB 2017 benchmark set. The current
benchmark-v2 list contains 240 instance names and is downloaded from
`https://miplib.zib.de/downloads/benchmark-v2.test`. Full source MPS files are
downloaded from `https://miplib.zib.de/downloads/benchmark.zip`.

## Exact derived instance

A derived instance is exact with respect to the stated downsampling rule:

1. Parse an official source MPS instance.
2. Keep only variables that are binary or integer with finite integral bounds.
3. Rank variables by constraint degree and select the first `max_vars`.
4. Fix every unselected variable to zero.
5. Keep constraints whose selected-variable coefficient vector is nonzero.
6. Keep at most `max_rows` constraints after sorting by descending selected
   support size and then name.
7. Approximate coefficients as rationals with denominator at most
   `max_denominator`.
8. Multiply each constraint row by the least common multiple of denominators,
   capped at `max_scale`, and round to integers.
9. Apply the same rational scaling rule to the objective.

This produces a deterministic mixed-integer linear problem with integer
coefficients and finite discrete domains. The generated `.mps` file is the
experiment instance; the source MIPLIB file is only provenance.

## Variable bounds

Each `metadata.json` stores:

```json
"variable_bounds": {
  "x": {"lb": 0, "ub": 1, "type": "B"}
}
```

Bounds are copied from the source MPS for the selected variables. Variables
without finite integral bounds are not selected.

## Number of feasible states

For generated instances, all variables have finite integer domains. The number
of candidate states before constraints is:

```text
prod_i (ub_i - lb_i + 1)
```

This is stored as `candidate_state_space_size`.

If `candidate_state_space_size <= 1,000,000`, the script enumerates all
assignments and checks the generated MILP constraints. The exact number of
constraint-satisfying assignments is stored as `number_of_feasible_states`, with
`feasible_state_count_exact = true`. If the cap is exceeded,
`number_of_feasible_states` is `null`.

## Benders-style decomposition

The script stores a deterministic decomposition in `metadata.json`:

- `master_variables`: the highest-degree selected variables.
- `subproblems`: connected components of the remaining variable co-occurrence
  graph after removing master variables.

Two non-master variables are connected if they appear together in at least one
kept constraint. This is a structural Benders-style split meant to support
experiments that solve or analyze smaller coupled blocks. The script does not
claim that the decomposition is unique or solver-optimal; it is reproducible and
derived from the generated MILP matrix.

## Reproducibility knobs

The main controls are:

```text
--instances
--max-vars
--max-domain
--max-rows
--max-denominator
--max-scale
```

The exact values used for a run are written to `metadata/manifest.json`.
