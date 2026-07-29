# QAOA Benchmark Results

This directory stores QAOA benchmark CSVs collected for the MIPLIB Benders
benchmark set.

## Files

```text
qaoa_feasible_benchmark_a100_20260720_163217_qaoa.csv
qaoa_feasible_benchmark_a100_20260720_163217.environment.json
qaoa_feasible_benchmark_local_nvars04_20260720_145741.csv
qaoa_feasible_benchmark_local_nvars04_20260720_145741.environment.json
qaoa_results_merged_qaoa.csv
qaoa_results_merged_cvIP_qaoa_first_feasible.csv
```

The two `qaoa_results_merged_*` files were merged on the A100 node from:

```text
/home/zhenyusen/MIP-quantum/data/miplib_benders/qaoa_results
```

`qaoa_results_merged_qaoa.csv` combines the standard benchmark worker CSVs.
`qaoa_results_merged_cvIP_qaoa_first_feasible.csv` combines the lower-level
`cvIP.qaoa` first-feasible telemetry CSVs.

Both merged files include a `source_file` column so each row can be traced back
to its original shard or run CSV.
