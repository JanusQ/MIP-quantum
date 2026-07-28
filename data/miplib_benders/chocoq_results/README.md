# Choco-Q Benchmark Results

This directory stores Choco-Q results for the MIPLIB Benders benchmark set.

## Source

The files were copied from the H100 node:

```text
Host: xixilabH100
Path: /home/zhenyusen/Choco-Q-main/data/miplib_benders/chocoq_results
```

The run scripts used on H100 are archived in:

```text
Choco-Q-main/tools/run_chocoq_miplib_binary_batch.py
Choco-Q-main/tools/run_chocoq_miplib_binary_smoke.py
```

## Files

```text
chocoq_cpu_3bit_20260723_132718.csv
chocoq_cpu_3bit_20260724_no_timeout_jobs4.log
chocoq_cpu_3bit_smoke2.csv
```

`chocoq_cpu_3bit_20260723_132718.csv` is the main completed run. It contains
700 benchmark rows and uses the `ddsim` provider with 3-bit integer encoding.

Run summary:

```text
status ok: 700
n_int_vars 04: 100
n_int_vars 05: 100
n_int_vars 06: 100
n_int_vars 07: 100
n_int_vars 08: 100
n_int_vars 09: 100
n_int_vars 10: 100
```

The log tail on H100 reached:

```text
[parallel 274/274] ... status=ok
```
