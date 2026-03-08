# RcppML Benchmark Harness

Automated benchmark suite for regression detection and performance profiling.

## Quick Start

```bash
# On a compute node (NEVER on the login node):
module load r/4.5.2
export OMP_NUM_THREADS=4

# Run all benchmarks (< 10 min):
Rscript benchmarks/harness/run_all.R

# Run a single suite:
Rscript benchmarks/harness/run_all.R --suite nmf_cpu_baseline

# Generate datasets only:
Rscript benchmarks/harness/datasets/generate.R

# Check for regressions against baseline:
Rscript benchmarks/harness/analysis/regression_check.R
```

## Directory Structure

```
harness/
├── config.yaml              # Suite definitions and parameters
├── run_all.R                # Master harness
├── datasets/
│   └── generate.R           # Synthetic data generators
├── suites/
│   ├── nmf_cpu_baseline.R   # MSE NMF on CPU
│   ├── nmf_gpu_baseline.R   # MSE NMF on GPU
│   ├── nmf_distributions.R  # GP, NB, Gamma
│   ├── nmf_cv.R             # Cross-validation
│   ├── nmf_streaming.R      # Streaming vs in-memory
│   ├── svd_methods.R        # SVD method comparison
│   └── nnls_crossover.R     # CD vs Cholesky crossover
├── results/
│   ├── baseline/            # Frozen baselines
│   └── current/             # Latest run
├── analysis/
│   ├── regression_check.R   # Flag > 5% regressions
│   └── generate_report.R    # Summary tables
└── README.md
```

## Design Constraints

- **10-minute wall-clock budget** for the full suite
- **5 replicates** per configuration for variance estimation
- **Fixed iterations** (`tol=1e-10`) to force `maxit` iterations
- **Reproducible** via pinned seed, dataset, and git commit
- **Schema**: Results stored as YAML with standard metadata

## Regression Detection

Run `analysis/regression_check.R` to compare current results against baseline.
A regression is flagged if any benchmark's mean time increases by > 5%.
