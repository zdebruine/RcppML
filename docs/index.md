# RcppML ![](reference/figures/logo.png)

[![CRAN
status](https://www.r-pkg.org/badges/version/RcppML)](https://cran.r-project.org/package=RcppML)
[![R-CMD-check](https://github.com/zdebruine/RcppML/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/zdebruine/RcppML/actions/workflows/R-CMD-check.yaml)
[![CRAN
downloads](https://cranlogs.r-pkg.org/badges/grand-total/RcppML)](https://cran.r-project.org/package=RcppML)
[![License: GPL
v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**RcppML** is an R package for high-performance **Non-negative Matrix
Factorization** (NMF), truncated **SVD/PCA**, and divisive clustering of
large sparse and dense matrices. It supports six statistical
distributions via IRLS, cross-validation for automatic rank selection,
optional CUDA GPU acceleration, out-of-core streaming for datasets
larger than memory, and a composable graph DSL for multi-modal and deep
NMF.

------------------------------------------------------------------------

## Overview

RcppML decomposes a matrix **A** into lower-rank non-negative factors:

**A ≈ W · diag(d) · H**

where **W** (features × k) and **H** (k × samples) have columns/rows
scaled to unit sum via a diagonal **d**, ensuring interpretable,
scale-invariant factors.

Key capabilities:

- **Fast NMF** with alternating least squares, coordinate descent or
  Cholesky NNLS, and OpenMP parallelism via the Eigen C++ library
- **Six statistical distributions** — Gaussian (MSE), Generalized
  Poisson, Negative Binomial, Gamma, Inverse Gaussian, Tweedie — fitted
  via Iteratively Reweighted Least Squares (IRLS)
- **Cross-validation** with speckled holdout masks for principled rank
  selection
- **GPU acceleration** via CUDA (cuBLAS/cuSPARSE) with automatic
  fallback to CPU
- **Streaming NMF** from SparsePress (`.spz`) files for datasets that
  exceed available memory
- **FactorNet graph API** for composable multi-modal, deep, and
  branching NMF pipelines
- **Truncated SVD** with five methods: deflation, Krylov, Lanczos,
  IRLBA, randomized
- **Divisive clustering** via recursive rank-2 factorizations

------------------------------------------------------------------------

## Installation

``` r
# Stable release from CRAN
install.packages("RcppML")

# Development version from GitHub (requires Rcpp and RcppEigen)
devtools::install_github("zdebruine/RcppML")
```

**GPU support** (optional): Requires CUDA Toolkit ≥ 11.0 installed on
the system. See
[`vignette("gpu-acceleration")`](https://zdebruine.github.io/RcppML/articles/gpu-acceleration.md)
for build instructions.

------------------------------------------------------------------------

## Quick Start

``` r
library(RcppML)

# Load a built-in single-cell gene expression dataset
data(pbmc3k)  # 32738 genes × 2700 cells sparse matrix

# Run NMF with rank k = 10
model <- nmf(pbmc3k, k = 10)
model
#> nmf model
#>   k = 10
#>   w: 32738 x 10
#>   d: 10
#>   h: 10 x 2700

# Inspect factor loadings
head(model@w)
model@d
model@h[, 1:5]

# Reconstruction error
evaluate(model, pbmc3k)

# Project onto new data
H_new <- predict(model, pbmc3k[, 1:100])

# Plot the model summary
plot(model)
```

------------------------------------------------------------------------

## Statistical Distributions

RcppML goes beyond standard Frobenius-norm NMF by supporting
distribution-appropriate loss functions via IRLS. This is critical for
count data (scRNA-seq, text mining) where Gaussian assumptions are
wrong.

| `loss=`              | Distribution        | Variance V(μ) | Use Case                           |
|----------------------|---------------------|---------------|------------------------------------|
| `"mse"`              | Gaussian            | constant      | Dense/general data (default)       |
| `"gp"`               | Generalized Poisson | μ(1+θ)²       | Overdispersed counts               |
| `"nb"`               | Negative Binomial   | μ + μ²/r      | scRNA-seq, quadratic variance-mean |
| `"gamma"`            | Gamma               | μ²            | Positive continuous data           |
| `"inverse_gaussian"` | Inverse Gaussian    | μ³            | Heavy right-tailed positive        |
| `"tweedie"`          | Tweedie             | μ^p           | Flexible power-law variance        |

### Example: Negative Binomial NMF for scRNA-seq

``` r
data(pbmc3k)

# NB is the natural choice for scRNA-seq count data
model_nb <- nmf(pbmc3k, k = 10, loss = "nb")

# Or use the distribution API with fine-grained control
model_nb <- nmf(pbmc3k, k = 10,
                distribution = "nb",
                distribution_config = list(
                  nb_size_init = 10,     # Initial size parameter r
                  nb_size_max = 1e6      # Upper bound for r
                ))

# Automatic distribution selection via BIC
auto <- auto_nmf_distribution(pbmc3k, k = 10)
auto$best       # "nb"
auto$comparison  # BIC/AIC table for each distribution
```

### Zero-Inflation Models

For data with excess zeros (e.g., droplet-based scRNA-seq),
zero-inflated GP and NB models estimate structural dropout probabilities
separately from the count model:

``` r
# Zero-inflated Negative Binomial with per-row dropout
model_zinb <- nmf(pbmc3k, k = 10, loss = "nb", zi = "row")

# Automatic zero-inflation detection
model <- nmf(pbmc3k, k = 10, loss = "nb", zero_inflation = "auto")
```

------------------------------------------------------------------------

## Cross-Validation

Cross-validation uses a **speckled holdout mask** — a random subset of
matrix entries are held out during training, and test error is computed
on these entries. This enables principled rank selection.

``` r
data(pbmc3k)

# Sweep k = 2 through 20, three replicates per rank
cv <- nmf(pbmc3k, k = 2:20, test_fraction = 0.1, cv_seed = 1:3)

# Visualize test error vs rank
plot(cv)

# Automatic rank selection (binary search)
model_auto <- nmf(pbmc3k, k = "auto")
model_auto@misc$rank_search$k_optimal
```

### Cross-Validation Options

``` r
cv <- nmf(data, k = 2:15,
          test_fraction = 0.1,      # 10% holdout
          mask_zeros = TRUE,        # Only non-zero entries in test set (for sparse data)
          patience = 5,             # Early stopping patience
          cv_seed = c(42, 123, 7))  # Multiple seeds for replicates
```

------------------------------------------------------------------------

## Regularization

RcppML supports a comprehensive suite of regularization methods that can
be combined freely:

### L1 (LASSO) — Sparsity

``` r
# Symmetric L1 on both W and H
model <- nmf(data, k = 10, L1 = 0.1)

# Asymmetric: sparse H (embedding), dense W (loadings)
model <- nmf(data, k = 10, L1 = c(0, 0.2))
```

### L2 (Ridge) — Shrinkage

``` r
model <- nmf(data, k = 10, L2 = c(0.01, 0.01))
```

### L21 (Group Sparsity) — Factor Selection

L21-norm regularization drives entire rows of W or columns of H toward
zero, effectively performing automatic factor selection:

``` r
model <- nmf(data, k = 20, L21 = c(0.1, 0))
# Some factors in W will be entirely zeroed out
```

### Angular — Decorrelation

Encourages orthogonality between factors:

``` r
model <- nmf(data, k = 10, angular = c(0.1, 0.1))
```

### Graph Laplacian — Spatial/Network Smoothness

If features or samples have a known graph structure (e.g., gene
regulatory network, spatial coordinates), graph regularization
encourages connected nodes to have similar factor representations:

``` r
# gene_graph: m × m sparse adjacency matrix
model <- nmf(data, k = 10, graph_W = gene_graph, graph_lambda = 0.5)

# Both feature and sample graphs
model <- nmf(data, k = 10,
             graph_W = gene_graph, graph_H = cell_graph,
             graph_lambda = c(0.5, 0.3))
```

### Upper Bound Constraints

``` r
# Box constraint: all entries in W and H between 0 and 1
model <- nmf(data, k = 10, upper_bound = c(1, 1))
```

### Robust NMF (Huber Loss)

``` r
# Huber-type robustness (less sensitive to outliers)
model <- nmf(data, k = 10, robust = TRUE)

# Custom Huber delta
model <- nmf(data, k = 10, robust = 2.0)

# MAE (L1 loss)
model <- nmf(data, k = 10, robust = "mae")
```

------------------------------------------------------------------------

## GPU Acceleration

RcppML optionally uses CUDA for GPU-accelerated NMF, delivering 10–20×
speedups on large matrices.

``` r
# Check GPU availability
gpu_available()
#> [1] TRUE

gpu_info()
#> $device_name: "NVIDIA A100"
#> $total_memory_mb: 40960

# Sparse NMF on GPU
model <- nmf(data, k = 20, resource = "gpu")

# Dense NMF on GPU
model <- nmf(as.matrix(data), k = 20, resource = "gpu")

# Auto-dispatch (default): uses GPU if available, falls back to CPU
model <- nmf(data, k = 20, resource = "auto")
```

The GPU backend supports: - Standard NMF (sparse and dense) -
Cross-validation NMF - MSE loss (GPU-native), all other losses via
automatic CPU fallback - OpenMP + CUDA hybrid execution - Zero-copy NMF
with
[`sp_read_gpu()`](https://zdebruine.github.io/RcppML/reference/sp_read_gpu.md)
for pre-loaded GPU data

------------------------------------------------------------------------

## Streaming Large Data (SparsePress `.spz` Files)

For datasets that exceed available RAM, RcppML streams data from
**SparsePress** (`.spz`) compressed files — a column-oriented binary
format with 10–20× compression via rANS entropy coding.

> **Note**: The SparsePress format will be renamed to **StreamPress** in
> an upcoming release.

### Writing and Reading SPZ Files

``` r
library(RcppML)
data(pbmc3k)

# Write sparse matrix to .spz file
spz_path <- tempfile(fileext = ".spz")
sp_write(pbmc3k, spz_path, include_transpose = TRUE)

# Read back into memory
mat <- sp_read(spz_path)
identical(dim(mat), dim(pbmc3k))  # TRUE

# File size comparison
file.size(spz_path)  # Much smaller than RDS/RDA
```

### Streaming NMF

``` r
# NMF directly from .spz file — data never fully loaded into memory
model <- nmf(spz_path, k = 10)

# Streaming cross-validation
cv <- nmf(spz_path, k = 2:15, test_fraction = 0.1)

# Streaming with non-MSE distributions
model <- nmf(spz_path, k = 10, loss = "nb")
```

Streaming NMF processes data in column panels with double-buffered
asynchronous I/O, maintaining O(m·k + n·k + chunk) memory regardless of
total matrix size.

------------------------------------------------------------------------

## FactorNet Graph API

The
[`factor_net()`](https://zdebruine.github.io/RcppML/reference/factor_net.md)
API composes complex factorization pipelines from simple building
blocks. Multi-modal, deep, and branching NMF networks are expressed as
directed graphs:

### Multi-Modal NMF

Share a common embedding **H** across two data matrices (e.g., RNA +
ATAC from the same cells):

``` r
# Define inputs
inp_rna  <- factor_input(rna_matrix, "rna")
inp_atac <- factor_input(atac_matrix, "atac")

# Create shared input node
shared <- factor_shared(inp_rna, inp_atac)

# Build NMF layer with per-factor regularization
layer <- shared |> nmf_layer(k = 10,
  W = W(L1 = 0.1),
  H = H(L1 = 0.05))

# Compile and fit the network
net <- factor_net(
  inputs = list(inp_rna, inp_atac),
  output = layer,
  config = factor_config(maxit = 100, seed = 42))

result <- fit(net)

# Access results
result$rna$W     # RNA feature loadings (genes × 10)
result$atac$W    # ATAC feature loadings (peaks × 10)
result$H         # Shared cell embedding (10 × n_cells)
```

### Deep NMF

Stack layers for hierarchical decomposition:

``` r
inp <- factor_input(data, "X")

# Encoder layer: reduce to 20 factors
enc <- inp |> nmf_layer(k = 20, name = "encoder")

# Bottleneck: compress to 5 factors
bot <- enc |> nmf_layer(k = 5, name = "bottleneck")

net <- factor_net(inputs = list(inp), output = bot,
                  config = factor_config(maxit = 100, seed = 42))
result <- fit(net)
```

### Conditional Factorization

Split samples by a condition and factorize each group:

``` r
inp <- factor_input(data, "X")
cond <- factor_condition(inp, groups = cell_types)

layer <- cond |> nmf_layer(k = 10)

net <- factor_net(inputs = list(inp), output = layer,
                  config = factor_config(seed = 42))
result <- fit(net)
```

------------------------------------------------------------------------

## Truncated SVD / PCA

RcppML provides five SVD algorithms with optional constraints and
cross-validation:

``` r
# Standard truncated SVD
result <- svd(data, k = 10)

# PCA (centered SVD)
result <- pca(data, k = 10)

# Non-negative PCA
result <- pca(data, k = 10, nonneg = TRUE)

# Sparse PCA with L1 penalty
result <- pca(data, k = 10, L1 = c(0, 0.1))

# Auto-rank selection
result <- svd(data, k = "auto")

# Method selection
result <- svd(data, k = 10, method = "lanczos")   # Fast unconstrained
result <- svd(data, k = 10, method = "krylov")     # Block method, all constraints
result <- svd(data, k = 10, method = "randomized") # Approximate, very fast
```

| Method       | Constraints | Speed     | Best For                   |
|--------------|-------------|-----------|----------------------------|
| `deflation`  | All         | Moderate  | Mixed constraints, small k |
| `krylov`     | All         | Fast      | Large k with constraints   |
| `lanczos`    | None        | Very fast | Unconstrained SVD          |
| `irlba`      | None        | Fast      | General unconstrained      |
| `randomized` | None        | Very fast | Approximate large-scale    |

------------------------------------------------------------------------

## Divisive Clustering

RcppML implements spectral clustering via recursive rank-2 NMF:

``` r
# Single bipartition
bp <- bipartition(data)
bp$samples  # Cluster assignments (0/1)

# Recursive divisive clustering
clusters <- dclust(data, min_samples = 50, min_dist = 0.05)
clusters$id  # Cluster labels

# Consensus clustering (multiple runs for stability)
cons <- consensus_nmf(data, k = 5, n_runs = 20)
plot(cons)
```

------------------------------------------------------------------------

## Non-Negative Least Squares (NNLS)

Project factor matrices onto new data:

``` r
# Given W from NMF, solve for H on new data
H_new <- nnls(w = model@w, A = new_data)

# Solve for W given H (transpose projection)
W_new <- nnls(h = model@h, A = new_data)

# Unconstrained LS (semi-NMF projection)
H_ls <- nnls(w = model@w, A = new_data, nonneg = c(TRUE, FALSE))

# With regularization
H_sparse <- nnls(w = model@w, A = new_data, L1 = c(0, 0.1))
```

------------------------------------------------------------------------

## Semi-NMF

Allow negative values in W (unconstrained) while keeping H non-negative:

``` r
model <- nmf(data, k = 10, nonneg = c(FALSE, TRUE))
any(model@w < 0)  # TRUE — W can have negative entries
all(model@h >= 0)  # TRUE — H remains non-negative
```

------------------------------------------------------------------------

## Key Parameters Reference

| Parameter       | Type              | Default        | Description                                                                        |
|-----------------|-------------------|----------------|------------------------------------------------------------------------------------|
| `k`             | int/vector/“auto” | —              | Rank (vector for CV sweep, “auto” for search)                                      |
| `loss`          | string            | `"mse"`        | Loss function: mse, gp, nb, gamma, inverse_gaussian, tweedie                       |
| `distribution`  | string            | `NULL`         | Alternative API: auto, gaussian, poisson, gp, nb, gamma, inverse_gaussian, tweedie |
| `L1`            | numeric(2)        | `c(0,0)`       | LASSO penalty \[W, H\]                                                             |
| `L2`            | numeric(2)        | `c(0,0)`       | Ridge penalty \[W, H\]                                                             |
| `L21`           | numeric(2)        | `c(0,0)`       | Group sparsity \[W, H\]                                                            |
| `angular`       | numeric(2)        | `c(0,0)`       | Decorrelation penalty \[W, H\]                                                     |
| `nonneg`        | logical(2)        | `c(TRUE,TRUE)` | Non-negativity \[W, H\]                                                            |
| `upper_bound`   | numeric(2)        | `c(0,0)`       | Box constraint \[W, H\] (0 = none)                                                 |
| `zi`            | string            | `"none"`       | Zero-inflation: none, row, col                                                     |
| `robust`        | logical/numeric   | `FALSE`        | Huber robustness (TRUE = δ=1.345, numeric = custom δ)                              |
| `solver`        | string            | `"auto"`       | NNLS solver: auto, cd, cholesky                                                    |
| `init`          | string            | `"random"`     | Initialization: random, lanczos, irlba                                             |
| `resource`      | string            | `"auto"`       | Compute: auto, cpu, gpu                                                            |
| `streaming`     | string/logical    | `"auto"`       | Out-of-core mode for .spz files                                                    |
| `test_fraction` | numeric           | `0`            | CV holdout fraction (0 = no CV)                                                    |
| `tol`           | numeric           | `1e-4`         | Convergence tolerance                                                              |
| `maxit`         | int               | `100`          | Maximum iterations                                                                 |
| `threads`       | int               | `0`            | OpenMP threads (0 = all)                                                           |
| `verbose`       | logical           | `FALSE`        | Print progress                                                                     |

------------------------------------------------------------------------

## Built-in Datasets

| Dataset       | Description                             | Dimensions   |
|---------------|-----------------------------------------|--------------|
| `pbmc3k`      | PBMC single-cell RNA-seq (10x Genomics) | 32738 × 2700 |
| `aml`         | AML leukemia gene expression            | 12023 × 200  |
| `golub`       | Golub leukemia microarray               | 7129 × 72    |
| `movielens`   | MovieLens 100K ratings                  | 610 × 9724   |
| `hawaiibirds` | Hawaii bird survey counts               | 40 × 23      |
| `olivetti`    | Olivetti face images                    | 4096 × 400   |
| `digits_full` | Handwritten digits (MNIST subset)       | 784 × 1000   |

------------------------------------------------------------------------

## Performance Tips

1.  **Sparse input**: Use `dgCMatrix` format (from the `Matrix` package)
    for sparse data — RcppML auto-detects and uses optimized sparse
    routines.

2.  **Solver selection**: For large k (\> 32), `solver = "cholesky"` can
    be faster than coordinate descent. Use `solver = "auto"` (default)
    for automatic selection.

3.  **Initialization**: `init = "lanczos"` provides better starting
    points and can reduce iteration count by 30–50%, but adds upfront
    SVD cost.

4.  **Threads**: RcppML uses OpenMP. Set `options(RcppML.threads = 4)`
    to control parallelism, or `threads = 0` for all available cores.

5.  **GPU**: For matrices with \> 10K rows and k \> 8, GPU acceleration
    provides significant speedup. Use `resource = "gpu"`.

6.  **Streaming**: For datasets larger than available RAM, write to
    `.spz` format with
    [`sp_write()`](https://zdebruine.github.io/RcppML/reference/sparsepress-deprecated.md)
    and factorize directly from the file path.

------------------------------------------------------------------------

## Contributing

See
[CONTRIBUTING.md](https://zdebruine.github.io/RcppML/CONTRIBUTING.md)
for guidelines on reporting bugs, requesting features, and submitting
pull requests.

------------------------------------------------------------------------

## Citation

``` r
citation("RcppML")
```

    DeBruine ZJ, Melcher K, Triche TJ (2021). "High-performance non-negative
    matrix factorization for large single-cell data." BioRXiv.
    doi:10.1101/2021.09.01.458620.

------------------------------------------------------------------------

## License

GPL (≥ 3)
