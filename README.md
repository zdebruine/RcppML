# RcppML <img src="man/figures/logo.png" align="right" height="139" />

[![R-CMD-check](https://github.com/zdebruine/RcppML/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/zdebruine/RcppML/actions/workflows/R-CMD-check.yaml)
[![](https://cranlogs.r-pkg.org/badges/grand-total/RcppML)](https://cran.r-project.org/package=RcppML)
[![](https://www.r-pkg.org/badges/version-last-release/RcppML)](https://cran.r-project.org/package=RcppML)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**RcppML** is an R package for high-performance Non-negative Matrix Factorization (NMF) and divisive clustering of large sparse and dense matrices.

## Key Features

- **Fast NMF**: Alternating least squares with coordinate descent NNLS, OpenMP parallelism, and Eigen linear algebra
- **Composable graphs**: `factor_net` API for multi-modal, deep NMF, and cross-validation pipelines
- **Diagonal scaling**: Factorization in the form A ≈ W·diag(d)·H for consistent, interpretable factors
- **Cross-validation**: Speckled holdout for principled rank selection with `k = 2:10` or `k = "auto"`
- **Multiple loss functions**: MSE, MAE, Huber, and KL divergence via IRLS
- **Regularization**: L1 (LASSO), L2 (Ridge), L21 (group sparsity), angular, and graph Laplacian
- **GPU acceleration**: Optional CUDA backend (cuBLAS/cuSPARSE) for 10-20× speedup
- **SparsePress I/O**: Custom binary format for fast sparse matrix serialization
- **Divisive clustering**: `bipartition()` and `dclust()` for spectral clustering

## Installation

```r
# From CRAN
install.packages("RcppML")

# Development version from GitHub
devtools::install_github("zdebruine/RcppML")
```

## Quick Start

```r
library(RcppML)
library(Matrix)

# Basic NMF
A <- rsparsematrix(1000, 500, 0.1)
model <- nmf(A, k = 10)
model

# Cross-validation for rank selection
cv <- nmf(A, k = 2:15, test_fraction = 0.1, cv_seed = 1:3)
plot(cv)

# Evaluate reconstruction
evaluate(model, A)

# Project onto new data
H_new <- predict(model, A[, 1:100])
```

### Regularization & Loss Functions

```r
# L1 regularization for sparse factors
model <- nmf(A, k = 10, L1 = c(0.1, 0.1))

# Robust NMF with Huber loss
model <- nmf(A, k = 10, loss = "huber", huber_delta = 1.0)

# Semi-NMF (allow negative W)
model <- nmf(A, k = 10, nonneg = c(FALSE, TRUE))
```

### GPU Acceleration

```r
# Auto-detects GPU and dispatches automatically
gpu_available()
model <- nmf(A, k = 20, resource = "gpu")
```

See `vignette("gpu-acceleration")` for build instructions and performance details.

### Composable Factorization Graphs

`factor_net` builds complex factorization pipelines from simple blocks — multi-modal, deep NMF, cross-validation, and streaming, all composable with `|>`:

```r
# Multi-modal: shared factorization across two data matrices
inp1 <- factor_input(rna_matrix, "rna")
inp2 <- factor_input(atac_matrix, "atac")
shared <- factor_shared(inp1, inp2)
L1 <- shared |> nmf_layer(k = 10, W = W(L1 = 0.1))

net <- factor_net(inputs = list(inp1, inp2), output = L1,
                  config = factor_config(maxit = 100, seed = 42))
result <- fit(net)
result$L1$W$rna   # RNA loadings
result$L1$W$atac  # ATAC loadings
result$L1$H       # shared embedding

# Deep NMF: stacked layers
inp <- factor_input(A, "X")
enc <- inp |> nmf_layer(k = 20, name = "encoder")
bot <- enc |> nmf_layer(k = 5,  name = "bottleneck")

net <- factor_net(inputs = inp, output = bot,
                  config = factor_config(maxit = 50, seed = 42))
result <- fit(net)

# Cross-validate rank
cv <- cross_validate_graph(
  inputs = inp,
  layer_fn = function(p) inp |> nmf_layer(k = p$k),
  params = list(k = c(3, 5, 8, 12, 20)),
  config = factor_config(maxit = 50, seed = 42, holdout_fraction = 0.1),
  reps = 3, verbose = FALSE)
cv$best_params$k
```

See `vignette("factor-net-graphs")` for the full guide including graph regularization, streaming, and training logging.

### Divisive Clustering

```r
A <- rsparsematrix(1000, 1000, 0.1)
clusters <- dclust(A, min_dist = 0.001, min_samples = 5)
partition <- bipartition(A)
```

## Documentation

- `vignette("getting-started")` — Basic NMF usage and interpretation
- `vignette("factor-net-graphs")` — Composable factorization graphs (multi-modal, deep, CV)
- `vignette("cross-validation")` — Rank selection with speckled holdout
- `vignette("gpu-acceleration")` — CUDA backend setup and usage
- [pkgdown site](https://zdebruine.github.io/RcppML/) — Full API reference

## Citation

DeBruine, ZJ, Melcher, K, and Triche, TJ. (2021). "High-performance non-negative matrix factorization for large single-cell data." *BioRXiv*. [doi:10.1101/2021.09.01.458620](https://www.biorxiv.org/content/10.1101/2021.09.01.458620v1)

## License

GPL (>= 3)
