# Truncated SVD / PCA with constraints and regularization

Compute a truncated SVD or PCA of a matrix using multiple algorithm
backends. Supports sparse and dense matrices, implicit PCA centering,
per-element regularization (L1/L2/non-negativity/bounds), Gram-level
regularization (L2,1/angular/graph Laplacian), and automatic rank
selection via speckled holdout cross-validation.

## Usage

``` r
svd(
  A,
  k = 10,
  tol = 1e-05,
  maxit = 200,
  center = FALSE,
  scale = FALSE,
  verbose = FALSE,
  seed = NULL,
  threads = 0,
  L1 = 0,
  L2 = 0,
  nonneg = FALSE,
  upper_bound = 0,
  L21 = 0,
  angular = 0,
  graph_U = NULL,
  graph_V = NULL,
  graph_lambda = 0,
  convergence = "factor",
  test_fraction = 0,
  cv_seed = NULL,
  patience = 3,
  mask_zeros = FALSE,
  obs_mask = NULL,
  robust = FALSE,
  k_max = 50,
  resource = "auto",
  method = "auto"
)

sparse_pca(A, k = 10, L1 = 0.5, ...)

nn_pca(A, k = 10, ...)

svd_pca(...)
```

## Arguments

- A:

  Input matrix. May be dense (`matrix`), sparse (`dgCMatrix`), or a path
  to a `.spz` file for out-of-core streaming SVD.

- k:

  Number of factors (rank). Use `"auto"` for automatic rank selection
  via cross-validation. Default: 10.

- tol:

  Convergence tolerance per rank-1 subproblem. Measures \\1 -
  \|\cos(u\_{new}, u\_{old})\|\\. Default: 1e-5.

- maxit:

  Maximum ALS iterations per factor. Default: 200.

- center:

  If `TRUE`, subtract row means (PCA mode). Default: `FALSE`.

- scale:

  If `TRUE`, divide each row by its standard deviation after centering
  (correlation PCA mode). Implies `center = TRUE`. This computes PCA on
  the correlation matrix rather than the covariance matrix. Default:
  `FALSE`.

- verbose:

  Print per-factor diagnostics. Default: `FALSE`.

- seed:

  Random seed for initialization and cross-validation. Default: NULL
  (random). Use an integer for reproducibility.

- threads:

  Number of OpenMP threads. 0 = all available. Default: 0.

- L1:

  L1 (lasso) penalty. Either a single value (applied to both u and v) or
  a length-2 vector `c(L1_u, L1_v)`. Default: 0.

- L2:

  L2 (ridge) penalty. Either a single value or length-2 vector. Default:
  0.

- nonneg:

  Non-negativity constraints. Either a single logical (both sides) or a
  length-2 logical vector `c(nonneg_u, nonneg_v)`. Default: `FALSE`.

- upper_bound:

  Upper bound constraints. Single value or length-2 vector. Default: 0
  (no bound).

- L21:

  L2,1 (group sparsity) penalty. Drives entire components toward zero.
  Single value or length-2 vector `c(L21_u, L21_v)`. Default: 0.
  Supported by: `deflation` (adaptive L2), `krylov` (Gram-level).

- angular:

  Angular (orthogonality) penalty. Decorrelates components. Single value
  or length-2 vector `c(angular_u, angular_v)`. Default: 0. Supported
  by: `deflation` and `krylov`.

- graph_U:

  Sparse graph Laplacian matrix for features (rows, m x m). Encourages
  smooth loadings along feature graph edges. Default: `NULL`. Supported
  by: `deflation` and `krylov`.

- graph_V:

  Sparse graph Laplacian matrix for samples (columns, n x n). Default:
  `NULL`. Supported by: `deflation` and `krylov`.

- graph_lambda:

  Graph regularization strength. Single value or length-2 vector
  `c(graph_u_lambda, graph_v_lambda)`. Default: 0.

- convergence:

  Convergence criterion: `"factor"` (track change in factors, default),
  `"loss"` (track change in explained variance), or `"both"` (stop when
  either criterion is met).

- test_fraction:

  Fraction of entries to hold out for cross-validation / auto-rank.
  Default: 0 (disabled). Set to a value in (0, 1) to enable CV holdout,
  or use `k = "auto"` which automatically sets this to 0.05.

- cv_seed:

  Separate seed for holdout mask. Default: NULL (derive from `seed`).

- patience:

  For auto-rank: stop after this many non-improving factors. Default: 3.

- mask_zeros:

  If `TRUE`, only non-zero entries can be holdout (for sparse
  recommendation data). Default: `FALSE`.

- obs_mask:

  Optional sparse matrix (`dgCMatrix`) of the same dimensions as `A`.
  Non-zero entries indicate observations to exclude from fitting (e.g.,
  known outliers or missing-data indicators). Currently supported by the
  `deflation` method. Default: `NULL` (no masking).

- robust:

  Robustness control for outlier downweighting via Huber loss. `FALSE`
  (default): standard MSE (no robustness). `TRUE`: Huber-type robustness
  with delta=1.345 (95% asymptotic efficiency). `"mae"`: near-MAE
  behavior (delta=1e-4). Numeric: custom Huber delta value. Uses IRLS
  reweighting within the ALS deflation loop. Supported by `deflation`
  method only; other methods are automatically redirected to deflation
  when robust is active.

- k_max:

  Maximum rank for auto-rank mode. Default: 50.

- resource:

  Compute backend: `"auto"` (default, auto-detect), `"cpu"`, or `"gpu"`.

- method:

  Algorithm to use. One of `"auto"` (default), `"deflation"`,
  `"krylov"`, `"lanczos"`, `"irlba"`, `"randomized"`. When `"auto"`:
  constraints route to deflation/krylov; otherwise method is selected
  based on rank and resource (GPU: Lanczos k\<32, Randomized 32\<=k\<64,
  IRLBA k\>=64; CPU: Lanczos k\<64, IRLBA k\>=64).

- ...:

  Additional arguments passed to `svd` (used by convenience wrappers).

## Value

An S4 object of class `svd_pca` with slots:

- `u`:

  Left singular vectors (scores), m x k matrix

- `d`:

  Singular values, length-k numeric vector

- `v`:

  Right singular vectors (loadings), n x k matrix

- `misc`:

  List with metadata: `centered`, `row_means`, `test_loss`,
  `iters_per_factor`, `wall_time_ms`

## Details

Multiple algorithms are available via the `method` parameter:

- `"deflation"`:

  Sequential rank-1 ALS with deflation correction. Supports all
  constraints/CV/auto-rank. Best for small k with mixed constraints.

- `"krylov"`:

  Krylov-Seeded Projected Refinement (KSPR). Block method: computes all
  k factors simultaneously via Lanczos seed + Gram-solve-then-project.
  Faster than deflation for larger k. Supports all regularization. CV is
  evaluated post-convergence.

- `"lanczos"`:

  Unconstrained Lanczos bidiagonalization. Fast for unconstrained SVD.
  No regularization support.

- `"irlba"`:

  Implicitly Restarted Lanczos. Good general-purpose unconstrained SVD.
  No regularization support.

- `"randomized"`:

  Randomized SVD with power iterations. Fast approximate SVD for large
  matrices. No regularization support.

## Note

This function shadows [`base::svd()`](https://rdrr.io/r/base/svd.html).
To use the base R version, call
[`base::svd()`](https://rdrr.io/r/base/svd.html) explicitly. The RcppML
version provides iterative SVD algorithms suitable for large sparse
matrices, whereas [`base::svd()`](https://rdrr.io/r/base/svd.html)
computes the full SVD using LAPACK.

## Auto-rank

When `k = "auto"`, a speckled holdout set is created and test MSE is
evaluated after each factor. Rank selection stops when test MSE fails to
improve for `patience` consecutive factors. The returned model is
truncated to the best rank.

## Convenience Aliases

- `pca(A, k, ...)`: PCA (same as `svd(A, k, center = TRUE, ...)`)

- `sparse_pca(A, k, L1, ...)`: PCA with L1 sparsity on v

- `nn_pca(A, k, ...)`: Non-negative PCA (centered, non-negative u and v)

- `svd_pca(...)`: Deprecated alias for `svd()`

## Unsupported Combinations

- Robust + non-deflation method:

  Huber-type robustness (`robust=TRUE` or numeric delta) requires IRLS
  reweighting and is only supported by the `deflation` method. Other
  methods are automatically redirected to deflation when `robust` is
  active.

- GPU dense streaming:

  GPU streaming SVD for dense matrices is not yet implemented. Dense
  streaming requires SPZ v3 dense format support.

- GPU Krylov dense:

  The Krylov method does not have a GPU dense implementation. Use
  `lanczos`, `irlba`, or `randomized` on GPU with dense input.

- Constraints + matrix-free methods:

  Regularization (L1, L2, L21, angular, graph, nonneg, upper_bound) is
  only supported by `deflation` and `krylov` methods. Using constraints
  with `lanczos`, `irlba`, or `randomized` will produce an error.

## Parameter Conventions (vs. nmf)

Several parameters in `svd()` differ intentionally from
[`nmf()`](https://zdebruine.github.io/RcppML/reference/nmf.md):

- Input naming:

  `svd()` uses `A` (linear algebra convention);
  [`nmf()`](https://zdebruine.github.io/RcppML/reference/nmf.md) uses
  `data` (statistical modeling convention).

- Graph parameters:

  `svd()` uses `graph_U`/`graph_V` matching SVD factor names (U, V);
  [`nmf()`](https://zdebruine.github.io/RcppML/reference/nmf.md) uses
  `graph_W`/`graph_H`.

- Penalty shape:

  `svd()` accepts scalars (expanded internally to length-2);
  [`nmf()`](https://zdebruine.github.io/RcppML/reference/nmf.md)
  requires explicit `c(w, h)` vectors. The scalar API is simpler for
  per-factor SVD iteration.

- Non-negativity shape:

  `svd()` accepts a single logical;
  [`nmf()`](https://zdebruine.github.io/RcppML/reference/nmf.md)
  requires `c(w, h)`. SVD rarely needs side-specific control.

- Tolerance:

  `svd()` defaults to `tol = 1e-5` (per-factor convergence);
  [`nmf()`](https://zdebruine.github.io/RcppML/reference/nmf.md)
  defaults to `tol = 1e-4` (global convergence across all factors
  simultaneously).

## Deprecated

`sparse_pca()` is deprecated. Use `svd(center=TRUE, L1=c(0, L1))`
instead.

`nn_pca()` is deprecated. Use `svd(center=TRUE, nonneg=TRUE)` instead.

`svd_pca()` is deprecated. Use `svd()` instead.

## See also

[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md),
[`project`](https://zdebruine.github.io/RcppML/reference/project.md),
[`pca`](https://zdebruine.github.io/RcppML/reference/pca.md)

## Examples

``` r
if (FALSE) { # \dontrun{
library(RcppML)
data(iris)
A <- as.matrix(iris[, 1:4])

# Standard SVD (3 factors)
s <- svd(A, k = 3)

# PCA with centering (convenience wrapper)
p <- pca(A, k = 3)

# Sparse PCA
sp <- sparse_pca(A, k = 3, L1 = 0.5)

# Non-negative PCA
nn <- nn_pca(A, k = 3)

# Auto-rank PCA
ar <- svd(A, k = "auto", center = TRUE, verbose = TRUE)

# Krylov with angular penalty
ka <- svd(A, k = 10, method = "krylov", angular = 0.1)

# Robust PCA (outlier-resistant)
rp <- svd(A, k = 3, center = TRUE, robust = TRUE, verbose = TRUE)
} # }
```
