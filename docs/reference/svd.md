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
  L1 = 0,
  L2 = 0,
  nonneg = FALSE,
  upper_bound = 0,
  test_fraction = 0,
  mask = NULL,
  robust = FALSE,
  resource = "auto",
  method = "auto",
  ...
)
```

## Arguments

- A:

  Input matrix. May be dense (`matrix`), sparse (`dgCMatrix`), or a path
  to a `.spz` file for out-of-core streaming SVD.

- k:

  Number of factors (rank). Use `"auto"` for automatic rank selection
  via cross-validation. Default: 10.

- tol:

  Convergence tolerance per rank-1 subproblem. Default: 1e-5.

- maxit:

  Maximum ALS iterations per factor. Default: 200.

- center:

  If `TRUE`, subtract row means (PCA mode). Default: `FALSE`.

- scale:

  If `TRUE`, divide each row by its standard deviation after centering.
  Default: `FALSE`.

- verbose:

  Print per-factor diagnostics. Default: `FALSE`.

- seed:

  Random seed for initialization and cross-validation. Default: NULL.

- L1:

  L1 (lasso) penalty. Single value or length-2 vector. Default: 0.

- L2:

  L2 (ridge) penalty. Single value or length-2 vector. Default: 0.

- nonneg:

  Non-negativity constraints. Single logical or length-2 vector.
  Default: `FALSE`.

- upper_bound:

  Upper bound constraints. Single value or length-2 vector. Default: 0.

- test_fraction:

  Fraction of entries to hold out for cross-validation. Default: 0.

- mask:

  Masking control. `NULL` (default, no masking), `"zeros"` (mask zero
  entries — only non-zero entries can be holdout in CV), a `dgCMatrix`
  (custom mask matrix where non-zero entries are excluded), or
  `list("zeros", <dgCMatrix>)` to combine both.

- robust:

  Robustness control: `FALSE` (default), `TRUE` (Huber delta=1.345),
  `"mae"` (near-MAE), or a positive numeric Huber delta.

- resource:

  Compute backend: `"auto"` (default), `"cpu"`, or `"gpu"`.

- method:

  Algorithm: `"auto"` (default), `"deflation"`, `"krylov"`, `"lanczos"`,
  `"irlba"`, `"randomized"`.

- ...:

  Advanced parameters. See **Advanced Parameters** section.

## Value

An S4 object of class `svd` with slots:

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
  Faster than deflation for larger k. Supports all regularization except
  robust. CV is evaluated post-convergence.

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
[`base::svd()`](https://rdrr.io/r/base/svd.html) explicitly.

## Auto-rank

When `k = "auto"`, a speckled holdout set is created and test MSE is
evaluated after each factor. Rank selection stops when test MSE fails to
improve for `patience` consecutive factors. The returned model is
truncated to the best rank.

## Advanced Parameters (via `...`)

The following parameters can be passed via `...`:

- `L21`:

  L2,1 (group sparsity) penalty. Single value or length-2 vector
  (default 0).

- `angular`:

  Angular (orthogonality) penalty. Single value or length-2 vector
  (default 0).

- `graph_U`:

  Sparse graph Laplacian for features (m x m). Default NULL.

- `graph_V`:

  Sparse graph Laplacian for samples (n x n). Default NULL.

- `graph_lambda`:

  Graph regularization strength. Single value or length-2 (default 0).

- `convergence`:

  Convergence criterion: `"factor"` (default) or `"global"`.

- `cv_seed`:

  Separate seed for holdout mask. Default NULL.

- `patience`:

  Auto-rank non-improving factor patience (default 3).

- `k_max`:

  Maximum rank for auto-rank mode (default 50).

- `threads`:

  Number of OpenMP threads. 0 = all (default 0).

## See also

[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md),
[`pca`](https://zdebruine.github.io/RcppML/reference/pca.md)

## Examples

``` r
# \donttest{
library(RcppML)
data(iris)
A <- as.matrix(iris[, 1:4])

# Standard SVD (3 factors)
s <- svd(A, k = 3)

# PCA with centering (convenience wrapper)
p <- pca(A, k = 3)

# Auto-rank PCA
ar <- svd(A, k = "auto", center = TRUE, verbose = TRUE)
#>   auto method: deflation (k=50, gpu=FALSE)
#>   CV: 39 test entries held out (39 zeroed), denom_correction=0.9500
#>   Factor 1: sigma=4.0891e+01  iters=4  test_mse=2.966425e+00
#>   Factor 2: sigma=1.9770e+01  iters=10  test_mse=5.827467e+00
#>   Factor 3: sigma=1.3789e+01  iters=6  test_mse=1.264707e+01
#>   Factor 4: sigma=1.1120e+01  iters=2  test_mse=1.705114e+01
#>   Auto-rank: patience exhausted at factor 4, best rank = 1 (test_mse=2.966425e+00)

# Robust PCA (outlier-resistant)
rp <- svd(A, k = 3, center = TRUE, robust = TRUE, verbose = TRUE)
#>   auto method: deflation (k=3, gpu=FALSE)
#>   Robust SVD: Huber delta=1.3450, IRLS reweighting active
#>   Factor 1: sigma=4.0970e+01  iters=2
#>   Factor 2: sigma=1.7062e+01  iters=3
#>   Factor 3: sigma=2.1213e+00  iters=3
# }
```
