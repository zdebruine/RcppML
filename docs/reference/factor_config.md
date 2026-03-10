# Global configuration for a factorization network

Sets network-wide defaults. Layer-level and factor-level settings
override these where specified.

## Usage

``` r
factor_config(
  maxit = 100,
  tol = 1e-04,
  loss = c("mse", "mae", "huber", "kl", "gp", "nb", "gamma", "inverse_gaussian",
    "tweedie"),
  verbose = FALSE,
  seed = NULL,
  threads = 0,
  norm = c("L1", "L2", "none"),
  solver = c("auto", "cholesky", "cd"),
  resource = "auto",
  test_fraction = 0,
  cv_seed = 0L,
  mask_zeros = FALSE,
  patience = 5L,
  holdout_fraction,
  cv_patience
)
```

## Arguments

- maxit:

  Maximum ALS iterations per layer. Default 100.

- tol:

  Convergence tolerance. Default 1e-4.

- loss:

  Loss function: "mse", "mae", "huber", "kl", "gp", "nb", "gamma",
  "inverse_gaussian", "tweedie". Default "mse".

- verbose:

  Print per-iteration diagnostics. Default FALSE.

- seed:

  Random seed. Default NULL (auto).

- threads:

  Number of CPU threads (0 = all). Default 0.

- norm:

  Normalization type: "L1", "L2", "none". Default "L1".

- solver:

  NNLS solver: "auto", "cholesky", "cd". Default "auto".

- resource:

  Resource override: "auto", "cpu", "gpu". Default "auto".

- test_fraction:

  Fraction of entries held out for cross-validation test set. 0 = no CV
  (standard fit). Default 0.

- cv_seed:

  Separate seed for CV mask (0 = derive from `seed`).

- mask_zeros:

  If TRUE, only non-zero entries are in the test set (suitable for
  recommendation systems). If FALSE, all entries including zeros may
  appear in the test set. Default FALSE.

- patience:

  Number of iterations without test-loss improvement before early
  stopping during CV. Default 5.

- holdout_fraction:

  Deprecated. Use `test_fraction` instead.

- cv_patience:

  Deprecated. Use `patience` instead.

## Value

An `fn_global_config` object.

## Details

**Unsupported combinations:**

- Cholesky solver with non-MSE losses (GP, NB, Gamma, etc.) is not
  supported. Use `solver="cd"` or `solver="auto"`.

- Zero-inflation is only available with GP and NB losses at the layer
  level; it cannot be set globally in `factor_config()`.

## See also

[`factor_net`](https://zdebruine.github.io/RcppML/reference/factor_net.md),
[`nmf_layer`](https://zdebruine.github.io/RcppML/reference/nmf_layer.md),
[`fit`](https://zdebruine.github.io/RcppML/reference/fit.md)

## Examples

``` r
# Default config
cfg <- factor_config()

# Poisson NMF with cross-validation
cfg <- factor_config(loss = "gp", test_fraction = 0.1, maxit = 50)
```
