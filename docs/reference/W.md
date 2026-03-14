# Per-factor configuration for factorization layers

Use `W()` and `H()` inside
[`nmf_layer()`](https://zdebruine.github.io/RcppML/reference/nmf_layer.md)
or
[`svd_layer()`](https://zdebruine.github.io/RcppML/reference/svd_layer.md)
to set factor-specific regularization, constraints, and targets.
Parameters not set inherit from the layer defaults.

## Usage

``` r
W(
  L1 = NULL,
  L2 = NULL,
  L21 = NULL,
  angular = NULL,
  upper_bound = NULL,
  nonneg = NULL,
  graph = NULL,
  graph_lambda = NULL,
  target = NULL,
  target_lambda = NULL
)

H(
  L1 = NULL,
  L2 = NULL,
  L21 = NULL,
  angular = NULL,
  upper_bound = NULL,
  nonneg = NULL,
  graph = NULL,
  graph_lambda = NULL,
  target = NULL,
  target_lambda = NULL
)
```

## Arguments

- L1:

  L1 (lasso) penalty. Default 0.

- L2:

  L2 (ridge) penalty. Default 0.

- L21:

  Group sparsity penalty. Default 0.

- angular:

  Orthogonality penalty. Default 0.

- upper_bound:

  Box constraint upper bound (0 = none). Default 0.

- nonneg:

  Non-negativity constraint. Default TRUE for NMF, FALSE for SVD.

- graph:

  Sparse graph Laplacian matrix (or NULL).

- graph_lambda:

  Graph regularization strength. Default 0.

- target:

  Target matrix for regularization (k x n for H, k x m for W). See
  [`compute_target`](https://zdebruine.github.io/RcppML/reference/compute_target.md)
  for constructing targets from labels.

- target_lambda:

  Target regularization strength. Default 0.

## Value

An `fn_factor_config` object.

## See also

[`factor_config`](https://zdebruine.github.io/RcppML/reference/factor_config.md),
[`nmf_layer`](https://zdebruine.github.io/RcppML/reference/nmf_layer.md),
[`factor_net`](https://zdebruine.github.io/RcppML/reference/factor_net.md)

## Examples

``` r
# Per-factor config for W with L1 sparsity
w_cfg <- W(L1 = 0.1, nonneg = TRUE)
h_cfg <- H(L2 = 0.01)
```
