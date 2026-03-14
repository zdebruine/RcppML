# Create an SVD/PCA factorization layer

Creates an unconstrained (SVD/PCA) factorization layer. No
non-negativity constraint by default; factors are signed.

## Usage

``` r
svd_layer(
  input,
  k,
  L1 = 0,
  L2 = 0,
  L21 = 0,
  angular = 0,
  upper_bound = 0,
  center = FALSE,
  scale = FALSE,
  method = c("auto", "deflation", "krylov", "lanczos", "irlba", "randomized"),
  mask = NULL,
  robust = FALSE,
  W = NULL,
  H = NULL,
  name = NULL,
  ...
)
```

## Arguments

- input:

  An `fn_node` (input, shared, condition, or another layer).

- k:

  Factorization rank.

- L1:

  L1 penalty (shared by U and V unless overridden). Default 0.

- L2:

  L2 penalty. Default 0.

- L21:

  Group sparsity penalty. Default 0.

- angular:

  Orthogonality penalty. Default 0.

- upper_bound:

  Box constraint. Default 0.

- center:

  Center columns before factorization (PCA mode). Default FALSE.

- scale:

  Scale columns to unit variance. Default FALSE.

- method:

  SVD algorithm: `"auto"`, `"deflation"`, `"krylov"`, `"lanczos"`,
  `"irlba"`, or `"randomized"`. Default `"auto"`.

- mask:

  Masking mode: NULL (none), `"zeros"`, or a sparse mask matrix. See
  [`svd`](https://zdebruine.github.io/RcppML/reference/svd.md) for
  details.

- robust:

  Use robust loss. Default FALSE.

- W:

  Optional [`W()`](https://zdebruine.github.io/RcppML/reference/W.md)
  config to override U-specific settings.

- H:

  Optional [`H()`](https://zdebruine.github.io/RcppML/reference/W.md)
  config to override V-specific settings.

- name:

  Optional layer name (for results access).

- ...:

  Additional arguments forwarded to
  [`svd`](https://zdebruine.github.io/RcppML/reference/svd.md) at fit
  time. Supports: `convergence`, `cv_seed`, `patience`, `k_max`,
  `graph_U`, `graph_V`, `graph_lambda`, `threads`. See
  [`?svd`](https://zdebruine.github.io/RcppML/reference/svd.md) for the
  complete list.

## Value

An `fn_node` of type "svd_layer".

## Details

SVD-specific parameters (`center`, `scale`, `method`) control the SVD
algorithm directly. Regularization parameters and `...` are forwarded to
[`svd`](https://zdebruine.github.io/RcppML/reference/svd.md) at fit
time.

## See also

[`nmf_layer`](https://zdebruine.github.io/RcppML/reference/nmf_layer.md),
[`factor_input`](https://zdebruine.github.io/RcppML/reference/factor_input.md),
[`factor_net`](https://zdebruine.github.io/RcppML/reference/factor_net.md),
[`svd`](https://zdebruine.github.io/RcppML/reference/svd.md)

## Examples

``` r
data(aml)
inp <- factor_input(aml)
layer <- svd_layer(inp, k = 5)

# PCA with centering and scaling
pca_layer <- svd_layer(inp, k = 10, center = TRUE, scale = TRUE)
```
