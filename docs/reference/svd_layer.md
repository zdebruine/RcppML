# Create an SVD/PCA factorization layer

Creates an unconstrained (SVD/PCA) factorization layer. No
non-negativity constraint; factors are signed.

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
  W = NULL,
  H = NULL,
  name = NULL
)
```

## Arguments

- input:

  An `fn_node` (input, shared, condition, or another layer).

- k:

  Factorization rank.

- L1:

  L1 penalty (shared by W and H unless overridden). Default 0.

- L2:

  L2 penalty. Default 0.

- L21:

  Group sparsity penalty. Default 0.

- angular:

  Orthogonality penalty. Default 0.

- upper_bound:

  Box constraint. Default 0.

- W:

  Optional [`W()`](https://zdebruine.github.io/RcppML/reference/W.md)
  config to override W-specific settings.

- H:

  Optional [`H()`](https://zdebruine.github.io/RcppML/reference/W.md)
  config to override H-specific settings.

- name:

  Optional layer name (for results access).

## Value

An `fn_node` of type "svd_layer".

## Examples

``` r
data(aml)
inp <- factor_input(aml)
layer <- svd_layer(inp, k = 5)
```
