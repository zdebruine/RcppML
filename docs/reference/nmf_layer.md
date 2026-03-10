# Create an NMF factorization layer

Creates a non-negative matrix factorization layer that decomposes its
input into W \* diag(d) \* H with non-negativity constraints.

## Usage

``` r
nmf_layer(
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

An `fn_node` of type "nmf_layer".

## Details

When used with `|>`, the input node is the first argument:
`x |> nmf_layer(k = 64)`.

Layer-level regularization parameters (L1, L2, etc.) apply to both W and
H unless overridden by
[`W()`](https://zdebruine.github.io/RcppML/reference/W.md) or
[`H()`](https://zdebruine.github.io/RcppML/reference/W.md).

## See also

[`svd_layer`](https://zdebruine.github.io/RcppML/reference/svd_layer.md),
[`factor_net`](https://zdebruine.github.io/RcppML/reference/factor_net.md),
[`factor_input`](https://zdebruine.github.io/RcppML/reference/factor_input.md)

## Examples

``` r
data(aml)
inp <- factor_input(aml)
layer <- nmf_layer(inp, k = 5)
layer
#> fn_node: nmf_layer, k=5
```
