# Compile a factorization network

Validates the graph topology, resolves config inheritance, and returns a
compiled network ready for
[`fit()`](https://zdebruine.github.io/RcppML/reference/fit.md).

## Usage

``` r
factor_net(inputs, output, config = factor_config())
```

## Arguments

- inputs:

  A single `fn_node` (input) or a list of input nodes.

- output:

  The output `fn_node` (typically a layer).

- config:

  A `fn_global_config` from
  [`factor_config()`](https://zdebruine.github.io/RcppML/reference/factor_config.md).
  Default uses
  [`factor_config()`](https://zdebruine.github.io/RcppML/reference/factor_config.md)
  defaults.

## Value

A `factor_net` object.

## See also

[`fit`](https://zdebruine.github.io/RcppML/reference/fit.md),
[`factor_config`](https://zdebruine.github.io/RcppML/reference/factor_config.md),
[`nmf_layer`](https://zdebruine.github.io/RcppML/reference/nmf_layer.md),
[`svd_layer`](https://zdebruine.github.io/RcppML/reference/svd_layer.md),
[`cross_validate_graph`](https://zdebruine.github.io/RcppML/reference/cross_validate_graph.md)

## Examples

``` r
data(aml)
inp <- factor_input(aml, "aml")
out <- nmf_layer(inp, k = 5)
net <- factor_net(inp, out, config = factor_config(maxit = 10))
net
#> factor_net: 1 layer(s), 1 input(s)
#>   L1: NMF(k=5)
#>   config: maxit=10, tol=0.0001, loss=mse
```
