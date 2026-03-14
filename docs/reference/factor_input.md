# Create an input node for a factorization network

Wraps a data matrix as a graph input node. The matrix can be dense or
sparse (dgCMatrix), or a file path to a .spz file for streaming.

## Usage

``` r
factor_input(data, name = NULL)
```

## Arguments

- data:

  A numeric matrix, sparse matrix (dgCMatrix), or character string path
  to a .spz file for out-of-core streaming NMF.

- name:

  Optional name for the input (used in multi-modal results).

## Value

An `fn_node` object of type "input".

## See also

[`nmf_layer`](https://zdebruine.github.io/RcppML/reference/nmf_layer.md),
[`svd_layer`](https://zdebruine.github.io/RcppML/reference/svd_layer.md),
[`factor_net`](https://zdebruine.github.io/RcppML/reference/factor_net.md)

## Examples

``` r
data(aml)
inp <- factor_input(aml, name = "aml")
inp
#> fn_node: input (aml)
```
