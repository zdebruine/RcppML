# Element-wise H addition (skip/residual connection)

Creates a node that adds H factors element-wise from multiple branches.
All branches must have the same rank k.

## Usage

``` r
factor_add(...)
```

## Arguments

- ...:

  Two or more `fn_node` objects with matching ranks.

## Value

An `fn_node` of type "add".

## See also

[`factor_concat`](https://zdebruine.github.io/RcppML/reference/factor_concat.md),
[`factor_shared`](https://zdebruine.github.io/RcppML/reference/factor_shared.md),
[`factor_net`](https://zdebruine.github.io/RcppML/reference/factor_net.md)

## Examples

``` r
data(aml)
inp <- factor_input(aml)
branch1 <- nmf_layer(inp, k = 5)
branch2 <- nmf_layer(inp, k = 5)
added <- factor_add(branch1, branch2)
```
