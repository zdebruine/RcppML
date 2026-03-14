# Concatenate conditioning metadata to a layer's H

Appends columns from a metadata matrix Z to the H factor before passing
downstream. The next layer's W learns to factor out the conditioning
variables.

## Usage

``` r
factor_condition(input, Z)
```

## Arguments

- input:

  An `fn_node` (typically an nmf_layer output).

- Z:

  Conditioning matrix (n x p) or (p x n). Will be oriented to match H
  dimensions.

## Value

An `fn_node` of type "condition".

## See also

[`nmf_layer`](https://zdebruine.github.io/RcppML/reference/nmf_layer.md),
[`factor_net`](https://zdebruine.github.io/RcppML/reference/factor_net.md)

## Examples

``` r
data(aml)
inp <- factor_input(aml)
layer1 <- nmf_layer(inp, k = 5)
# Condition on batch metadata (2 batches)
Z <- matrix(rep(c(1, 0, 0, 1), c(60, 75, 60, 75)), nrow = 135, ncol = 2)
conditioned <- factor_condition(layer1, Z)
```
