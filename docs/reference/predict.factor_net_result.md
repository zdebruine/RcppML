# Project new data through a trained factor network

For single-layer results, projects new samples using the trained W and
d. For multi-layer results, chains through each layer.

## Usage

``` r
# S3 method for class 'factor_net_result'
predict(object, newdata, ...)
```

## Arguments

- object:

  A `factor_net_result` from
  [`fit()`](https://zdebruine.github.io/RcppML/reference/fit.md).

- newdata:

  A matrix (features x samples) to project.

- ...:

  Additional arguments (currently unused).

## Value

For single-layer: a matrix (k x n_new). For multi-layer: a list of H
matrices, one per layer.

## See also

[`fit`](https://zdebruine.github.io/RcppML/reference/fit.md),
[`factor_net`](https://zdebruine.github.io/RcppML/reference/factor_net.md)
