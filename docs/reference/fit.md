# Fit a factorization network

Executes the compiled graph from
[`factor_net()`](https://zdebruine.github.io/RcppML/reference/factor_net.md).
For single-layer networks, this delegates directly to
[`nmf()`](https://zdebruine.github.io/RcppML/reference/nmf.md) for
maximum performance. Multi-layer networks use R-level alternating least
squares.

## Usage

``` r
fit(object, ...)

# S3 method for class 'factor_net'
fit(object, logger = NULL, ...)
```

## Arguments

- object:

  A `factor_net` object from
  [`factor_net()`](https://zdebruine.github.io/RcppML/reference/factor_net.md).

- ...:

  Additional arguments (currently unused).

- logger:

  Optional `training_logger` from
  [`training_logger()`](https://zdebruine.github.io/RcppML/reference/training_logger.md)
  for per-iteration diagnostics (loss, norms, classifier metrics).

## Value

A `factor_net_result` object. Access per-layer results by name (e.g.
`result$L1`). Each layer is a list with components `W`, `d`, `H`, and
`iterations`. Additional fields: `n_layers`, `multi_modal`,
`total_iterations`, `total_loss`, and `converged`.

## See also

[`factor_net`](https://zdebruine.github.io/RcppML/reference/factor_net.md),
[`predict.factor_net_result`](https://zdebruine.github.io/RcppML/reference/predict.factor_net_result.md),
[`cross_validate_graph`](https://zdebruine.github.io/RcppML/reference/cross_validate_graph.md)

## Examples

``` r
if (FALSE) { # \dontrun{
library(Matrix)
X <- rsparsematrix(100, 50, 0.1)
inp <- factor_input(X, "X")
L1 <- inp |> nmf_layer(k = 5, name = "L1")
net <- factor_net(inputs = inp, output = L1,
                  config = factor_config(maxit = 100, seed = 42))
res <- fit(net)
res$L1$W   # m x k basis matrix
res$L1$H   # k x n coefficient matrix
} # }
```
