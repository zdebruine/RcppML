# Plot training log

Produces a multi-panel plot of loss, classifier metrics, and factor
norms over training iterations.

## Usage

``` r
# S3 method for class 'training_logger'
plot(x, ...)
```

## Arguments

- x:

  A `training_logger` object.

- ...:

  Ignored.

## Value

Invisibly returns `NULL`. Called for its side effect (plotting).

## See also

[`training_logger`](https://zdebruine.github.io/RcppML/reference/training_logger.md)
