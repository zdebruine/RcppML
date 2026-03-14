# Convert training log to data.frame

Converts the internal training log from an NMF run into a tidy data
frame with one row per logged iteration, suitable for plotting
convergence curves.

## Usage

``` r
# S3 method for class 'training_logger'
as.data.frame(x, ...)
```

## Arguments

- x:

  A `training_logger` object (from `nmf(..., log = TRUE)`).

- ...:

  Ignored.

## Value

A data.frame with columns `iteration`, `wall_sec`, and optionally
`total_loss`, per-layer loss columns, and norm-tracking columns.

## See also

[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md),
[`plot.training_logger`](https://zdebruine.github.io/RcppML/reference/plot.training_logger.md)
