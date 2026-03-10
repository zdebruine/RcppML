# Compare Multiple NMF Models

Overlay training curves from multiple NMF models to compare convergence
behavior.

## Usage

``` r
compare_nmf(..., labels = NULL, metric = "loss", smooth = TRUE, span = 0.3)
```

## Arguments

- ...:

  nmf objects to compare

- labels:

  optional character vector of model labels

- metric:

  what to compare: "loss" (default), "sparsity"

- smooth:

  apply smoothing (default TRUE)

- span:

  smoothing span for LOESS (default 0.3)

## Value

A `ggplot2` object comparing the requested metric across models.

## See also

[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md),
[`plot.nmf`](https://zdebruine.github.io/RcppML/reference/plot.nmf.md)

## Examples

``` r
# \donttest{
library(Matrix)
A <- abs(rsparsematrix(100, 50, 0.2))
model1 <- nmf(A, k = 5, L1 = 0.01)
model2 <- nmf(A, k = 5, L1 = 0.1)
compare_nmf(model1, model2, labels = c("L1=0.01", "L1=0.1"))

# }
```
