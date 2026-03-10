# Plot Cross-Validation Results

Visualize NMF cross-validation results showing test (and optionally
train) loss across candidate ranks. Useful for selecting the optimal
factorization rank.

## Usage

``` r
# S3 method for class 'nmfCrossValidate'
plot(x, show_train = NULL, point_size = 3, interactive = FALSE, ...)
```

## Arguments

- x:

  object of class `nmfCrossValidate` (a data.frame from
  `nmf(k = 2:10, test_fraction = 0.1)`)

- show_train:

  logical, overlay train loss on the plot (default TRUE if `train_mse`
  column is available and non-NA)

- point_size:

  size of data points (default 3)

- interactive:

  create interactive plotly plot (default FALSE)

- ...:

  additional arguments (unused)

## Value

ggplot2 or plotly object

## See also

[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md)

## Examples

``` r
# \donttest{
library(Matrix)
A <- abs(rsparsematrix(50, 30, 0.3))
cv_result <- nmf(A, k = 2:6, test_fraction = 0.1, cv_seed = 1, maxit = 20)
plot(cv_result)

# }
```
