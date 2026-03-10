# Biplot for NMF factors

Produces a biplot from the output of
[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md)

## Usage

``` r
# S4 method for class 'nmf'
biplot(x, factors = c(1, 2), matrix = "w", group_by = NULL, ...)
```

## Arguments

- x:

  an object of class "`nmf`"

- factors:

  length 2 vector specifying factors to plot.

- matrix:

  either `w` or `h`

- group_by:

  a discrete factor giving groupings for samples or features. Must be of
  the same length as number of samples in `object$h` or number of
  features in `object$w`.

- ...:

  for consistency with `biplot` generic

## Value

ggplot2 object

## See also

[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md)

## Examples

``` r
# \donttest{
library(Matrix)
A <- rsparsematrix(100, 50, 0.1)
model <- nmf(A, k = 3, seed = 42, tol = 1e-2, maxit = 10)
if (requireNamespace("ggplot2", quietly = TRUE)) {
  biplot(model, factors = c(1, 2))
}

# }
```
