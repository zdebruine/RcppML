# Summarize NMF factors

`summary` method for class "`nmf`". Describes metadata representation in
NMF factors. Returns object of class `nmfSummary`. Plot results using
`plot`.

## Usage

``` r
# S4 method for class 'nmf'
summary(object, group_by, stat = "sum", ...)

# S3 method for class 'nmfSummary'
plot(x, ...)
```

## Arguments

- object:

  an object of class "`nmf`", usually, a result of a call to
  [`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md)

- group_by:

  a discrete factor giving groupings for samples or features. Must be of
  the same length as number of samples in `object$h` or number of
  features in `object$w`.

- stat:

  either `sum` (sum of factor weights falling within each group), or
  `mean` (mean factor weight falling within each group).

- ...:

  Additional arguments (unused).

- x:

  `nmfSummary` object, the result of calling `summary` on an `nmf`
  object

## Value

`data.frame` with columns `group`, `factor`, and `stat`

A `ggplot2` object showing factor representation by group.

## See also

[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md),
[`sparsity`](https://zdebruine.github.io/RcppML/reference/sparsity.md)

`summary,nmf-method`,
[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md)

## Examples

``` r
# \donttest{
library(Matrix)
A <- rsparsematrix(100, 50, 0.1)
model <- nmf(A, k = 3, seed = 42, tol = 1e-2, maxit = 10)
groups <- factor(sample(c("A", "B"), 50, replace = TRUE))
s <- summary(model, group_by = groups)
# }

# \donttest{
library(Matrix)
A <- rsparsematrix(100, 50, 0.1)
model <- nmf(A, k = 3, seed = 42, tol = 1e-2, maxit = 10)
groups <- factor(sample(c("A", "B"), 50, replace = TRUE))
s <- summary(model, group_by = groups)
if (requireNamespace("ggplot2", quietly = TRUE)) {
  plot(s)
}

# }
```
