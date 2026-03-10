# Mean squared error of factor model

Same as the `evaluate` S4 method for the `nmf` class, but allows one to
input the \`w\`, \`d\`, \`h\`, and \`data\` independently.

## Usage

``` r
mse(w, d = NULL, h, data, mask = NULL, missing_only = FALSE, ...)
```

## Arguments

- w:

  feature factor matrix (features as rows)

- d:

  scaling diagonal vector (if applicable)

- h:

  sample factor matrix (samples as columns)

- data:

  dense or sparse matrix of features in rows and samples in columns.
  Prefer `matrix` or `Matrix::dgCMatrix`, respectively. Also accepts a
  file path (character string) which will be auto-loaded based on
  extension.

- mask:

  dense or sparse matrix of values in `data` to handle as missing.
  Alternatively, specify "`zeros`" or "`NA`".

- missing_only:

  only calculate mean squared error at masked values

- ...:

  additional arguments

## Value

A single numeric value: the mean squared error of the factorization.

## Examples

``` r
# \donttest{
data <- simulateNMF(50, 30, k = 3, seed = 1)
model <- nmf(data$A, 3, seed = 1, maxit = 50)
RcppML:::mse(model$w, model$d, model$h, data$A)
#> [1] 0.03836534
# }
```
