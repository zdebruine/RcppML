# Evaluate an NMF model

Calculate loss for an NMF model using the specified loss function,
accounting for any masking schemes requested during fitting.

## Usage

``` r
evaluate(x, ...)

# S4 method for class 'nmf'
evaluate(
  x,
  data,
  mask = NULL,
  missing_only = FALSE,
  loss = c("mse", "gp"),
  test_fraction = 0,
  test_seed = NULL,
  eval_set = c("all", "test", "train"),
  threads = 0,
  verbose = FALSE,
  ...
)
```

## Arguments

- x:

  fitted model, class `nmf`, generally the result of calling `nmf`, with
  models of equal dimensions as `data`

- ...:

  advanced parameters. See **Advanced Parameters** section.

- data:

  dense or sparse matrix of features in rows and samples in columns.
  Prefer `matrix` or `Matrix::dgCMatrix`, respectively. Also accepts a
  file path (character string) which will be auto-loaded based on
  extension.

- mask:

  missing data mask. Accepts: `NULL` (no masking), `"zeros"` (mask
  zeros), `"NA"` (mask NAs), a dgCMatrix/matrix (custom mask), or
  `list("zeros", <matrix>)` to mask zeros and use a custom mask
  simultaneously.

- missing_only:

  calculate loss only for missing values specified as a matrix in `mask`

- loss:

  loss function to use: "mse" (Mean Squared Error, default) or "gp"
  (Generalized Poisson / KL divergence)

- test_fraction:

  fraction of entries to hold out as test set (default 0 = disabled).
  When \> 0, creates a random mask for test/train split.

- test_seed:

  seed for test set generation. If NULL, attempts to use test mask from
  model's @misc\$test_mask if available.

- eval_set:

  which set to evaluate: "all" (default), "test" (held-out entries
  only), or "train" (non-held-out entries only). Only used when
  test_fraction \> 0 or test mask exists in model.

- threads:

  number of threads for OpenMP parallelization (default 0 = all
  available)

- verbose:

  print progress information (default FALSE)

## Value

A single numeric value: the loss (MSE or GP/KL divergence) of the model
on the data.

## See also

[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md)

## Examples

``` r
# \donttest{
library(Matrix)
A <- rsparsematrix(100, 50, 0.1)
model <- nmf(A, 3, seed = 1, maxit = 50, tol = 1e-4)
evaluate(model, A)  # MSE
#> [1] 0.09021044
# }
```
