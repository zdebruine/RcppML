# Auto-select NMF distribution

Fits NMF with multiple loss functions (distributions) and selects the
best based on per-element AIC/BIC. Useful for determining whether count
data is best modeled with Gaussian (MSE), Poisson/GP, or Negative
Binomial loss.

## Usage

``` r
auto_nmf_distribution(
  data,
  k,
  distributions = c("mse", "gp", "nb"),
  criterion = c("bic", "aic"),
  maxit = 50,
  seed = NULL,
  verbose = FALSE,
  ...
)
```

## Arguments

- data:

  Input matrix (dense or sparse dgCMatrix)

- k:

  Factorization rank

- distributions:

  Character vector of distributions to compare. Default:
  `c("mse", "gp", "nb")`

- criterion:

  Selection criterion: `"bic"` (default) or `"aic"`

- maxit:

  Maximum iterations per fit

- seed:

  Random seed for reproducibility

- verbose:

  Print progress and comparison table

- ...:

  Additional arguments passed to
  [`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md)

## Value

A list with:

- loss:

  Character string: name of the best distribution (loss function)

- comparison:

  Data frame with distribution, nll, df, aic, bic, selected

- models:

  Named list of fitted nmf objects

## Details

For each distribution, NMF is fit and the final negative log-likelihood
(NLL) is computed. For GP and NB, the C++ loss is already the total NLL.
For MSE (Gaussian), the C++ loss is the sum of squared errors (SSE),
which is converted to Gaussian NLL: \\\text{NLL} = (N/2)(1 + \log(2\pi
\cdot \text{SSE}/N))\\.

The number of effective parameters is:

- `mse`: \\k(m + n) + 1\\ (factor params + noise variance)

- `gp`: \\k(m + n) + m\\ (factor params + dispersion per row)

- `nb`: \\k(m + n) + m\\ (factor params + size per row)

BIC = \\2 \times \text{NLL} + \text{df} \times \log(N)\\ where \\N\\ is
the number of observations (nonzeros for sparse, all entries for dense).
AIC = \\2 \times \text{NLL} + 2 \times \text{df}\\.

## See also

[`score_test_distribution`](https://zdebruine.github.io/RcppML/reference/score_test_distribution.md),
[`diagnose_zero_inflation`](https://zdebruine.github.io/RcppML/reference/diagnose_zero_inflation.md),
[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md)

## Examples

``` r
# \donttest{
library(Matrix)
set.seed(42)
A <- abs(rsparsematrix(100, 50, 0.3))
result <- auto_nmf_distribution(A, k = 5)
print(result$comparison)
#>   distribution      nll  df      aic       bic selected
#> 1          mse 1791.759 751 5085.518  9075.746     TRUE
#> 2           gp 3184.812 850 8069.623 12585.860    FALSE
#> 3           nb 1487.201 850 4674.401  9190.639    FALSE
cat("Best distribution:", result$loss, "\n")
#> Best distribution: mse 
# }
```
