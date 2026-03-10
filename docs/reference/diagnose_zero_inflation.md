# Diagnose zero inflation

Tests whether a dataset has excess zeros beyond what the chosen
distribution predicts, and recommends a zero-inflation mode.

## Usage

``` r
diagnose_zero_inflation(data, model, threshold = 0.05)
```

## Arguments

- data:

  Input matrix (sparse or dense)

- model:

  A fitted NMF model (any distribution)

- threshold:

  Minimum excess zero fraction to declare ZI. Default 0.05.

## Value

A list with:

- excess_zero_rate:

  Fraction of zeros exceeding the expected count

- has_zi:

  Logical: TRUE if excess_zero_rate \> threshold

- zi_mode:

  Recommended mode: "none", "row", "col", or "twoway"

- row_excess:

  Per-row excess zero rates

- col_excess:

  Per-col excess zero rates

## Details

Computes the expected number of zeros under the fitted distribution
(Poisson approximation: \\P(X=0) \approx e^{-\mu}\\), compares to the
observed zero count, and recommends ZI if the excess is large.

ZI granularity is determined by whether per-row and per-col excess rates
have high variance (suggesting different rows/cols have different ZI
levels).

## See also

[`auto_nmf_distribution`](https://zdebruine.github.io/RcppML/reference/auto_nmf_distribution.md),
[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md)

## Examples

``` r
# \donttest{
data(aml)
model <- nmf(aml, k = 3, maxit = 10, seed = 1)
zi <- diagnose_zero_inflation(aml, model)
zi$has_zi
#> [1] FALSE
# }
```
