# Score-test distribution diagnostic

Given a baseline MSE-fitted NMF model and the original data, computes
score-test statistics for the power-variance family (\\V(\mu) = \mu^p\\)
to determine the best-fitting distribution without refitting. Optionally
tests for NB overdispersion.

## Usage

``` r
score_test_distribution(
  data,
  model,
  powers = c(0, 1, 2, 3),
  test_nb = TRUE,
  min_mu = 1e-06
)
```

## Arguments

- data:

  Original data matrix (sparse or dense)

- model:

  A fitted NMF object (from
  [`nmf()`](https://zdebruine.github.io/RcppML/reference/nmf.md) with
  `loss="mse"`)

- powers:

  Numeric vector of variance powers to test. Default `c(0, 1, 2, 3)`
  covers Gaussian, Poisson, Gamma, Inverse Gaussian.

- test_nb:

  Logical; if `TRUE` and data is integer-valued, also test the NB
  overdispersion diagnostic.

- min_mu:

  Floor for predicted values to avoid division by zero. Default 1e-6.

## Value

A list with:

- scores:

  Data frame with columns `power`, `T_stat`, `abs_T`, `distribution`
  (label)

- best_power:

  Numeric: the power p with smallest `|T_p|`

- best_distribution:

  Character: name of the best-matching distribution

- nb_diagnostic:

  If `test_nb=TRUE`: list with `T_NB` and `overdispersed` (logical)

## Details

The score test statistic for variance power \\p\\ is: \$\$T_p =
\text{mean}\left(\frac{r\_{ij}^2}{\mu\_{ij}^p} - 1\right)\$\$ where
\\r\_{ij} = x\_{ij} - \mu\_{ij}\\ are residuals and \\\mu\_{ij} =
(WH)\_{ij}\\ are predicted values.

Under the correct model, \\E\[T_p\] = 0\\. The power minimizing
\\\|T_p\|\\ best matches the observed variance-mean relationship.

The NB diagnostic tests for quadratic overdispersion: \$\$T\_{NB} =
\text{mean}\left(\frac{r\_{ij}^2 - \mu\_{ij}}{\mu\_{ij}^2}\right)\$\$ If
\\T\_{NB} \> 0.1\\, there is substantial overdispersion beyond Poisson,
suggesting NB may be preferable to GP.

## See also

[`auto_nmf_distribution`](https://zdebruine.github.io/RcppML/reference/auto_nmf_distribution.md),
[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md)

## Examples

``` r
# \donttest{
A <- abs(rsparsematrix(200, 100, 0.3))
model <- nmf(A, k = 5, loss = "mse")
diag <- score_test_distribution(A, model)
print(diag$scores)
#>   power        T_stat        abs_T     distribution
#> 1     0 -4.365022e-01 4.365022e-01         gaussian
#> 2     1  5.237193e+02 5.237193e+02               gp
#> 3     2  5.222982e+08 5.222982e+08            gamma
#> 4     3  5.222981e+14 5.222981e+14 inverse_gaussian
cat("Best distribution:", diag$best_distribution, "\n")
#> Best distribution: gaussian 
# }
```
