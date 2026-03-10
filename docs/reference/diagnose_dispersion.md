# Diagnose dispersion mode

Determines whether dispersion should be estimated per-row, per-column,
or globally by examining the coefficient of variation of per-row and
per-column dispersion estimates.

## Usage

``` r
diagnose_dispersion(data, model, cv_threshold = 0.5, min_mu = 1e-06)
```

## Arguments

- data:

  Input matrix (sparse or dense)

- model:

  A fitted NMF model

- cv_threshold:

  CV threshold for declaring structured dispersion. Default 0.5.

- min_mu:

  Floor for predicted values. Default 1e-6.

## Value

A list with:

- mode:

  Recommended DispersionMode: "global", "per_row", or "per_col"

- global_phi:

  Global dispersion estimate

- row_cv:

  CV of per-row dispersion estimates

- col_cv:

  CV of per-col dispersion estimates

## Details

Computes moment-based dispersion estimates \\\hat{\phi} = r^2 / \mu^p\\
(where \\p\\ is determined by the distribution) per row and per column.
If the coefficient of variation (CV) of per-row estimates exceeds
`cv_threshold`, per-row dispersion is recommended; similarly for
per-column. If both CVs are low, global dispersion suffices.

## See also

[`auto_nmf_distribution`](https://zdebruine.github.io/RcppML/reference/auto_nmf_distribution.md),
[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md)

## Examples

``` r
# \donttest{
data(aml)
model <- nmf(aml, k = 3, maxit = 10, seed = 1)
disp <- diagnose_dispersion(aml, model)
disp$mode
#> [1] "per_row"
# }
```
