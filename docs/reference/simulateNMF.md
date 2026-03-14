# Simulate an NMF dataset

Generate a random nonnegative matrix with known factor structure for
benchmarking NMF recovery. Uses block-diagonal construction: each factor
owns a disjoint subset of features (rows) and dominates a disjoint
subset of samples (columns), with small cross-talk for realism. This
produces clearly recoverable factors even at moderate noise levels.
Inspired by `NMF::syntheticNMF`.

## Usage

``` r
simulateNMF(nrow, ncol, k, noise = 0.5, dropout = 0, seed = NULL)
```

## Arguments

- nrow:

  number of rows (features)

- ncol:

  number of columns (samples)

- k:

  true rank (number of factors)

- noise:

  noise level as a multiplier on the mean signal. A value of 1.0 means
  the noise standard deviation equals the mean signal value. Default:
  0.5.

- dropout:

  fraction of entries to set to zero (0 = no dropout). Default: 0.

- seed:

  seed for random number generation

## Value

list of dense matrix `A` and true `w` and `h` models

## See also

[`simulateSwimmer`](https://zdebruine.github.io/RcppML/reference/simulateSwimmer.md),
[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md)

## Examples

``` r
data <- simulateNMF(50, 30, k = 3, noise = 0.1, seed = 42)
dim(data$A)  # 50 x 3
#> [1] 50 30
dim(data$w)  # 50 x 3
#> [1] 50  3
dim(data$h)  # 3 x 30
#> [1]  3 30
```
