# Simulate an NMF dataset

Generate a random matrix that follows some defined NMF model to test NMF
factorizations. Adapts methods from `NMF::syntheticNMF`.

## Usage

``` r
simulateNMF(nrow, ncol, k, noise = 0.5, dropout = 0.5, seed = NULL)
```

## Arguments

- nrow:

  number of rows

- ncol:

  number of columns

- k:

  true rank of simulated model

- noise:

  standard deviation of Gaussian noise centered at 0 to add to input
  matrix. Any negative values after noise addition are set to 0.

- dropout:

  density of dropout events

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
dim(data$A)  # 50 x 30
#> [1] 50 30
dim(data$w)  # 50 x 3
#> [1] 50  3
dim(data$h)  # 3 x 30
#> [1]  3 30
```
