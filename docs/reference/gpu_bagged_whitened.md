# GPU Bagged Whitened Centroid (A17)

Bootstrap-bagged ZCA whitening with OAS shrinkage. Averages B bootstrap
ZCA transforms for variance reduction.

## Usage

``` r
gpu_bagged_whitened(
  H,
  labels,
  batch = NULL,
  lambda_c = 0.8,
  lambda_n = 0,
  n_bootstrap = 50L,
  beta_power = 1,
  use_geomed = FALSE
)
```

## Arguments

- H:

  n x k embedding matrix

- labels:

  factor of class labels

- batch:

  factor of batch labels (NULL if no nullspace)

- lambda_c:

  centroid strength

- lambda_n:

  nullspace strength

- n_bootstrap:

  number of bootstrap resamples

- beta_power:

  whitening power

- use_geomed:

  use geometric median centroids

## Value

corrected n x k embedding matrix
