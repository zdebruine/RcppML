# GPU Iterative Shrinkage Whitened Centroid

Iterative (ZCA+centroid+nullspace) with shrinkage covariance.

## Usage

``` r
gpu_iter_shrinkage_whitened(
  H,
  labels,
  batch = NULL,
  T_iter = 3L,
  lambda_c = 0.8,
  alpha_shrink = 0.2,
  lambda_n = 0,
  alpha = 1
)
```

## Arguments

- H:

  n x k embedding matrix

- labels:

  factor of class labels

- batch:

  factor of batch labels

- T_iter:

  number of iterations

- lambda_c:

  centroid strength

- alpha_shrink:

  shrinkage parameter

- lambda_n:

  nullspace strength

- alpha:

  annealing factor

## Value

corrected n x k embedding matrix
