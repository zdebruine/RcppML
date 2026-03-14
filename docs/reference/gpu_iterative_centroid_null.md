# GPU Iterative Centroid + Nullspace

Annealed iterative application of centroid guide and nullspace removal.

## Usage

``` r
gpu_iterative_centroid_null(
  H,
  labels,
  batch,
  T_iter = 3,
  lambda_c = 0.3,
  lambda_n = 0.5,
  alpha = 0.7
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

- lambda_n:

  nullspace strength

- alpha:

  annealing factor per iteration

## Value

corrected n x k embedding matrix
