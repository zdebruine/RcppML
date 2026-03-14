# GPU Iterative Whitened Centroid

Iterative re-whitening: T rounds of ZCA -\> centroid -\> nullspace.
Re-computes covariance at each iteration for progressive refinement.

## Usage

``` r
gpu_iter_whitened(
  H,
  labels,
  batch = NULL,
  T_iter = 3L,
  lambda_c = 0.8,
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

  factor of batch labels (NULL if no nullspace)

- T_iter:

  number of iterations

- lambda_c:

  centroid strength

- lambda_n:

  nullspace strength (0 = skip)

- alpha:

  annealing factor (lambda decreases by alpha^t each iteration)

## Value

corrected n x k embedding matrix
