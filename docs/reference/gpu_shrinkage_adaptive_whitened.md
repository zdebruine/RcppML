# GPU Shrinkage Adaptive Whitened Centroid

Shrinkage ZCA + silhouette-adaptive per-class lambda.

## Usage

``` r
gpu_shrinkage_adaptive_whitened(
  H,
  labels,
  lambda_max = 0.8,
  alpha_shrink = 0.2,
  lambda_n = 0,
  batch = NULL
)
```

## Arguments

- H:

  n x k embedding matrix

- labels:

  factor of class labels

- lambda_max:

  maximum centroid strength

- alpha_shrink:

  shrinkage parameter

- lambda_n:

  nullspace strength

- batch:

  factor of batch labels

## Value

corrected n x k embedding matrix
