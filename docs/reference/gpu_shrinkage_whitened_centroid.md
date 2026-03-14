# GPU Shrinkage Whitened Centroid

Ledoit-Wolf-style shrinkage ZCA whitening followed by centroid guide.
Regularizes covariance toward scaled identity: Sigma_reg =
(1-alpha)\*Sigma + alpha\*(trace/k)\*I

## Usage

``` r
gpu_shrinkage_whitened_centroid(
  H,
  labels,
  lambda_c = 0.8,
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

- lambda_c:

  centroid shift strength

- alpha_shrink:

  shrinkage parameter (0 = no shrinkage, 1 = identity)

- lambda_n:

  nullspace strength (0 = skip)

- batch:

  factor of batch labels (NULL if no nullspace)

## Value

corrected n x k embedding matrix
