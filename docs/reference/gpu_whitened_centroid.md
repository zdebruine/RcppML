# GPU Whitened Centroid Guide

ZCA whitening followed by centroid guide and optional nullspace.

## Usage

``` r
gpu_whitened_centroid(H, labels, lambda_c = 0.5, lambda_n = 0, batch = NULL)
```

## Arguments

- H:

  n x k embedding matrix

- labels:

  factor of class labels

- lambda_c:

  centroid shift strength

- lambda_n:

  nullspace removal strength (0 = skip)

- batch:

  factor of batch labels (NULL if no nullspace)

## Value

corrected n x k embedding matrix
