# GPU Adaptive Centroid Guide

Silhouette-weighted per-class centroid shift.

## Usage

``` r
gpu_adaptive_centroid(H, labels, lambda_max = 0.8)
```

## Arguments

- H:

  n x k embedding matrix

- labels:

  factor of class labels

- lambda_max:

  maximum shift strength

## Value

corrected n x k embedding matrix
