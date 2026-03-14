# GPU Soft Centroid Guide

Harmony-style soft assignment centroid shift.

## Usage

``` r
gpu_soft_centroid(H, labels, lambda = 0.5, tau = 0.5)
```

## Arguments

- H:

  n x k embedding matrix

- labels:

  factor of class labels

- lambda:

  shift strength

- tau:

  softmax temperature

## Value

corrected n x k embedding matrix
