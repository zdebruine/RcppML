# GPU Centroid Guide

Shift each cell toward its class centroid, away from grand mean.

## Usage

``` r
gpu_centroid_guide(H, labels, lambda = 0.4)
```

## Arguments

- H:

  n x k embedding matrix

- labels:

  factor of class labels (length n)

- lambda:

  shift strength (0-1)

## Value

corrected n x k embedding matrix
