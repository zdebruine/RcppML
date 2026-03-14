# GPU Whitened Adaptive Centroid

ZCA whitening followed by silhouette-weighted per-class centroid.
Classes with worse separation receive stronger correction.

## Usage

``` r
gpu_whitened_adaptive(H, labels, lambda_max = 0.8, lambda_n = 0, batch = NULL)
```

## Arguments

- H:

  n x k embedding matrix

- labels:

  factor of class labels

- lambda_max:

  maximum centroid shift strength

- lambda_n:

  nullspace strength (0 = skip)

- batch:

  factor of batch labels (NULL if no nullspace)

## Value

corrected n x k embedding matrix
