# GPU Double Whitened Centroid

Two-pass: ZCA-\>centroid-\>re-ZCA-\>centroid + optional nullspace.

## Usage

``` r
gpu_double_whitened(H, labels, lambda_c = 0.8, lambda_n = 0, batch = NULL)
```

## Arguments

- H:

  n x k embedding matrix

- labels:

  factor of class labels

- lambda_c:

  centroid strength

- lambda_n:

  nullspace strength (0 = skip)

- batch:

  factor of batch labels (NULL if no nullspace)

## Value

corrected n x k embedding matrix
