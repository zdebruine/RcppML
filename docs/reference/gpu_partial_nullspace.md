# GPU Partial Nullspace

Cell-type conditioned batch removal.

## Usage

``` r
gpu_partial_nullspace(H, labels, batch, lambda = 1)
```

## Arguments

- H:

  n x k embedding matrix

- labels:

  factor of cell-type labels

- batch:

  factor of batch labels

- lambda:

  projection strength

## Value

corrected n x k embedding matrix
