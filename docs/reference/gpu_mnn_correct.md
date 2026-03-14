# GPU MNN Batch Correction

Mutual nearest neighbor anchor-based batch correction.

## Usage

``` r
gpu_mnn_correct(H, batch, lambda = 0.5, k_mnn = 10, seed = 42)
```

## Arguments

- H:

  n x k embedding matrix

- batch:

  factor of batch labels

- lambda:

  correction strength

- k_mnn:

  number of MNN neighbors

- seed:

  random seed

## Value

corrected n x k embedding matrix
