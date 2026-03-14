# GPU Nullspace Removal

Project out confound-predictive directions from embedding.

## Usage

``` r
gpu_nullspace_removal(H, confound, lambda = 1)
```

## Arguments

- H:

  n x k embedding matrix

- confound:

  factor of confound labels (e.g., batch)

- lambda:

  projection strength (0-1)

## Value

corrected n x k embedding matrix
