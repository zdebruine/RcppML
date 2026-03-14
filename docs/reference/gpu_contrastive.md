# GPU Contrastive Correction

Same-class attraction + cross-class repulsion.

## Usage

``` r
gpu_contrastive(
  H,
  labels,
  lambda_pos = 0.1,
  lambda_neg = 0.05,
  k_pos = 10,
  k_neg = 5,
  seed = 42
)
```

## Arguments

- H:

  n x k embedding matrix

- labels:

  factor of class labels

- lambda_pos:

  positive (attraction) strength

- lambda_neg:

  negative (repulsion) strength

- k_pos:

  number of positive neighbors

- k_neg:

  number of negative neighbors

- seed:

  random seed

## Value

corrected n x k embedding matrix
