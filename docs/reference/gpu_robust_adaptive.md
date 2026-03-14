# GPU Robust Adaptive Whitened Centroid (A16)

Combines OAS auto-shrinkage, partial whitening power, kNN label
propagation, and geometric median centroids for maximum robustness to
partial/noisy labels.

## Usage

``` r
gpu_robust_adaptive(
  H,
  labels,
  batch = NULL,
  lambda_c = 0.8,
  lambda_n = 0,
  beta_power = 1,
  use_label_prop = FALSE,
  k_nn = 15L,
  label_mask = NULL
)
```

## Arguments

- H:

  n x k embedding matrix

- labels:

  factor of class labels

- batch:

  factor of batch labels (NULL if no nullspace)

- lambda_c:

  centroid shift strength

- lambda_n:

  nullspace strength (0 = skip)

- beta_power:

  whitening power in `[0,1]`. 1=full ZCA, 0.5=half-whiten

- use_label_prop:

  logical; propagate labels via kNN before guide

- k_nn:

  kNN parameter for label propagation

## Value

corrected n x k embedding matrix
