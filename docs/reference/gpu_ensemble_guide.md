# GPU Ensemble Guide

Averages embeddings from ZCA whitened centroid (A6) and iterative
centroid+nullspace (A7) to get the best of both.

## Usage

``` r
gpu_ensemble_guide(
  H,
  labels,
  batch,
  lambda_c_a6 = 0.8,
  lambda_n_a6 = 1,
  T_iter_a7 = 5L,
  lambda_c_a7 = 0.3,
  lambda_n_a7 = 0.5,
  alpha_a7 = 0.7
)
```

## Arguments

- H:

  n x k embedding matrix

- labels:

  factor of class labels

- batch:

  factor of batch labels

- lambda_c_a6:

  centroid strength for A6 path

- lambda_n_a6:

  nullspace strength for A6 path

- T_iter_a7:

  iterations for A7 path

- lambda_c_a7:

  centroid strength for A7 path

- lambda_n_a7:

  nullspace strength for A7 path

- alpha_a7:

  annealing for A7 path

## Value

corrected n x k embedding matrix
