# GPU NMF with Alignment Guide Regularization

Run NMF on GPU with NNLS regularization toward pre-computed targets. The
alignment guide modifies the Gram matrix G and RHS B before each NNLS
solve to steer H and/or W toward target embeddings.

## Usage

``` r
gpu_aligned_nmf(
  A,
  k,
  align_type = 1L,
  lambda_H = 0.1,
  lambda_W = 0,
  lambda_batch = 0,
  lambda_c = 0.3,
  h_target = NULL,
  w_target = NULL,
  labels = NULL,
  n_classes = 0L,
  precision = NULL,
  batch_sub = NULL,
  warmup_iters = 10L,
  anneal_schedule = 0L,
  update_interval = 10L,
  maxit = 100L,
  tol = 1e-05,
  seed = 123L,
  L1 = c(0, 0),
  L2 = c(0, 0),
  cd_maxit = 100L,
  verbose = FALSE,
  nonneg = c(TRUE, TRUE),
  norm = "L1",
  solver_mode = 0L,
  loss_every = 1L,
  patience = 5L,
  w_init = NULL,
  guide_labels = NULL,
  guide_n_classes = 0L,
  guide_lambda = 0
)
```

## Arguments

- A:

  dgCMatrix (sparse, CSC format)

- k:

  Factorization rank

- align_type:

  Integer alignment type (1-9, see details)

- lambda_H:

  H-side regularization strength

- lambda_W:

  W-side regularization strength

- lambda_batch:

  Batch nullspace repulsion strength

- lambda_c:

  Classifier guide lambda for target derivation

- h_target:

  Pre-computed H-side target matrix (k x n or k x n_classes)

- w_target:

  Pre-computed W-side target matrix (k x m)

- labels:

  Integer vector of per-sample labels (0-indexed, -1 = unlabeled)

- n_classes:

  Number of classes

- precision:

  Pre-computed precision matrix (k x k) for Mahalanobis (A21)

- batch_sub:

  Pre-computed batch subspace matrix (k x k) for A22/A23

- warmup_iters:

  Iterations before enabling regularization (default 10)

- anneal_schedule:

  Annealing: 0=none, 1=cosine, 2=linear, 3=cutoff_50, 4=cutoff_25

- update_interval:

  Recompute targets every N iters for A26

- maxit:

  Maximum NMF iterations

- tol:

  Convergence tolerance

- seed:

  Random seed

- L1:

  L1 regularization (length 2: H, W)

- L2:

  L2 regularization (length 2: H, W)

- cd_maxit:

  NNLS CD iterations

- verbose:

  Print progress

- nonneg:

  Nonnegativity constraints (length 2: W, H)

- norm:

  Normalization type ("L1", "L2", "none")

- solver_mode:

  0=CD, 1=Cholesky

- loss_every:

  Compute loss every N iterations

- patience:

  Convergence patience

- w_init:

  Optional W initialization (m x k)

- guide_labels:

  Optional classifier guide labels (integer, 0-indexed)

- guide_n_classes:

  Number of classes for classifier guide

- guide_lambda:

  Classifier guide lambda

## Value

List with w (m x k), d (k), h (k x n), iter, converged, loss, tol

## Details

Alignment types:

- 1 (A20): Static per-cell target, M=I

- 2 (A21): Mahalanobis target, M=precision

- 3 (A22): Batch nullspace repulsion, M=VV^T

- 4 (A23): Combined target + nullspace

- 5 (A24): W-side target only

- 6 (A25): Dual W+H targets

- 7 (A26): Dynamic target update

- 8 (A27): Whitened centroid targets

- 9 (A28): Annealed regularization
