# Per-Epoch Guided NMF (GPU)

Runs NMF with per-epoch OAS-ZCA guided injection between H and W
updates. All computation on GPU via cuBLAS + custom CUDA kernels. The
guide (ZCA whitening + centroid shift + optional nullspace removal) is
blended into H after each H-update, before the W-update, with an
annealed blending weight lambda(t).

## Usage

``` r
gpu_perepoch_nmf(
  A,
  k,
  W_init,
  d_init,
  labels,
  batch = NULL,
  maxit = 100L,
  cd_maxit = 10L,
  lambda_c_max = 1,
  lambda_n_max = 1,
  oas_auto = TRUE,
  beta_power = 1,
  do_zca = TRUE,
  use_geomed = FALSE,
  schedule = "constant",
  schedule_p1 = 0,
  schedule_p2 = 0,
  loss_every = 5L,
  verbose = FALSE
)
```

## Arguments

- A:

  Sparse matrix (dgCMatrix, m×n)

- k:

  Factorization rank (must be ≤ 32)

- W_init:

  Initial W matrix (m×k). All configs should share the same init.

- d_init:

  Initial d vector (k). Scaling diagonal.

- labels:

  Integer vector of cell-type labels (length n, 1-based or factor)

- batch:

  Integer vector of batch labels (length n, 1-based or factor). NULL to
  skip nullspace removal.

- maxit:

  Number of NMF epochs (default 100)

- cd_maxit:

  Coordinate descent iterations per NNLS solve (default 10)

- lambda_c_max:

  Maximum centroid guide strength `[0,1]` (default 1.0)

- lambda_n_max:

  Maximum nullspace removal strength `[0,1]` (default 1.0)

- oas_auto:

  Use OAS-optimal shrinkage for ZCA (default TRUE)

- beta_power:

  ZCA whitening power (default 1.0 = standard ZCA)

- do_zca:

  Apply ZCA whitening before centroid guide (default TRUE)

- use_geomed:

  Use geometric median for centroids (default FALSE)

- schedule:

  Annealing schedule: "constant", "linear", "exponential", "cyclic",
  "logarithmic", "warmup", "cosine_decay"

- schedule_p1:

  Schedule parameter 1 (tau for exponential, period for cyclic,
  warmup_epochs for warmup)

- schedule_p2:

  Schedule parameter 2 (reserved)

- loss_every:

  Compute loss every N epochs (0 = never). Default 5.

- verbose:

  Print progress

## Value

List with W (m×k), d (k), H (n×k), loss (numeric vector)
