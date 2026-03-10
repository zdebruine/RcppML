# Dense-matrix GPU SVD/PCA

GPU SVD/PCA for a base R dense matrix (is.matrix(A) == TRUE). For
deflation: uses cuBLAS GEMV instead of cuSPARSE SpMV (significantly
faster for dense input). For other algorithms: converts to CSC
internally.

## Usage

``` r
.gpu_svd_pca_dense(
  A,
  k_max,
  tol,
  max_iter,
  center,
  verbose,
  seed,
  threads,
  L1_u,
  L1_v,
  L2_u,
  L2_v,
  nonneg_u,
  nonneg_v,
  upper_bound_u,
  upper_bound_v,
  L21_u = 0,
  L21_v = 0,
  angular_u = 0,
  angular_v = 0,
  test_fraction,
  cv_seed,
  patience,
  mask_zeros,
  method = "deflation",
  graph_U = NULL,
  graph_V = NULL,
  graph_lambda = c(0, 0),
  obs_mask = NULL,
  robust_delta = 0,
  irls_max_iter = 5L,
  irls_tol = 1e-04
)
```
