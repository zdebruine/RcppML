#!/usr/bin/env Rscript
# generate.R — Generate synthetic benchmark datasets
#
# Produces:
#   sparse_5k_2k.rds — 5000 x 2000 sparse matrix (~10% density)
#   dense_1k_500.rds — 1000 x 500 dense matrix
#
# Both generated from known W*H + noise for verifiable factorizations.

library(Matrix)

dataset_dir <- normalizePath(dirname(sys.frame(1)$ofile %||%
                              "benchmarks/harness/datasets/generate.R"),
                            mustWork = FALSE)
if (!dir.exists(dataset_dir)) {
  dataset_dir <- file.path(getwd(), "benchmarks", "harness", "datasets")
}
dir.create(dataset_dir, recursive = TRUE, showWarnings = FALSE)

set.seed(42)

# ============================================================================
# Sparse dataset: 5000 x 2000, ~10% nonzeros
# ============================================================================
cat("Generating sparse 5000x2000 dataset...\n")
m_sp <- 5000L
n_sp <- 2000L
k_true <- 10L

# Generate ground-truth factors (non-negative)
W_true <- matrix(abs(rnorm(m_sp * k_true)), m_sp, k_true)
H_true <- matrix(abs(rnorm(k_true * n_sp)), k_true, n_sp)

# Low-rank approximation + noise
A_dense <- W_true %*% H_true
A_dense <- A_dense + matrix(abs(rnorm(m_sp * n_sp, sd = 0.1)), m_sp, n_sp)

# Sparsify: keep ~10% of entries (set rest to zero)
mask <- matrix(runif(m_sp * n_sp) < 0.1, m_sp, n_sp)
A_dense[!mask] <- 0

A_sparse <- as(A_dense, "dgCMatrix")
cat(sprintf("  Dimensions: %d x %d, nnz: %d (%.1f%%)\n",
            nrow(A_sparse), ncol(A_sparse), nnzero(A_sparse),
            100 * nnzero(A_sparse) / (nrow(A_sparse) * ncol(A_sparse))))

saveRDS(A_sparse, file.path(dataset_dir, "sparse_5k_2k.rds"))

# ============================================================================
# Dense dataset: 1000 x 500
# ============================================================================
cat("Generating dense 1000x500 dataset...\n")
m_dn <- 1000L
n_dn <- 500L

W_dn <- matrix(abs(rnorm(m_dn * k_true)), m_dn, k_true)
H_dn <- matrix(abs(rnorm(k_true * n_dn)), k_true, n_dn)
A_dn <- W_dn %*% H_dn + matrix(abs(rnorm(m_dn * n_dn, sd = 0.1)), m_dn, n_dn)

cat(sprintf("  Dimensions: %d x %d\n", nrow(A_dn), ncol(A_dn)))
saveRDS(A_dn, file.path(dataset_dir, "dense_1k_500.rds"))

cat("Dataset generation complete.\n")
