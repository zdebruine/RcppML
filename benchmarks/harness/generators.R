#!/usr/bin/env Rscript
# generators.R — Standard test matrix generators for benchmarks
#
# Source this file to get gen_sparse(), gen_dense(), gen_nmf_truth(), gen_spz_file().
#
# Usage:
#   source("benchmarks/harness/generators.R")
#   A <- gen_sparse(10000, 5000, density = 0.1)
#   truth <- gen_nmf_truth(1000, 500, k = 10)

suppressPackageStartupMessages(library(Matrix))

# ============================================================================
# Standard sizes (used by suite scripts)
# ============================================================================
BENCH_SIZES <- list(
  small        = list(m = 1000L,  n = 500L,   density = 0.10),
  medium       = list(m = 10000L, n = 5000L,  density = 0.10),
  large        = list(m = 50000L, n = 20000L, density = 0.05),
  dense_small  = list(m = 500L,   n = 200L,   density = 1.0),
  dense_medium = list(m = 2000L,  n = 1000L,  density = 1.0)
)

# ============================================================================
# Generators
# ============================================================================

#' Generate a sparse non-negative matrix (dgCMatrix)
#'
#' @param m     Number of rows
#' @param n     Number of columns
#' @param density Fraction of non-zero entries (0,1]
#' @param seed  RNG seed
#' @return dgCMatrix
gen_sparse <- function(m, n, density = 0.1, seed = 42L) {
  set.seed(seed)
  nnz_target <- as.integer(m * n * density)

  # Sample random positions
  idx <- sample.int(as.double(m) * n, nnz_target, replace = FALSE)
  rows <- as.integer((idx - 1L) %% m)
  cols <- as.integer((idx - 1L) %/% m)
  vals <- abs(rnorm(nnz_target))

  A <- sparseMatrix(i = rows + 1L, j = cols + 1L, x = vals,
                    dims = c(m, n), repr = "C")
  A
}

#' Generate a dense non-negative matrix
#'
#' @param m    Number of rows
#' @param n    Number of columns
#' @param seed RNG seed
#' @return matrix
gen_dense <- function(m, n, seed = 42L) {
  set.seed(seed)
  matrix(abs(rnorm(m * n)), m, n)
}

#' Generate a matrix with known NMF ground truth
#'
#' @param m     Number of rows
#' @param n     Number of columns
#' @param k     True rank
#' @param noise Noise level (SD of additive noise)
#' @param seed  RNG seed
#' @return list(A, W_true, H_true)
gen_nmf_truth <- function(m, n, k, noise = 0.05, seed = 42L) {
  set.seed(seed)
  W_true <- matrix(abs(rnorm(m * k)), m, k)
  H_true <- matrix(abs(rnorm(k * n)), k, n)
  A <- W_true %*% H_true
  A <- A + matrix(abs(rnorm(m * n, sd = noise)), m, n)
  list(A = A, W_true = W_true, H_true = H_true)
}

#' Generate a sparse matrix and write it as an SPZ file
#'
#' @param m       Number of rows
#' @param n       Number of columns
#' @param density Fraction of non-zero entries
#' @param path    File path for the SPZ file (if NULL, uses tempfile)
#' @param seed    RNG seed
#' @return Path to the written SPZ file
gen_spz_file <- function(m, n, density = 0.1, path = NULL, seed = 42L) {
  if (is.null(path)) path <- tempfile(fileext = ".spz")

  A <- gen_sparse(m, n, density = density, seed = seed)

  if (!requireNamespace("RcppML", quietly = TRUE)) {
    stop("RcppML must be loaded to write SPZ files")
  }
  sp_write(A, path, include_transpose = TRUE)
  path
}
