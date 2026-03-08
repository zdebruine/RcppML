# Test Helper Utilities
# Shared functions for comprehensive NMF testing

# Load required libraries
library(RcppML)
library(Matrix)

# Force CPU for all non-GPU tests. GPU test files override this locally
# with options(RcppML.gpu = TRUE) or resource = "gpu" as needed.
# This prevents auto-dispatch to a stale GPU .so during testing.
options(RcppML.gpu = FALSE)

#' Align factors to resolve permutation and scaling ambiguity
#' 
#' Uses bipartite matching to find optimal permutation
#' 
#' @param W,H Estimated factors
#' @param W_true,H_true Ground truth factors
#' @return List with aligned W and H, and correlation metrics
align_nmf_factors <- function(W, H, W_true, H_true) {
  k <- ncol(W)
  
  # Compute correlation matrix between estimated and true factors
  cor_matrix <- matrix(0, k, k)
  for (i in 1:k) {
    for (j in 1:k) {
      cor_matrix[i, j] <- 1 - abs(cor(W[, i], W_true[, j]))
    }
  }
  
  # Find optimal permutation using bipartite matching (minimize cost)
  matching <- bipartiteMatch(cor_matrix)
  
  # bipartiteMatch returns 0-based indices in $assignment
  # assignment[i] = j means row i matches to column j (0-based)
  perm <- matching$assignment + 1  # Convert to 1-based indexing
  
  # Permute estimated factors
  W_aligned <- W[, perm, drop = FALSE]
  H_aligned <- H[perm, , drop = FALSE]
  
  # Scale to match signs (correlation can be negative)
  for (i in 1:k) {
    if (cor(W_aligned[, i], W_true[, i]) < 0) {
      W_aligned[, i] <- -W_aligned[, i]
      H_aligned[i, ] <- -H_aligned[i, ]
    }
  }
  
  # Compute final correlations
  W_cor <- sapply(1:k, function(i) cor(W_aligned[, i], W_true[, i]))
  H_cor <- sapply(1:k, function(i) cor(H_aligned[i, ], H_true[i, ]))
  
  list(
    W = W_aligned,
    H = H_aligned,
    W_cor = W_cor,
    H_cor = H_cor,
    mean_W_cor = mean(W_cor),
    mean_H_cor = mean(H_cor)
  )
}

#' Compute total loss with all regularization components
#' 
#' @param A Data matrix
#' @param W,H Factor matrices
#' @param L1 L1 penalty coefficients c(L1_W, L1_H)
#' @param L2 L2 penalty coefficients c(L2_W, L2_H)
#' @param loss_type One of "mse", "mae", "huber", "kl", "gp"
#' @param huber_delta Delta parameter for Huber loss
#' @param mask Optional mask matrix or "zeros"
#' @return Named list of loss components
compute_nmf_loss <- function(A, W, H, L1 = c(0, 0), L2 = c(0, 0),
                             loss_type = "mse", huber_delta = 1.0,
                             mask = NULL) {
  # "gp" is the public name; "kl" is the internal formula
  if (loss_type == "gp") loss_type <- "kl"
  
  # Reconstruction
  A_recon <- W %*% H
  residuals <- A - A_recon
  
  # Determine mask and count for proper normalization
  n_entries <- length(residuals)
  if (!is.null(mask)) {
    if (is.character(mask) && mask == "zeros") {
      # Zero-masked: only non-zero entries contribute
      # Use actual non-zero locations for proper MSE calculation
      mask_matrix <- (A != 0)
      n_entries <- sum(mask_matrix)
      # Zero out masked entries
      residuals[!mask_matrix] <- 0
    } else {
      # Explicit mask matrix
      n_entries <- sum(mask != 0)
      residuals <- residuals * mask
    }
  }
  
  # Compute data loss with proper normalization
  data_loss <- switch(loss_type,
    mse = sum(residuals^2) / n_entries,
    mae = sum(abs(residuals)) / n_entries,
    huber = {
      sum(sapply(as.vector(residuals), function(r) {
        if (abs(r) <= huber_delta) {
          0.5 * r^2
        } else {
          huber_delta * (abs(r) - 0.5 * huber_delta)
        }
      })) / n_entries
    },
    kl = {
      # KL divergence: sum(A * log(A / A_recon) - A + A_recon)
      A_recon_safe <- pmax(as.matrix(A_recon), 1e-10)
      A_safe <- pmax(as.matrix(A), 1e-10)
      sum(A_safe * log(A_safe / A_recon_safe) - A_safe + A_recon_safe) / n_entries
    },
    stop("Unknown loss type: ", loss_type)
  )
  
  # L1 penalties
  l1_W <- L1[1] * sum(abs(W))
  l1_H <- L1[2] * sum(abs(H))
  
  # L2 penalties
  l2_W <- (L2[1] / 2) * sum(W^2)
  l2_H <- (L2[2] / 2) * sum(H^2)
  
  # Total loss
  total <- data_loss + l1_W + l1_H + l2_W + l2_H
  
  list(
    total = total,
    data_loss = data_loss,
    l1_W = l1_W,
    l1_H = l1_H,
    l2_W = l2_W,
    l2_H = l2_H
  )
}

#' Extract loss trajectory from NMF model
#' 
#' Recompute loss at each iteration if model doesn't store trajectory
#' 
#' @param model NMF model object
#' @param A Data matrix
#' @param ... Additional loss computation parameters
#' @return Vector of losses at each iteration
get_loss_trajectory <- function(model, A, ...) {
  # If model has loss trajectory, use it
  if (!is.null(model$loss_trajectory)) {
    return(model$loss_trajectory)
  }
  
  # Otherwise, just return final loss
  if (!is.null(model$loss)) {
    return(model$loss)
  }
  
  # Compute it manually
  loss <- compute_nmf_loss(A, model$w, model$h, ...)
  return(loss$total)
}

#' Check if matrix is approximately non-negative
#' 
#' @param X Matrix to check
#' @param tol Tolerance for negativity (default: -1e-10)
#' @return TRUE if all elements >= tol
is_nonnegative <- function(X, tol = -1e-10) {
  all(X >= tol)
}

#' Compute sparsity (fraction of zeros) in matrix
#' 
#' @param X Matrix
#' @param threshold Values below this are considered zero
#' @return Fraction of elements that are zero
compute_sparsity <- function(X, threshold = 1e-10) {
  sum(abs(X) < threshold) / length(X)
}

#' Generate test mask for CV validation
#' 
#' @param A Data matrix
#' @param holdout_fraction Fraction to hold out
#' @param seed Random seed
#' @return Logical matrix (TRUE = test entry)
generate_test_mask <- function(A, holdout_fraction = 0.1, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  
  n <- nrow(A)
  p <- ncol(A)
  
  mask <- matrix(FALSE, n, p)
  n_test <- round(n * p * holdout_fraction)
  test_idx <- sample(n * p, n_test)
  mask[test_idx] <- TRUE
  
  mask
}

#' Compute reconstruction error
#' 
#' @param A True data
#' @param A_recon Reconstructed data
#' @param relative If TRUE, return relative error
#' @return Reconstruction error
reconstruction_error <- function(A, A_recon, relative = TRUE) {
  err <- norm(as.matrix(A - A_recon), "F")
  if (relative) {
    err <- err / norm(as.matrix(A), "F")
  }
  err
}

# ---- GPU test helpers ----

#' Extract W matrix from nmf result (S4 or list)
get_W <- function(model) {
  if (is(model, "nmf")) model@w else model$w
}

#' Extract H matrix from nmf result (S4 or list)
get_H <- function(model) {
  if (is(model, "nmf")) model@h else model$h
}

#' Extract d vector from nmf result (S4 or list)
get_d <- function(model) {
  if (is(model, "nmf")) model@d else model$d
}

#' Compute MSE at non-zero positions: ||A - W*diag(d)*H||^2 / nnz
compute_mse <- function(A, model) {
  W <- get_W(model); d <- get_d(model); H <- get_H(model)
  approx <- W %*% diag(d) %*% H
  idx <- which(A != 0, arr.ind = TRUE)
  sum((A[idx] - approx[idx])^2) / nrow(idx)
}

#' Compute MSE from raw factors: ||A - W*diag(d)*H||^2 / n_elements
#' Used by GPU test files that pass W, d, H, A individually.
mse <- function(W, d, H, A) {
  approx <- W %*% diag(d, nrow = length(d)) %*% H
  mean((as.matrix(A) - approx)^2)
}

#' Skip test if GPU is not available
skip_if_no_gpu <- function() {
  skip_if_not(
    exists("gpu_available", where = asNamespace("RcppML")) &&
      RcppML:::gpu_available(),
    "GPU not available"
  )
}

skip_if_no_sp_gpu <- function() {
  skip_if_no_gpu()
  has_sym <- tryCatch({
    getNativeSymbolInfo("rcppml_sp_read_gpu", RcppML:::.gpu_env$dll)
    TRUE
  }, error = function(e) FALSE)
  skip_if_not(has_sym, "sp_read_gpu not compiled into GPU .so")
}
