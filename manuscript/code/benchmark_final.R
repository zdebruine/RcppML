#!/usr/bin/env Rscript
# Benchmark continuation - SKIP singlet CV due to crashes

library(RcppML)
library(singlet)
library(Matrix)
library(dplyr)

cat("=== Benchmark Continuation (skipping singlet CV) ===\n")

MAXIT <- 5
TEST_FRACTION <- 1/16
SEED_BASE <- 42

# Load existing results
if (file.exists("benchmark_final.csv")) {
  results <- read.csv("benchmark_final.csv")
} else if (file.exists("benchmark_results_v2.csv")) {
  results <- read.csv("benchmark_results_v2.csv")
} else {
  results <- data.frame()
}
cat("Existing results:", nrow(results), "\n")

# All datasets
all_datasets <- c("movielens", "hawaiibirds", "aml", "pbmc3k", "olivetti", "digits_full", "golub")
RANKS <- c(2, 4, 8, 16, 32, 64)

# Helper functions
extract_factors <- function(fit) {
  if (inherits(fit, "nmf")) {
    return(list(w = as.matrix(fit@w), h = as.matrix(fit@h)))
  } else if (is.list(fit)) {
    return(list(w = as.matrix(fit$w), h = as.matrix(fit$h)))
  }
  stop("Unknown NMF result type")
}

compute_mse <- function(A, W, H) {
  W <- as.matrix(W)
  H <- as.matrix(H)
  if (inherits(A, "sparseMatrix")) {
    A_coo <- Matrix::summary(A)
    pred <- rowSums(W[A_coo$i, , drop = FALSE] * t(H[, A_coo$j, drop = FALSE]))
    return(mean((A_coo$x - pred)^2))
  }
  WH <- W %*% H
  return(mean((as.matrix(A) - WH)^2))
}

compute_cv_mse <- function(A, W, H, seed, test_fraction) {
  W <- as.matrix(W)
  H <- as.matrix(H)
  
  if (inherits(A, "sparseMatrix")) {
    A_coo <- Matrix::summary(A)
    n_vals <- nrow(A_coo)
    set.seed(seed)
    holdout <- sample(c(TRUE, FALSE), n_vals, replace = TRUE, 
                      prob = c(test_fraction, 1 - test_fraction))
    pred <- rowSums(W[A_coo$i, , drop = FALSE] * t(H[, A_coo$j, drop = FALSE]))
    actual <- A_coo$x
    train_mse <- mean((actual[!holdout] - pred[!holdout])^2)
    test_mse <- mean((actual[holdout] - pred[holdout])^2)
  } else {
    A_dense <- as.matrix(A)
    WH <- W %*% H
    n_vals <- length(A_dense)
    set.seed(seed)
    holdout <- matrix(sample(c(TRUE, FALSE), n_vals, replace = TRUE,
                             prob = c(test_fraction, 1 - test_fraction)),
                      nrow = nrow(A_dense), ncol = ncol(A_dense))
    diff <- A_dense - WH
    train_mse <- mean(diff[!holdout]^2)
    test_mse <- mean(diff[holdout]^2)
  }
  return(list(train_mse = train_mse, test_mse = test_mse))
}

# Run benchmarks
for (dataset_name in all_datasets) {
  cat(sprintf("\n=== Processing %s ===\n", dataset_name))
  
  # Load data
  data(list = dataset_name, package = "RcppML")
  A <- get(dataset_name)
  if (!inherits(A, "sparseMatrix")) {
    A <- as(A, "dgCMatrix")
  }
  
  max_rank <- min(nrow(A), ncol(A)) - 1
  cat(sprintf("Dimensions: %d x %d, max_rank: %d\n", nrow(A), ncol(A), max_rank))
  
  for (k in RANKS) {
    if (k > max_rank) {
      cat(sprintf("Skipping k=%d (too large)\n", k))
      next
    }
    
    for (rep in 1:3) {
      # Check if this config is already done (need at least 6 results for RcppML methods)
      existing_count <- sum(results$dataset == dataset_name & 
                           results$rank == k & 
                           results$replicate == rep &
                           results$method == "RcppML")
      if (existing_count >= 5) {
        next  # Skip - RcppML already done for this config
      }
      
      seed <- SEED_BASE + rep
      cat(sprintf("[%s] k=%d, rep=%d\n", dataset_name, k, rep))
      
      # 1. RcppML Standard (sparse=FALSE)
      tryCatch({
        t <- system.time(fit <- RcppML::nmf(A, k=k, tol=1e-10, maxit=MAXIT, verbose=FALSE, 
                                           seed=seed, sparse=FALSE))
        f <- extract_factors(fit)
        mse <- compute_mse(A, f$w, f$h)
        results <- rbind(results, data.frame(
          dataset=dataset_name, rank=k, replicate=rep, method="RcppML", variant="standard",
          mask_zeros=FALSE, precision="double", cv=FALSE, total_time=t["elapsed"],
          time_per_epoch=t["elapsed"]/MAXIT, train_mse=mse, test_mse=NA, stringsAsFactors=FALSE))
      }, error = function(e) cat(sprintf("  ERR RcppML std: %s\n", e$message)))
      
      # 2. RcppML Standard (sparse=TRUE)
      tryCatch({
        t <- system.time(fit <- RcppML::nmf(A, k=k, tol=1e-10, maxit=MAXIT, verbose=FALSE,
                                           seed=seed, sparse=TRUE))
        f <- extract_factors(fit)
        mse <- compute_mse(A, f$w, f$h)
        results <- rbind(results, data.frame(
          dataset=dataset_name, rank=k, replicate=rep, method="RcppML", variant="standard",
          mask_zeros=TRUE, precision="double", cv=FALSE, total_time=t["elapsed"],
          time_per_epoch=t["elapsed"]/MAXIT, train_mse=mse, test_mse=NA, stringsAsFactors=FALSE))
      }, error = function(e) cat(sprintf("  ERR RcppML sparse: %s\n", e$message)))
      
      # 3. RcppML CV (double)
      tryCatch({
        t <- system.time(fit <- RcppML::nmf(A, k=k, tol=1e-10, maxit=MAXIT, verbose=FALSE,
                                           seed=seed, sparse=FALSE, test_fraction=TEST_FRACTION,
                                           cv_seed=seed, fp32=FALSE))
        f <- extract_factors(fit)
        cv_mse <- compute_cv_mse(A, f$w, f$h, seed, TEST_FRACTION)
        results <- rbind(results, data.frame(
          dataset=dataset_name, rank=k, replicate=rep, method="RcppML", variant="cv",
          mask_zeros=FALSE, precision="double", cv=TRUE, total_time=t["elapsed"],
          time_per_epoch=t["elapsed"]/MAXIT, train_mse=cv_mse$train_mse, test_mse=cv_mse$test_mse, stringsAsFactors=FALSE))
      }, error = function(e) cat(sprintf("  ERR RcppML CV double: %s\n", e$message)))
      
      # 4. RcppML CV (sparse=TRUE, double)
      tryCatch({
        t <- system.time(fit <- RcppML::nmf(A, k=k, tol=1e-10, maxit=MAXIT, verbose=FALSE,
                                           seed=seed, sparse=TRUE, test_fraction=TEST_FRACTION,
                                           cv_seed=seed, fp32=FALSE))
        f <- extract_factors(fit)
        cv_mse <- compute_cv_mse(A, f$w, f$h, seed, TEST_FRACTION)
        results <- rbind(results, data.frame(
          dataset=dataset_name, rank=k, replicate=rep, method="RcppML", variant="cv",
          mask_zeros=TRUE, precision="double", cv=TRUE, total_time=t["elapsed"],
          time_per_epoch=t["elapsed"]/MAXIT, train_mse=cv_mse$train_mse, test_mse=cv_mse$test_mse, stringsAsFactors=FALSE))
      }, error = function(e) cat(sprintf("  ERR RcppML CV sparse: %s\n", e$message)))
      
      # 5. RcppML CV (float)
      tryCatch({
        t <- system.time(fit <- RcppML::nmf(A, k=k, tol=1e-10, maxit=MAXIT, verbose=FALSE,
                                           seed=seed, sparse=FALSE, test_fraction=TEST_FRACTION,
                                           cv_seed=seed, fp32=TRUE))
        f <- extract_factors(fit)
        cv_mse <- compute_cv_mse(A, f$w, f$h, seed, TEST_FRACTION)
        results <- rbind(results, data.frame(
          dataset=dataset_name, rank=k, replicate=rep, method="RcppML", variant="cv",
          mask_zeros=FALSE, precision="float", cv=TRUE, total_time=t["elapsed"],
          time_per_epoch=t["elapsed"]/MAXIT, train_mse=cv_mse$train_mse, test_mse=cv_mse$test_mse, stringsAsFactors=FALSE))
      }, error = function(e) cat(sprintf("  ERR RcppML CV float: %s\n", e$message)))
      
      # 6. singlet Standard (ONLY)
      tryCatch({
        t <- system.time(fit <- singlet::run_nmf(A, rank=k, tol=1e-10, maxit=MAXIT, verbose=FALSE, L1=0, L2=0))
        f <- extract_factors(fit)
        mse <- compute_mse(A, f$w, f$h)
        results <- rbind(results, data.frame(
          dataset=dataset_name, rank=k, replicate=rep, method="singlet", variant="standard",
          mask_zeros=FALSE, precision="double", cv=FALSE, total_time=t["elapsed"],
          time_per_epoch=t["elapsed"]/MAXIT, train_mse=mse, test_mse=NA, stringsAsFactors=FALSE))
      }, error = function(e) cat(sprintf("  ERR singlet std: %s\n", e$message)))
      
      # SKIP singlet CV - it crashes R for certain configurations
    }
  }
  
  # Save after each dataset
  write.csv(results, "benchmark_final.csv", row.names = FALSE)
  cat(sprintf("Saved %d results\n", nrow(results)))
}

cat("\n=== Creating Summary ===\n")

summary_stats <- results %>%
  group_by(dataset, rank, method, variant, mask_zeros, precision, cv) %>%
  summarise(
    n = n(),
    mean_time = mean(total_time, na.rm = TRUE),
    sd_time = sd(total_time, na.rm = TRUE),
    mean_time_per_epoch = mean(time_per_epoch, na.rm = TRUE),
    mean_train_mse = mean(train_mse, na.rm = TRUE),
    mean_test_mse = mean(test_mse, na.rm = TRUE),
    .groups = "drop"
  )

write.csv(summary_stats, "benchmark_final_summary.csv", row.names = FALSE)
cat("Saved summary\n")

cat("\n=== Done ===\n")
cat(sprintf("Total: %d results\n", nrow(results)))
