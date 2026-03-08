#!/usr/bin/env Rscript
# =============================================================================
# Subsampled CV Mask Experiment
# =============================================================================
# Test whether restricting the CV holdout mask to a subset of columns
# produces consistent test error (rank selection) while reducing cost.
#
# Key design decisions:
#   1. Col-only subsampling (row_subsample_frac=1.0) - preserves test set quality
#   2. sparse=TRUE (mask_zeros=TRUE) for sparse data - only nonzeros in test set
#   3. Same seed per (dataset, k, seed) ensures same W0 across fracs
#   4. Force many iterations (tol=1e-10, maxit=100) for fair timing
# =============================================================================

library(RcppML)

# ---- Configuration ----
RANKS <- c(8, 16, 32, 64)
COL_FRACS <- c(1.0, 0.75, 0.5, 0.25, 0.1)
SEEDS <- c(42, 123, 456, 789, 2025)
MAXIT <- 100
TEST_FRACTION <- 0.1
THREADS <- 1  # Single-threaded for fair timing

OUT_DIR <- "benchmarks/results"
if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)
OUT_FILE <- file.path(OUT_DIR, "subsampled_cv_results.csv")

# ---- Load datasets ----
datasets <- list()

cat("Loading datasets...\n")

data(hawaiibirds, package = "RcppML")
datasets[["hawaiibirds"]] <- list(
  data = hawaiibirds,
  sparse_mode = TRUE,
  desc = sprintf("%d x %d, nnz=%d", nrow(hawaiibirds), ncol(hawaiibirds), 
                 length(hawaiibirds@x))
)

data(movielens, package = "RcppML")
datasets[["movielens"]] <- list(
  data = movielens,
  sparse_mode = TRUE,
  desc = sprintf("%d x %d, nnz=%d", nrow(movielens), ncol(movielens),
                 length(movielens@x))
)

data(pbmc3k, package = "RcppML")
datasets[["pbmc3k"]] <- list(
  data = pbmc3k,
  sparse_mode = TRUE,
  desc = sprintf("%d x %d, nnz=%d", nrow(pbmc3k), ncol(pbmc3k),
                 length(pbmc3k@x))
)

for (nm in names(datasets)) {
  cat(sprintf("  %s: %s\n", nm, datasets[[nm]]$desc))
}

# ---- Run experiment ----
results <- data.frame(
  dataset = character(),
  m = integer(),
  n = integer(),
  nnz = integer(),
  k = integer(),
  seed = integer(),
  col_frac = numeric(),
  test_loss = numeric(),
  train_loss = numeric(),
  best_iter = integer(),
  total_iter = integer(),
  time_sec = numeric(),
  stringsAsFactors = FALSE
)

total_runs <- length(datasets) * length(RANKS) * length(SEEDS) * length(COL_FRACS)
run_idx <- 0

cat(sprintf("\nTotal configurations: %d\n", total_runs))

for (ds_name in names(datasets)) {
  ds <- datasets[[ds_name]]
  A <- ds$data
  m <- nrow(A)
  n <- ncol(A)
  nnz <- length(A@x)
  
  cat(sprintf("\n--- Dataset: %s (%s) ---\n", ds_name, ds$desc))
  
  for (k in RANKS) {
    if (k >= min(m, n)) {
      cat(sprintf("  Skipping k=%d (>= min dim %d)\n", k, min(m, n)))
      next
    }
    
    for (seed in SEEDS) {
      cat(sprintf("  k=%d, seed=%d: ", k, seed))
      
      for (frac in COL_FRACS) {
        run_idx <- run_idx + 1
        
        tryCatch({
          t <- system.time(
            res <- nmf(A, k = k, seed = seed, maxit = MAXIT, tol = 1e-10,
                       sparse = ds$sparse_mode,
                       test_fraction = TEST_FRACTION,
                       col_subsample_frac = frac,
                       row_subsample_frac = 1.0,
                       threads = THREADS,
                       verbose = FALSE)
          )
          
          row <- data.frame(
            dataset = ds_name,
            m = m, n = n, nnz = nnz,
            k = k, seed = seed, col_frac = frac,
            test_loss = res@misc$test_loss,
            train_loss = res@misc$train_loss,
            best_iter = res@misc$best_iter,
            total_iter = res@misc$iter,
            time_sec = as.numeric(t[3]),
            stringsAsFactors = FALSE
          )
          
          results <- rbind(results, row)
          cat(sprintf("f=%.2f:%.3f(%.1fs) ", frac, res@misc$test_loss, t[3]))
          
        }, error = function(e) {
          cat(sprintf("f=%.2f:ERROR(%s) ", frac, conditionMessage(e)))
        })
      }
      cat("\n")
    }
  }
}

# ---- Save results ----
write.csv(results, OUT_FILE, row.names = FALSE)
cat(sprintf("\nResults saved to %s (%d rows)\n", OUT_FILE, nrow(results)))

# ---- Quick summary ----
cat("\n========== SUMMARY ==========\n")

for (ds_name in unique(results$dataset)) {
  cat(sprintf("\n--- %s ---\n", ds_name))
  sub <- results[results$dataset == ds_name, ]
  
  for (k in unique(sub$k)) {
    cat(sprintf("  k=%d:\n", k))
    
    ksub <- sub[sub$k == k, ]
    
    agg <- aggregate(cbind(test_loss, time_sec) ~ col_frac, data = ksub,
                     FUN = function(x) c(mean = mean(x), sd = sd(x)))
    
    baseline_test <- agg$test_loss[agg$col_frac == 1.0, "mean"]
    baseline_time <- agg$time_sec[agg$col_frac == 1.0, "mean"]
    
    for (i in 1:nrow(agg)) {
      frac <- agg$col_frac[i]
      test_mean <- agg$test_loss[i, "mean"]
      test_sd <- agg$test_loss[i, "sd"]
      time_mean <- agg$time_sec[i, "mean"]
      speedup <- baseline_time / time_mean
      pct_diff <- (test_mean - baseline_test) / baseline_test * 100
      
      cat(sprintf("    frac=%.2f: test=%.3f +/- %.3f (%+.1f%%), time=%.2fs (%.2fx)\n",
                  frac, test_mean, test_sd, pct_diff, time_mean, speedup))
    }
  }
}

# ---- Rank selection consistency ----
cat("\n========== RANK SELECTION CONSISTENCY ==========\n")
cat("Optimal k (lowest test_loss) per seed and fraction:\n\n")

for (ds_name in unique(results$dataset)) {
  cat(sprintf("--- %s ---\n", ds_name))
  sub <- results[results$dataset == ds_name, ]
  
  cat(sprintf("  %-8s", "seed"))
  for (frac in COL_FRACS) cat(sprintf("  frac=%.2f", frac))
  cat("\n")
  
  for (seed in SEEDS) {
    cat(sprintf("  seed=%-4d", seed))
    for (frac in COL_FRACS) {
      sf <- sub[sub$seed == seed & sub$col_frac == frac, ]
      if (nrow(sf) > 0) {
        best_idx <- which.min(sf$test_loss)
        cat(sprintf("  k=%-5d ", sf$k[best_idx]))
      } else {
        cat(sprintf("  %-8s", "N/A"))
      }
    }
    cat("\n")
  }
  cat("\n")
}

cat("Done.\n")
