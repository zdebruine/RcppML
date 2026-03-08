#!/usr/bin/env Rscript
# =============================================================================
# Overfitting Onset Experiment: Subsampled CV Mask
# =============================================================================
# Scan k=2..64 on pbmc3k to find where overfitting begins (test error rises
# while train error falls). Compare full mask vs row+col subsampled masks.
#
# Overfitting onset: the rank at which test_loss/train_loss > threshold (e.g. 2.0)
# or equivalently where the test-train gap dramatically widens.
#
# Uses fp32=FALSE (double precision) to avoid NaN at higher ranks.
# =============================================================================

library(RcppML)

data(pbmc3k, package = "RcppML")
A <- pbmc3k
cat(sprintf("pbmc3k: %d x %d, nnz=%d, sparsity=%.1f%%\n",
    nrow(A), ncol(A), length(A@x),
    (1 - length(A@x)/(nrow(A)*ncol(A))) * 100))

# ---- Configuration ----
RANKS <- 2:64
SEEDS <- c(42, 123, 456, 789, 2025)
MAXIT <- 100
TEST_FRACTION <- 0.1

# Subsampling configs: (col_frac, row_frac, label)
CONFIGS <- list(
  list(col=1.0,  row=1.0,  label="full"),
  list(col=0.5,  row=1.0,  label="col50"),
  list(col=0.25, row=1.0,  label="col25"),
  list(col=1.0,  row=0.5,  label="row50"),
  list(col=1.0,  row=0.25, label="row25"),
  list(col=0.5,  row=0.5,  label="both50"),
  list(col=0.25, row=0.25, label="both25")
)

OUT_DIR <- "benchmarks/results"
if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)
OUT_FILE <- file.path(OUT_DIR, "overfitting_onset_results.csv")

# ---- Run experiment ----
results <- data.frame()
total <- length(CONFIGS) * length(SEEDS) * length(RANKS)
cat(sprintf("Total runs: %d\n", total))

for (cfg in CONFIGS) {
  cat(sprintf("\n=== Config: %s (col=%.2f, row=%.2f) ===\n", cfg$label, cfg$col, cfg$row))
  
  for (seed in SEEDS) {
    cat(sprintf("  seed=%d: ", seed))
    
    for (k in RANKS) {
      tryCatch({
        t <- system.time(
          res <- nmf(A, k=k, seed=seed, maxit=MAXIT, tol=1e-10,
                     sparse=TRUE, fp32=FALSE,
                     test_fraction=TEST_FRACTION,
                     col_subsample_frac=cfg$col,
                     row_subsample_frac=cfg$row,
                     verbose=FALSE)
        )
        
        results <- rbind(results, data.frame(
          config = cfg$label,
          col_frac = cfg$col,
          row_frac = cfg$row,
          seed = seed,
          k = k,
          train_loss = res@misc$train_loss,
          test_loss = res@misc$test_loss,
          best_iter = res@misc$best_iter,
          total_iter = res@misc$iter,
          time_sec = as.numeric(t[3]),
          ratio = res@misc$test_loss / res@misc$train_loss,
          stringsAsFactors = FALSE
        ))
      }, error = function(e) {
        cat(sprintf("ERR(k=%d) ", k))
      })
    }
    cat(sprintf("done (%d ranks)\n", length(RANKS)))
  }
}

write.csv(results, OUT_FILE, row.names = FALSE)
cat(sprintf("\nSaved %d rows to %s\n", nrow(results), OUT_FILE))

# ---- Analysis: Overfitting Onset Detection ----
cat("\n========== OVERFITTING ONSET ANALYSIS ==========\n")
cat("Finding rank where test/train ratio first exceeds thresholds\n\n")

thresholds <- c(1.5, 1.75, 2.0, 2.5)

for (cfg_label in unique(results$config)) {
  cat(sprintf("--- %s ---\n", cfg_label))
  sub <- results[results$config == cfg_label, ]
  
  cat(sprintf("  %-8s", "seed"))
  for (th in thresholds) cat(sprintf("  ratio>%.2f", th))
  cat("  min_test_k\n")
  
  for (seed in SEEDS) {
    ss <- sub[sub$seed == seed, ]
    ss <- ss[order(ss$k), ]
    # Remove NaN rows
    ss <- ss[!is.nan(ss$ratio), ]
    
    cat(sprintf("  %-8d", seed))
    for (th in thresholds) {
      onset <- ss$k[which(ss$ratio > th)[1]]
      if (is.na(onset)) onset <- ">64"
      cat(sprintf("  k=%-6s", as.character(onset)))
    }
    # Also show the k with minimum test_loss
    min_test_k <- ss$k[which.min(ss$test_loss)]
    cat(sprintf("  k=%d", min_test_k))
    cat("\n")
  }
  cat("\n")
}

# ---- Summary table: median overfitting onset per config ----
cat("\n========== MEDIAN OVERFITTING ONSET (ratio > 2.0) ==========\n")
cat(sprintf("%-12s  %-12s  %-12s  %-12s\n", "config", "median_onset", "mean_onset", "sd_onset"))

for (cfg_label in unique(results$config)) {
  sub <- results[results$config == cfg_label, ]
  onsets <- numeric()
  
  for (seed in SEEDS) {
    ss <- sub[sub$seed == seed, ]
    ss <- ss[order(ss$k), ]
    ss <- ss[!is.nan(ss$ratio), ]
    onset <- ss$k[which(ss$ratio > 2.0)[1]]
    if (!is.na(onset)) onsets <- c(onsets, onset)
  }
  
  if (length(onsets) > 0) {
    cat(sprintf("%-12s  k=%-10.0f  k=%-10.1f  %-12.1f\n",
        cfg_label, median(onsets), mean(onsets), sd(onsets)))
  } else {
    cat(sprintf("%-12s  never\n", cfg_label))
  }
}

# ---- Mean test/train curves ----
cat("\n========== MEAN LOSS CURVES (averaged over 5 seeds) ==========\n")
cat("Showing every 4th rank for readability\n\n")

show_ranks <- seq(2, 64, by=4)

for (cfg_label in c("full", "col25", "row25", "both25")) {
  sub <- results[results$config == cfg_label, ]
  if (nrow(sub) == 0) next
  
  cat(sprintf("--- %s ---\n", cfg_label))
  cat(sprintf("  %-4s  %-10s  %-10s  %-8s  %-8s\n", "k", "train", "test", "ratio", "time"))
  
  for (k in show_ranks) {
    ks <- sub[sub$k == k, ]
    ks <- ks[!is.nan(ks$train_loss), ]
    if (nrow(ks) == 0) next
    
    cat(sprintf("  %-4d  %-10.3f  %-10.3f  %-8.2f  %-8.3f\n",
        k, mean(ks$train_loss), mean(ks$test_loss),
        mean(ks$test_loss) / mean(ks$train_loss),
        mean(ks$time_sec)))
  }
  cat("\n")
}

cat("Done.\n")
