#!/usr/bin/env Rscript
# Performance Baselines & External Comparison (P4.6, P4.7)
# 
# Establishes performance baselines for RcppML NMF and compares against
# external packages (NMF, irlba).
# 
# Output: benchmarks/results/performance_baselines.rds

library(RcppML)
library(Matrix)

cat("==========================================================\n")
cat("  RcppML Performance Baselines (P4.6 + P4.7)\n")
cat("==========================================================\n")
cat("Threads:", getOption("RcppML.threads", 0), "\n")
cat("Date:", format(Sys.time()), "\n\n")

results <- list()

# ============================================================================
# 1. Standard NMF baselines across matrix sizes and ranks
# ============================================================================
cat("--- Section 1: RcppML NMF Baselines ---\n")

sizes <- list(
  small   = list(m = 500, n = 300, density = 0.3),
  medium  = list(m = 2000, n = 1000, density = 0.1),
  large   = list(m = 5000, n = 3000, density = 0.05),
  xlarge  = list(m = 10000, n = 5000, density = 0.02)
)

ranks <- c(2, 5, 10, 20, 50)

baselines <- data.frame()

for (sz_name in names(sizes)) {
  sz <- sizes[[sz_name]]
  set.seed(42)
  A <- rsparsematrix(sz$m, sz$n, sz$density)
  A@x <- abs(A@x)
  
  for (k in ranks) {
    if (k >= min(sz$m, sz$n)) next
    
    times <- numeric(3)
    for (rep in 1:3) {
      t0 <- proc.time()["elapsed"]
      m <- nmf(A, k = k, maxit = 50, tol = 1e-10, seed = rep, verbose = FALSE)
      times[rep] <- proc.time()["elapsed"] - t0
    }
    
    row <- data.frame(
      size = sz_name, m = sz$m, n = sz$n, nnz = length(A@x),
      k = k, median_time_s = median(times), mean_time_s = mean(times),
      sd_time_s = sd(times), loss = m@misc$loss, iter = m@misc$iter,
      stringsAsFactors = FALSE
    )
    baselines <- rbind(baselines, row)
    cat(sprintf("  %s k=%d: %.3fs (median), loss=%.6f\n", 
                sz_name, k, median(times), m@misc$loss))
  }
}
results$baselines <- baselines

# ============================================================================
# 2. Loss function comparison
# ============================================================================
cat("\n--- Section 2: Loss Function Comparison ---\n")

set.seed(42)
A_loss <- rsparsematrix(2000, 1000, 0.1)
A_loss@x <- abs(A_loss@x)

loss_results <- data.frame()
for (loss in c("mse", "mae", "huber")) {
  times <- numeric(3)
  for (rep in 1:3) {
    t0 <- proc.time()["elapsed"]
    m <- nmf(A_loss, k = 10, loss = loss, maxit = 50, tol = 1e-10, 
             seed = rep, verbose = FALSE)
    times[rep] <- proc.time()["elapsed"] - t0
  }
  row <- data.frame(
    loss = loss, median_time_s = median(times),
    final_loss = m@misc$loss, iter = m@misc$iter,
    stringsAsFactors = FALSE
  )
  loss_results <- rbind(loss_results, row)
  cat(sprintf("  %s: %.3fs, loss=%.6f\n", loss, median(times), m@misc$loss))
}
results$loss_comparison <- loss_results

# ============================================================================
# 3. Cross-validation overhead
# ============================================================================
cat("\n--- Section 3: CV Overhead ---\n")

set.seed(42)
A_cv <- rsparsematrix(2000, 1000, 0.1)
A_cv@x <- abs(A_cv@x)

cv_results <- data.frame()
for (mode in c("standard", "cv")) {
  times <- numeric(3)
  for (rep in 1:3) {
    t0 <- proc.time()["elapsed"]
    if (mode == "standard") {
      m <- nmf(A_cv, k = 10, maxit = 50, tol = 1e-10, seed = rep, verbose = FALSE)
    } else {
      m <- nmf(A_cv, k = 10, maxit = 50, tol = 1e-10, seed = rep,
               test_fraction = 0.1, cv_seed = 42, verbose = FALSE)
    }
    times[rep] <- proc.time()["elapsed"] - t0
  }
  row <- data.frame(
    mode = mode, median_time_s = median(times),
    stringsAsFactors = FALSE
  )
  cv_results <- rbind(cv_results, row)
  cat(sprintf("  %s: %.3fs\n", mode, median(times)))
}
results$cv_overhead <- cv_results

# ============================================================================
# 4. Thread scaling
# ============================================================================
cat("\n--- Section 4: Thread Scaling ---\n")

thread_counts <- c(1, 2, 4, 8)
max_threads <- as.integer(Sys.getenv("OMP_NUM_THREADS", "8"))
thread_counts <- thread_counts[thread_counts <= max_threads]

set.seed(42)
A_thr <- rsparsematrix(5000, 3000, 0.05)
A_thr@x <- abs(A_thr@x)

thread_results <- data.frame()
for (nt in thread_counts) {
  options(RcppML.threads = nt)
  times <- numeric(3)
  for (rep in 1:3) {
    t0 <- proc.time()["elapsed"]
    m <- nmf(A_thr, k = 20, maxit = 50, tol = 1e-10, seed = rep, verbose = FALSE)
    times[rep] <- proc.time()["elapsed"] - t0
  }
  row <- data.frame(
    threads = nt, median_time_s = median(times),
    speedup = NA_real_,
    stringsAsFactors = FALSE
  )
  thread_results <- rbind(thread_results, row)
  cat(sprintf("  %d threads: %.3fs\n", nt, median(times)))
}
if (nrow(thread_results) > 0) {
  t1 <- thread_results$median_time_s[1]
  thread_results$speedup <- t1 / thread_results$median_time_s
}
results$thread_scaling <- thread_results
options(RcppML.threads = 0)  # Reset

# ============================================================================
# 5. External package comparison (P4.7)
# ============================================================================
cat("\n--- Section 5: External Package Comparison ---\n")

set.seed(42)
A_ext <- rsparsematrix(2000, 1000, 0.1)
A_ext@x <- abs(A_ext@x)
A_ext_dense <- as.matrix(A_ext)

ext_results <- data.frame()

# RcppML (our package)
for (k in c(5, 10, 20)) {
  times <- numeric(3)
  for (rep in 1:3) {
    t0 <- proc.time()["elapsed"]
    m <- nmf(A_ext, k = k, maxit = 100, tol = 1e-6, seed = rep, verbose = FALSE)
    times[rep] <- proc.time()["elapsed"] - t0
  }
  row <- data.frame(
    package = "RcppML", k = k, 
    median_time_s = median(times),
    loss = m@misc$loss,
    stringsAsFactors = FALSE
  )
  ext_results <- rbind(ext_results, row)
  cat(sprintf("  RcppML k=%d: %.3fs, loss=%.6f\n", k, median(times), m@misc$loss))
}

# NMF package (if available)
if (requireNamespace("NMF", quietly = TRUE)) {
  cat("  NMF package found — running comparison\n")
  for (k in c(5, 10, 20)) {
    times <- numeric(3)
    for (rep in 1:3) {
      t0 <- proc.time()["elapsed"]
      m_nmf <- NMF::nmf(A_ext_dense, rank = k, method = "brunet", 
                          seed = rep, .options = "v-p")
      times[rep] <- proc.time()["elapsed"] - t0
    }
    # Compute MSE
    recon <- NMF::basis(m_nmf) %*% NMF::coef(m_nmf)
    mse_val <- mean((A_ext_dense - recon)^2)
    row <- data.frame(
      package = "NMF", k = k,
      median_time_s = median(times),
      loss = mse_val,
      stringsAsFactors = FALSE
    )
    ext_results <- rbind(ext_results, row)
    cat(sprintf("  NMF k=%d: %.3fs, MSE=%.6f\n", k, median(times), mse_val))
  }
} else {
  cat("  NMF package not installed — skipping\n")
}

results$external_comparison <- ext_results

# ============================================================================
# 6. Warm-start benefit measurement
# ============================================================================
cat("\n--- Section 6: Warm-Start Benefit ---\n")

set.seed(42)
A_ws <- rsparsematrix(3000, 2000, 0.05)
A_ws@x <- abs(A_ws@x)

# Current implementation uses warm-start (iter > 0)
ws_times <- numeric(3)
for (rep in 1:3) {
  t0 <- proc.time()["elapsed"]
  m <- nmf(A_ws, k = 20, maxit = 100, tol = 1e-10, seed = rep, verbose = FALSE)
  ws_times[rep] <- proc.time()["elapsed"] - t0
}
cat(sprintf("  Warm-start (100 iter): %.3fs (median)\n", median(ws_times)))
cat(sprintf("    Final loss: %.8f, Iterations: %d\n", m@misc$loss, m@misc$iter))

results$warm_start <- data.frame(
  mode = "warm_start",
  median_time_s = median(ws_times),
  final_loss = m@misc$loss,
  iter = m@misc$iter
)

# ============================================================================
# Save results
# ============================================================================
outfile <- "benchmarks/results/performance_baselines.rds"
saveRDS(results, outfile)
cat(sprintf("\nResults saved to %s\n", outfile))

# Print summary table
cat("\n=== BASELINE SUMMARY ===\n")
cat("\nNMF Baselines:\n")
print(baselines[, c("size", "k", "median_time_s", "loss")])
cat("\nLoss Functions:\n") 
print(loss_results)
cat("\nCV Overhead:\n")
print(cv_results)
cat("\nThread Scaling:\n")
print(thread_results)
cat("\nExternal Comparison:\n")
print(ext_results)

cat("\n=== REGRESSION THRESHOLDS ===\n")
cat("These baselines can be used for future regression testing.\n")
cat("A regression is flagged if median time exceeds 2x the baseline.\n\n")

for (i in seq_len(nrow(baselines))) {
  cat(sprintf("  %s k=%d: threshold = %.3fs\n", 
              baselines$size[i], baselines$k[i], 
              baselines$median_time_s[i] * 2))
}

cat("\nDone.\n")
