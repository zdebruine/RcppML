#!/usr/bin/env Rscript
# CPU Scaling Benchmark for bigmem nodes
# Called from: benchmarks/slurm/bigmem_scaling.sbatch

library(RcppML)
library(Matrix)

cat("\n=== CPU Scaling Benchmark ===\n")
OUTDIR <- "benchmarks/results"
timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
ncpus <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "8"))

# -------------------------------------------------------------------
# Helper: benchmark a single NMF call
# -------------------------------------------------------------------
bench_nmf <- function(A, k, threads, maxit = 50, loss = "mse",
                      precision = "double", cv = FALSE, reps = 3) {
  times <- numeric(reps)
  final_mse <- NA

  for (r in seq_len(reps)) {
    if (r == 1) {
      # Warm-up
      suppressWarnings(nmf(A, k = min(k, 2), seed = 1, maxit = 5,
                           tol = 1e-10, threads = threads, resource = "cpu"))
    }

    t0 <- proc.time()[3]
    if (cv) {
      result <- nmf(A, k = k, seed = r, maxit = maxit, tol = 1e-10,
                    threads = threads, resource = "cpu",
                    test_fraction = 0.1)
    } else {
      result <- nmf(A, k = k, seed = r, maxit = maxit, tol = 1e-10,
                    threads = threads, resource = "cpu",
                    loss = loss, precision = precision)
    }
    times[r] <- proc.time()[3] - t0

    if (r == reps && !cv) {
      final_mse <- mse(result@w, result@d, result@h, A)
    }
  }

  data.frame(
    median_time = median(times),
    min_time = min(times),
    max_time = max(times),
    final_mse = final_mse
  )
}

all_results <- data.frame()

# -------------------------------------------------------------------
# Benchmark 1: Matrix size scaling
# -------------------------------------------------------------------
cat("\n--- Benchmark 1: Matrix size scaling ---\n")
sizes <- list(
  c(500,   1000,  0.05),
  c(1000,  2000,  0.05),
  c(2000,  5000,  0.02),
  c(5000,  10000, 0.01),
  c(10000, 20000, 0.005),
  c(20000, 50000, 0.002)
)

for (sz in sizes) {
  m <- sz[1]; n <- sz[2]; density <- sz[3]; k <- 10
  nnz <- as.integer(m * n * density)

  cat(sprintf("  m=%d n=%d nnz=%d k=%d ... ", m, n, nnz, k))
  set.seed(42)
  A <- rsparsematrix(m, n, nnz = nnz)
  A@x <- abs(A@x)
  A <- as(A, "dgCMatrix")

  res <- bench_nmf(A, k = k, threads = ncpus, reps = 3)
  cat(sprintf("%.2fs (MSE=%.4e)\n", res$median_time, res$final_mse))

  all_results <- rbind(all_results, data.frame(
    benchmark = "size_scaling", m = m, n = n, nnz = nnz, k = k,
    threads = ncpus, loss = "mse", precision = "double", cv = FALSE,
    res
  ))
  rm(A); gc()
}

# -------------------------------------------------------------------
# Benchmark 2: Rank scaling
# -------------------------------------------------------------------
cat("\n--- Benchmark 2: Rank scaling ---\n")
set.seed(42)
m <- 2000; n <- 5000; density <- 0.02
A <- rsparsematrix(m, n, nnz = as.integer(m * n * density))
A@x <- abs(A@x)
A <- as(A, "dgCMatrix")

for (k in c(2, 4, 8, 16, 32, 64)) {
  cat(sprintf("  k=%d ... ", k))
  res <- bench_nmf(A, k = k, threads = ncpus, reps = 3)
  cat(sprintf("%.2fs (MSE=%.4e)\n", res$median_time, res$final_mse))

  all_results <- rbind(all_results, data.frame(
    benchmark = "rank_scaling", m = m, n = n, nnz = nnzero(A), k = k,
    threads = ncpus, loss = "mse", precision = "double", cv = FALSE,
    res
  ))
}

# -------------------------------------------------------------------
# Benchmark 3: Thread scaling
# -------------------------------------------------------------------
cat("\n--- Benchmark 3: Thread scaling ---\n")
base_time <- NULL
for (thr in c(1, 2, 4, 8, 16, 24, 32, 40)) {
  if (thr > ncpus) next
  cat(sprintf("  threads=%d ... ", thr))
  res <- bench_nmf(A, k = 10, threads = thr, reps = 3)
  if (is.null(base_time)) base_time <- res$median_time
  speedup <- base_time / res$median_time
  cat(sprintf("%.2fs (speedup=%.1fx vs 1 thread)\n", res$median_time, speedup))

  all_results <- rbind(all_results, data.frame(
    benchmark = "thread_scaling", m = m, n = n, nnz = nnzero(A), k = 10,
    threads = thr, loss = "mse", precision = "double", cv = FALSE,
    res
  ))
}

# -------------------------------------------------------------------
# Benchmark 4: Loss function comparison
# -------------------------------------------------------------------
cat("\n--- Benchmark 4: Loss function comparison ---\n")
set.seed(42)
A_loss <- rsparsematrix(1000, 2000, nnz = 20000)
A_loss@x <- abs(A_loss@x)
A_loss <- as(A_loss, "dgCMatrix")

for (loss in c("mse", "mae", "huber")) {
  cat(sprintf("  loss=%s ... ", loss))
  res <- bench_nmf(A_loss, k = 8, threads = ncpus, loss = loss, reps = 3)
  cat(sprintf("%.2fs\n", res$median_time))

  all_results <- rbind(all_results, data.frame(
    benchmark = "loss_comparison", m = 1000, n = 2000, nnz = 20000, k = 8,
    threads = ncpus, loss = loss, precision = "double", cv = FALSE,
    res
  ))
}

# -------------------------------------------------------------------
# Benchmark 5: CV NMF overhead
# -------------------------------------------------------------------
cat("\n--- Benchmark 5: CV NMF overhead ---\n")
set.seed(42)
A_cv <- rsparsematrix(500, 1000, nnz = 25000)
A_cv@x <- abs(A_cv@x)
A_cv <- as(A_cv, "dgCMatrix")

for (k in c(4, 8, 16)) {
  cat(sprintf("  CV k=%d ... ", k))
  res_std <- bench_nmf(A_cv, k = k, threads = ncpus, reps = 3)
  res_cv  <- bench_nmf(A_cv, k = k, threads = ncpus, cv = TRUE, reps = 3)
  overhead <- res_cv$median_time / res_std$median_time
  cat(sprintf("std=%.2fs cv=%.2fs overhead=%.1fx\n",
      res_std$median_time, res_cv$median_time, overhead))

  all_results <- rbind(all_results, data.frame(
    benchmark = "cv_overhead_std", m = 500, n = 1000, nnz = 25000, k = k,
    threads = ncpus, loss = "mse", precision = "double", cv = FALSE,
    res_std
  ))
  all_results <- rbind(all_results, data.frame(
    benchmark = "cv_overhead_cv", m = 500, n = 1000, nnz = 25000, k = k,
    threads = ncpus, loss = "mse", precision = "double", cv = TRUE,
    res_cv
  ))
}

rm(A, A_loss, A_cv); gc()

# -------------------------------------------------------------------
# Benchmark 6: Very large matrix (bigmem exclusive)
# -------------------------------------------------------------------
cat("\n--- Benchmark 6: Very large matrix (bigmem) ---\n")
set.seed(42)
m <- 50000; n <- 100000; density <- 0.001
nnz_target <- as.integer(m * n * density)
cat(sprintf("  Generating %d x %d matrix with %d nnz...\n", m, n, nnz_target))

A_big <- rsparsematrix(m, n, nnz = nnz_target)
A_big@x <- abs(A_big@x)
A_big <- as(A_big, "dgCMatrix")
cat(sprintf("  Matrix created: %d x %d, nnz = %d (%.1f MB)\n",
    nrow(A_big), ncol(A_big), nnzero(A_big),
    object.size(A_big) / 1024^2))

for (k in c(5, 10, 20)) {
  cat(sprintf("  k=%d ... ", k))
  t0 <- proc.time()[3]
  result <- nmf(A_big, k = k, seed = 1, maxit = 20, tol = 1e-10,
                threads = ncpus, resource = "cpu")
  elapsed <- proc.time()[3] - t0
  final_mse <- mse(result@w, result@d, result@h, A_big)
  cat(sprintf("%.2fs (MSE=%.4e)\n", elapsed, final_mse))

  all_results <- rbind(all_results, data.frame(
    benchmark = "bigmem_large", m = m, n = n, nnz = nnzero(A_big), k = k,
    threads = ncpus, loss = "mse", precision = "double", cv = FALSE,
    median_time = elapsed, min_time = elapsed, max_time = elapsed,
    final_mse = final_mse
  ))
}
rm(A_big); gc()

# -------------------------------------------------------------------
# Save results
# -------------------------------------------------------------------
outfile <- file.path(OUTDIR, sprintf("cpu_scaling_%s.csv", timestamp))
write.csv(all_results, outfile, row.names = FALSE)
cat(sprintf("\n=== Results saved to %s ===\n", outfile))
cat(sprintf("Total benchmarks: %d\n", nrow(all_results)))
print(all_results[, c("benchmark", "m", "n", "k", "threads", "loss",
                       "median_time", "final_mse")])
