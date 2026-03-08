#!/usr/bin/env Rscript
# GPU vs CPU Scaling Benchmark
# Called from: benchmarks/slurm/gpu_cpu_scaling.sbatch

library(RcppML)
library(Matrix)

cat("\n=== GPU vs CPU Benchmark ===\n")
OUTDIR <- "benchmarks/results"
timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
ncpus <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "28"))

# Verify GPU
stopifnot(gpu_available(force_recheck = TRUE))
cat("GPU: "); print(gpu_info())

bench <- function(A, k, resource, precision = "double", maxit = 50,
                  L1 = c(0, 0), cv = FALSE, reps = 3) {
  times <- numeric(reps)
  mse_val <- NA

  for (r in seq_len(reps)) {
    if (r == 1) {
      suppressWarnings(nmf(A, k = min(k, 2), seed = 1, maxit = 3,
                           tol = 1e-10, resource = resource))
    }
    t0 <- proc.time()[3]
    if (cv) {
      result <- nmf(A, k = k, seed = r, maxit = maxit, tol = 1e-10,
                    resource = resource, precision = precision,
                    test_fraction = 0.1)
    } else {
      result <- nmf(A, k = k, seed = r, maxit = maxit, tol = 1e-10,
                    resource = resource, precision = precision, L1 = L1)
    }
    times[r] <- proc.time()[3] - t0
    if (r == reps && !cv) {
      mse_val <- mse(result@w, result@d, result@h, A)
    }
  }

  data.frame(median_time = median(times), min_time = min(times),
             max_time = max(times), final_mse = mse_val)
}

results <- data.frame()

# -------------------------------------------------------------------
# Benchmark 1: GPU speedup vs matrix size
# -------------------------------------------------------------------
cat("\n--- 1: GPU speedup vs matrix size ---\n")
sizes <- list(
  c(500,   1000,  0.05),
  c(1000,  2000,  0.05),
  c(2000,  5000,  0.02),
  c(5000,  10000, 0.01),
  c(10000, 20000, 0.005)
)

for (sz in sizes) {
  m <- sz[1]; n <- sz[2]; density <- sz[3]; k <- 10
  set.seed(42)
  A <- rsparsematrix(m, n, nnz = as.integer(m * n * density))
  A@x <- abs(A@x); A <- as(A, "dgCMatrix")

  cat(sprintf("  %dx%d nnz=%d: ", m, n, nnzero(A)))
  rc <- bench(A, k, "cpu"); rg <- bench(A, k, "gpu")
  speedup <- rc$median_time / rg$median_time
  cat(sprintf("CPU=%.2fs GPU=%.2fs speedup=%.1fx\n",
      rc$median_time, rg$median_time, speedup))

  results <- rbind(results, data.frame(
    bench = "size_speedup", m = m, n = n, nnz = nnzero(A), k = k,
    resource = "cpu", precision = "double", L1 = 0, cv = FALSE, rc))
  results <- rbind(results, data.frame(
    bench = "size_speedup", m = m, n = n, nnz = nnzero(A), k = k,
    resource = "gpu", precision = "double", L1 = 0, cv = FALSE, rg))
  rm(A); gc()
}

# -------------------------------------------------------------------
# Benchmark 2: GPU speedup vs rank
# -------------------------------------------------------------------
cat("\n--- 2: GPU speedup vs rank ---\n")
set.seed(42)
A <- rsparsematrix(2000, 5000, nnz = 200000)
A@x <- abs(A@x); A <- as(A, "dgCMatrix")

for (k in c(2, 4, 8, 16, 32, 64)) {
  cat(sprintf("  k=%d: ", k))
  rc <- bench(A, k, "cpu"); rg <- bench(A, k, "gpu")
  speedup <- rc$median_time / rg$median_time
  cat(sprintf("CPU=%.2fs GPU=%.2fs speedup=%.1fx\n",
      rc$median_time, rg$median_time, speedup))

  results <- rbind(results, data.frame(
    bench = "rank_speedup", m = 2000, n = 5000, nnz = nnzero(A), k = k,
    resource = "cpu", precision = "double", L1 = 0, cv = FALSE, rc))
  results <- rbind(results, data.frame(
    bench = "rank_speedup", m = 2000, n = 5000, nnz = nnzero(A), k = k,
    resource = "gpu", precision = "double", L1 = 0, cv = FALSE, rg))
}

# -------------------------------------------------------------------
# Benchmark 3: fp32 vs fp64 on GPU
# -------------------------------------------------------------------
cat("\n--- 3: fp32 vs fp64 on GPU ---\n")
for (k in c(8, 16, 32)) {
  cat(sprintf("  k=%d: ", k))
  rg64 <- bench(A, k, "gpu", precision = "double")
  rg32 <- bench(A, k, "gpu", precision = "float")
  speedup <- rg64$median_time / rg32$median_time
  cat(sprintf("fp64=%.2fs fp32=%.2fs speedup=%.1fx (MSE fp64=%.4e fp32=%.4e)\n",
      rg64$median_time, rg32$median_time, speedup,
      rg64$final_mse, rg32$final_mse))

  results <- rbind(results, data.frame(
    bench = "precision", m = 2000, n = 5000, nnz = nnzero(A), k = k,
    resource = "gpu", precision = "double", L1 = 0, cv = FALSE, rg64))
  results <- rbind(results, data.frame(
    bench = "precision", m = 2000, n = 5000, nnz = nnzero(A), k = k,
    resource = "gpu", precision = "float", L1 = 0, cv = FALSE, rg32))
}

# -------------------------------------------------------------------
# Benchmark 4: Regularization overhead on GPU
# -------------------------------------------------------------------
cat("\n--- 4: Regularization overhead on GPU ---\n")
rg_base <- bench(A, 10, "gpu")
rg_l1   <- bench(A, 10, "gpu", L1 = c(0.1, 0.1))
cat(sprintf("  No reg: %.2fs  L1=0.1: %.2fs  overhead: %.0f%%\n",
    rg_base$median_time, rg_l1$median_time,
    (rg_l1$median_time / rg_base$median_time - 1) * 100))

results <- rbind(results, data.frame(
  bench = "gpu_reg", m = 2000, n = 5000, nnz = nnzero(A), k = 10,
  resource = "gpu", precision = "double", L1 = 0, cv = FALSE, rg_base))
results <- rbind(results, data.frame(
  bench = "gpu_reg", m = 2000, n = 5000, nnz = nnzero(A), k = 10,
  resource = "gpu", precision = "double", L1 = 0.1, cv = FALSE, rg_l1))

# -------------------------------------------------------------------
# Benchmark 5: GPU CV NMF
# -------------------------------------------------------------------
cat("\n--- 5: GPU CV NMF ---\n")
set.seed(42)
A_cv <- rsparsematrix(1000, 2000, nnz = 50000)
A_cv@x <- abs(A_cv@x); A_cv <- as(A_cv, "dgCMatrix")

for (k in c(4, 8, 16)) {
  cat(sprintf("  k=%d: ", k))
  rc_std <- bench(A_cv, k, "cpu")
  rg_std <- bench(A_cv, k, "gpu")
  rc_cv  <- bench(A_cv, k, "cpu", cv = TRUE)
  tryCatch({
    rg_cv <- bench(A_cv, k, "gpu", cv = TRUE)
    cat(sprintf("CPU=%.2f/%.2fs GPU=%.2f/%.2fs (std/cv)\n",
        rc_std$median_time, rc_cv$median_time,
        rg_std$median_time, rg_cv$median_time))

    results <- rbind(results, data.frame(
      bench = "gpu_cv", m = 1000, n = 2000, nnz = 50000, k = k,
      resource = "gpu", precision = "double", L1 = 0, cv = TRUE, rg_cv))
  }, error = function(e) {
    cat(sprintf("GPU CV error: %s\n", conditionMessage(e)))
  })

  results <- rbind(results, data.frame(
    bench = "gpu_cv_cpu", m = 1000, n = 2000, nnz = 50000, k = k,
    resource = "cpu", precision = "double", L1 = 0, cv = TRUE, rc_cv))
}

# -------------------------------------------------------------------
# Save
# -------------------------------------------------------------------
outfile <- file.path(OUTDIR, sprintf("gpu_scaling_%s.csv", timestamp))
write.csv(results, outfile, row.names = FALSE)
cat(sprintf("\n=== Results saved to %s (%d rows) ===\n", outfile, nrow(results)))

# Print speedup summary
cat("\n=== Speedup Summary ===\n")
speed <- results[results$bench == "size_speedup",]
if (nrow(speed) > 0) {
  cpu_rows <- speed[speed$resource == "cpu",]
  gpu_rows <- speed[speed$resource == "gpu",]
  for (i in seq_len(nrow(cpu_rows))) {
    cat(sprintf("  %dx%d: %.1fx GPU speedup\n", cpu_rows$m[i], cpu_rows$n[i],
        cpu_rows$median_time[i] / gpu_rows$median_time[i]))
  }
}
