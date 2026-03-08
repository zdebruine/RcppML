#!/usr/bin/env Rscript
# =============================================================================
# GPU vs CPU NMF Benchmark
#
# Compares RcppML NMF across:
#   - Backend: CPU (multi-threaded), GPU (auto-dispatch), GPU (forced single)
#   - Rank: 2, 4, 8, 16, 32, 64
#   - Precision: fp64 (double), fp32 (float)
#   - Dataset: pbmc3k (packaged), plus larger datasets if available
#
# Metrics:
#   - Wall-clock time (seconds)
#   - Time per iteration (ms)
#   - Final MSE loss
#   - Factor agreement (cosine similarity vs reference)
#   - GPU dispatch mode
#
# Usage:
#   Rscript benchmarks/R/gpu_cpu_benchmark.R [output_dir]
#
# Requirements:
#   - RcppML package installed (devtools::load_all("."))
#   - GPU library compiled (make -f src/Makefile.gpu install)
# =============================================================================

suppressPackageStartupMessages({
  library(Matrix)
})

# Load package — prefer development mode
if (file.exists("DESCRIPTION")) {
  devtools::load_all(".", quiet = TRUE)
} else {
  library(RcppML)
}

args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) > 0) args[1] else "benchmarks/results"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# =============================================================================
# Configuration
# =============================================================================

RANKS <- c(2, 4, 8, 16, 32, 64)
MAXIT <- 50         # Enough iterations for stable timing
TOL <- 1e-10        # Effectively disable early stopping
REPLICATES <- 3     # Statistical replicates
CPU_THREADS <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", 28))  # Match SLURM allocation

# Datasets: package bundled + any external binary datasets
load_datasets <- function() {
  datasets <- list()
  
  # Always available
  data(pbmc3k, envir = environment())
  datasets[["pbmc3k"]] <- pbmc3k
  
  # Check for larger external datasets (binary format from benchmarks/cuda/)
  # Note: Binary loading is slow in R for large files. Skip unless explicitly requested.
  if (Sys.getenv("LOAD_BIN_DATASETS", "0") == "1") {
    bin_dir <- "benchmarks/comprehensive"
    bin_files <- list.files(bin_dir, pattern = "\\.bin$", full.names = TRUE)
    
    if (length(bin_files) > 0) {
      cat("Found external binary datasets:\n")
      for (f in bin_files) {
        name <- tools::file_path_sans_ext(basename(f))
        tryCatch({
          con <- file(f, "rb")
          on.exit(close(con), add = TRUE)
          header <- readBin(con, "integer", n = 3, size = 4)
          m <- header[1]; n <- header[2]; nnz <- header[3]
          
          if (is.na(m) || is.na(n) || is.na(nnz) || m <= 0 || n <= 0 || nnz <= 0) {
            cat(sprintf("  %s: SKIPPED (invalid header)\n", name))
            next
          }
          
          # Read all triplets: each is (int32, int32, double) = 16 bytes
          rows <- integer(nnz); cols <- integer(nnz); vals <- numeric(nnz)
          for (idx in seq_len(nnz)) {
            rows[idx] <- readBin(con, "integer", n = 1, size = 4) + 1L
            cols[idx] <- readBin(con, "integer", n = 1, size = 4) + 1L
            vals[idx] <- readBin(con, "double", n = 1, size = 8)
          }
          close(con); on.exit(NULL)
          
          mat <- sparseMatrix(i = rows, j = cols, x = vals, dims = c(m, n))
          datasets[[name]] <- as(mat, "dgCMatrix")
          cat(sprintf("  %s: %d x %d, nnz=%d\n", name, m, n, nnz))
        }, error = function(e) {
          cat(sprintf("  %s: SKIPPED (%s)\n", name, conditionMessage(e)))
        })
      }
    }
  } else {
    cat("(Skipping external binary datasets; set LOAD_BIN_DATASETS=1 to include)\n")
  }

  # Use full pbmc3k for benchmarking (already loaded above)
  # Also load other packaged datasets for broader coverage
  tryCatch({
    data(aml, envir = environment())
    datasets[["aml"]] <- as(aml, "dgCMatrix")
  }, error = function(e) NULL)
  
  datasets
}

# =============================================================================
# Helper: compute MSE loss for result verification
# =============================================================================

compute_mse <- function(A, model) {
  # Uses RcppML internal loss if available
  if (is(model, "nmf")) {
    return(model@misc$loss)
  }
  
  # For GPU results returned as list
  if (is.list(model) && !is.null(model$loss)) {
    return(model$loss)
  }
  
  return(NA_real_)
}

# =============================================================================
# Helper: compare two NMF solutions by factor cosine similarity
# =============================================================================

factor_agreement <- function(ref, test) {
  # Extract W matrices
  if (is(ref, "nmf")) {
    W_ref <- ref@w
  } else {
    W_ref <- ref$w
  }
  
  if (is(test, "nmf")) {
    W_test <- test@w
  } else {
    W_test <- test$w
  }
  
  k <- ncol(W_ref)
  if (ncol(W_test) != k) return(NA_real_)
  
  # Compute pairwise cosine similarities and find best match
  sims <- numeric(k)
  for (i in seq_len(k)) {
    best <- -1
    for (j in seq_len(k)) {
      cs <- sum(W_ref[, i] * W_test[, j]) /
            (sqrt(sum(W_ref[, i]^2)) * sqrt(sum(W_test[, j]^2)) + 1e-16)
      if (abs(cs) > abs(best)) best <- cs
    }
    sims[i] <- abs(best)
  }
  
  mean(sims)
}

# =============================================================================
# Benchmark runner
# =============================================================================

run_benchmark <- function(A, k, backend, precision, threads = CPU_THREADS,
                          maxit = MAXIT, tol = TOL, seed = 42L) {
  
  # Configure backend
  if (backend == "cpu") {
    options(RcppML.gpu = FALSE)
    options(RcppML.threads = threads)
    prec_arg <- if (precision == "fp32") "float" else "double"
  } else if (backend == "gpu") {
    options(RcppML.gpu = TRUE)
    prec_arg <- if (precision == "fp32") "float" else "double"
  } else if (backend == "gpu_single") {
    options(RcppML.gpu = TRUE)
    options(RcppML.max_gpus = 1L)
    prec_arg <- if (precision == "fp32") "float" else "double"
  } else {
    stop("Unknown backend: ", backend)
  }
  
  # Warm-up run (small)
  tryCatch({
    invisible(nmf(A, k = k, maxit = 3, tol = 1, verbose = FALSE,
                  seed = seed, precision = prec_arg))
  }, error = function(e) NULL)
  gc()
  
  # Timed run
  start_time <- proc.time()
  result <- tryCatch({
    nmf(A, k = k, maxit = maxit, tol = tol, verbose = FALSE,
        seed = seed, precision = prec_arg)
  }, error = function(e) {
    message(sprintf("  ERROR (%s/%s/k=%d): %s", backend, precision, k,
                    conditionMessage(e)))
    NULL
  })
  elapsed <- (proc.time() - start_time)["elapsed"]
  
  # Reset options
  options(RcppML.gpu = "auto")
  options(RcppML.max_gpus = 0L)
  
  if (is.null(result)) {
    return(list(
      elapsed = NA, loss = NA, iterations = NA,
      time_per_iter = NA, gpu_mode = NA
    ))
  }
  
  # Extract metrics
  if (is(result, "nmf")) {
    iterations <- result@misc$iter
    loss <- result@misc$loss
    gpu_mode <- if (!is.null(attr(result, "gpu_mode"))) attr(result, "gpu_mode") else "cpu"
  } else {
    iterations <- result$iter
    loss <- result$loss
    gpu_mode <- if (!is.null(result$gpu_mode)) result$gpu_mode else "cpu"
  }
  
  list(
    elapsed = as.numeric(elapsed),
    loss = loss,
    iterations = iterations,
    time_per_iter = as.numeric(elapsed) / max(iterations, 1) * 1000,
    gpu_mode = gpu_mode,
    result = result
  )
}

# =============================================================================
# Main benchmark loop
# =============================================================================

cat("============================================================\n")
cat("RcppML NMF: GPU vs CPU Benchmark\n")
cat("============================================================\n\n")

# Check GPU availability
cat("GPU status: ")
if (gpu_available()) {
  info <- gpu_info()
  cat(sprintf("AVAILABLE (%d GPU%s)\n", nrow(info), if (nrow(info) > 1) "s" else ""))
  print(info)
  cat("\n")
  backends <- c("cpu", "gpu")
  if (nrow(info) > 1) backends <- c(backends, "gpu_single")
} else {
  cat("NOT AVAILABLE (CPU-only benchmark)\n\n")
  backends <- c("cpu")
}

precisions <- c("fp64", "fp32")

datasets <- load_datasets()
cat(sprintf("Datasets: %s\n", paste(names(datasets), collapse = ", ")))
cat(sprintf("Ranks: %s\n", paste(RANKS, collapse = ", ")))
cat(sprintf("Backends: %s\n", paste(backends, collapse = ", ")))
cat(sprintf("Precisions: %s\n", paste(precisions, collapse = ", ")))
cat(sprintf("Replicates: %d\n", REPLICATES))
cat(sprintf("Max iterations: %d\n\n", MAXIT))

results <- data.frame()
total_runs <- length(datasets) * length(RANKS) * length(backends) *
              length(precisions) * REPLICATES
run_count <- 0

for (ds_name in names(datasets)) {
  A <- datasets[[ds_name]]
  m <- nrow(A); n <- ncol(A); nnz <- length(A@x)
  cat(sprintf("\n--- Dataset: %s (%d x %d, nnz=%d) ---\n", ds_name, m, n, nnz))
  
  for (k in RANKS) {
    # CPU reference for accuracy comparison
    ref_result <- NULL
    
    for (backend in backends) {
      for (prec in precisions) {
        for (rep in seq_len(REPLICATES)) {
          run_count <- run_count + 1
          cat(sprintf("[%d/%d] %s/%s/%s/k=%d/rep=%d ... ",
                      run_count, total_runs, ds_name, backend, prec, k, rep))
          
          timing <- run_benchmark(A, k, backend, prec, seed = 42L + rep)
          
          # Compute factor agreement with CPU fp64 reference
          agreement <- NA_real_
          if (backend == "cpu" && prec == "fp64" && rep == 1 && !is.null(timing$result)) {
            ref_result <- timing$result
          } else if (!is.null(ref_result) && !is.null(timing$result)) {
            agreement <- tryCatch(
              factor_agreement(ref_result, timing$result),
              error = function(e) NA_real_
            )
          }
          
          row <- data.frame(
            dataset = ds_name,
            rows = m, cols = n, nnz = nnz,
            rank = k,
            backend = backend,
            precision = prec,
            replicate = rep,
            elapsed_sec = timing$elapsed,
            time_per_iter_ms = timing$time_per_iter,
            final_loss = timing$loss,
            iterations = timing$iterations,
            gpu_mode = as.character(timing$gpu_mode),
            factor_agreement = agreement,
            stringsAsFactors = FALSE
          )
          results <- rbind(results, row)
          
          if (is.na(timing$elapsed)) {
            cat("FAILED\n")
          } else {
            cat(sprintf("%.2f sec (%.1f ms/iter) loss=%.6e",
                        timing$elapsed, timing$time_per_iter, timing$loss))
            if (!is.na(agreement)) cat(sprintf(" agree=%.4f", agreement))
            if (timing$gpu_mode != "cpu") cat(sprintf(" [%s]", timing$gpu_mode))
            cat("\n")
          }
          
          # Clean up per-iteration
          timing$result <- NULL
          gc()
        }
      }
    }
  }
}

# =============================================================================
# Save results
# =============================================================================

timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
output_file <- file.path(output_dir, sprintf("gpu_cpu_benchmark_%s.csv", timestamp))
write.csv(results, output_file, row.names = FALSE)
cat(sprintf("\n\nResults saved to: %s\n", output_file))

# =============================================================================
# Summary statistics
# =============================================================================

cat("\n============================================================\n")
cat("SUMMARY: Median time per iteration (ms)\n")
cat("============================================================\n")

if (nrow(results) > 0) {
  agg <- aggregate(
    time_per_iter_ms ~ dataset + rank + backend + precision,
    data = results, FUN = median, na.rm = TRUE
  )
  
  # Reshape for comparison
  for (ds in unique(agg$dataset)) {
    cat(sprintf("\n--- %s ---\n", ds))
    sub <- agg[agg$dataset == ds, ]
    
    for (prec in precisions) {
      sub_p <- sub[sub$precision == prec, ]
      if (nrow(sub_p) == 0) next
      
      cat(sprintf("\n  Precision: %s\n", prec))
      cat(sprintf("  %6s", "k"))
      for (b in backends) cat(sprintf(" %12s", b))
      if (length(backends) >= 2) cat(sprintf(" %12s", "speedup"))
      cat("\n")
      
      for (k in sort(unique(sub_p$rank))) {
        cat(sprintf("  %6d", k))
        cpu_time <- NA
        gpu_time <- NA
        for (b in backends) {
          val <- sub_p$time_per_iter_ms[sub_p$rank == k & sub_p$backend == b]
          if (length(val) > 0) {
            cat(sprintf(" %10.1f ms", val[1]))
            if (b == "cpu") cpu_time <- val[1]
            if (b == "gpu") gpu_time <- val[1]
          } else {
            cat(sprintf(" %12s", "N/A"))
          }
        }
        if (length(backends) >= 2 && !is.na(cpu_time) && !is.na(gpu_time) && gpu_time > 0) {
          cat(sprintf(" %10.1fx", cpu_time / gpu_time))
        }
        cat("\n")
      }
    }
  }
}

# =============================================================================
# Accuracy regression check
# =============================================================================

cat("\n============================================================\n")
cat("ACCURACY CHECK: Factor agreement (cosine) vs CPU fp64 reference\n")
cat("============================================================\n")

if (nrow(results) > 0 && any(!is.na(results$factor_agreement))) {
  acc <- results[!is.na(results$factor_agreement), ]
  agg_acc <- aggregate(
    factor_agreement ~ dataset + rank + backend + precision,
    data = acc, FUN = min, na.rm = TRUE
  )
  
  low_agreement <- agg_acc[agg_acc$factor_agreement < 0.90, ]
  if (nrow(low_agreement) > 0) {
    cat("\nWARNING: Low factor agreement cases:\n")
    print(low_agreement)
  } else {
    cat("\nAll factor agreements >= 0.90 — no accuracy regressions detected.\n")
  }
  
  # Also check loss convergence
  loss_check <- aggregate(
    final_loss ~ dataset + rank + backend + precision,
    data = results[!is.na(results$final_loss), ],
    FUN = median, na.rm = TRUE
  )
  
  cat("\nMedian final loss by backend/precision:\n")
  print(loss_check)
}

cat("\nBenchmark complete.\n")
