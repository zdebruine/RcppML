#!/usr/bin/env Rscript
# =============================================================================
# Streaming NMF Deep Profiling Investigation
# =============================================================================
# Purpose: Understand WHERE time is spent in streaming vs in-memory NMF,
#          identify bottlenecks, and quantify the gap.
#
# Approach: Run identical factorization parameters on the same data through
#           both the in-memory and streaming (SPZ) paths, with fine-grained
#           profiling enabled to break down per-section timings.
#
# Datasets: Real single-cell data from SeuratData (hcabm40k, bmcite, pbmc3k)
#           plus synthetic matrices for controlled size/sparsity sweeps.
#
# Output: CSV files with per-section profiling + summary analysis to stdout
# =============================================================================

suppressPackageStartupMessages({
  library(RcppML, lib.loc = "/tmp/rcppml_agent7")
  library(Matrix)
})

cat("============================================================\n")
cat("STREAMING NMF DEEP PROFILING INVESTIGATION\n")
cat("============================================================\n\n")

# ---- Configuration ----
OUT_DIR <- "/mnt/home/debruinz/RcppML-2/benchmarks/stream_opt"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
SPZ_DIR <- file.path(OUT_DIR, "spz_data")
dir.create(SPZ_DIR, showWarnings = FALSE)

RANKS <- c(8, 16, 32)
MAXIT <- 10
TOL   <- 1e-20  # Force exactly MAXIT iterations
THREADS <- 4
SEED  <- 42
NREPS <- 3  # Replicates per condition

# ---- Helper: Create SPZ from sparse matrix ----
create_spz <- function(mat, name) {
  spz_path <- file.path(SPZ_DIR, paste0(name, ".spz"))
  if (!file.exists(spz_path)) {
    cat(sprintf("  Creating SPZ: %s (%d x %d, nnz=%d)\n",
                name, nrow(mat), ncol(mat), length(mat@x)))
    sp_write(mat, spz_path, include_transpose = TRUE)
  }
  spz_path
}

# ---- Helper: Run one NMF benchmark ----
run_benchmark <- function(data, k, mode, label, rep_id) {
  gc(verbose = FALSE)
  
  t0 <- proc.time()
  
  if (mode == "memory") {
    result <- nmf(data, k = k, tol = TOL, maxit = MAXIT,
                  threads = THREADS, seed = SEED, verbose = FALSE,
                  profile = TRUE)
  } else if (mode == "stream") {
    result <- nmf(data, k = k, tol = TOL, maxit = MAXIT,
                  threads = THREADS, seed = SEED, verbose = FALSE,
                  profile = TRUE)
  }
  
  elapsed <- (proc.time() - t0)[["elapsed"]]
  
  # Extract profiling data
  prof <- NULL
  if (!is.null(result@misc$profile)) {
    prof <- result@misc$profile
  }
  
  # Build result row
  data.frame(
    label     = label,
    mode      = mode,
    rank      = k,
    rep       = rep_id,
    total_sec = elapsed,
    iterations = result@misc$iterations,
    final_loss = tail(result@misc$loss_history, 1),
    # Profile sections (NA if not available)
    t_total_iter    = if (!is.null(prof)) prof["total_iter"]    else NA_real_,
    t_chunk_gram    = if (!is.null(prof)) prof["chunk_gram"]    else NA_real_,
    t_chunk_read    = if (!is.null(prof)) prof["chunk_read"]    else NA_real_,
    t_chunk_rhs     = if (!is.null(prof)) prof["chunk_rhs"]     else NA_real_,
    t_chunk_nnls    = if (!is.null(prof)) prof["chunk_nnls"]    else NA_real_,
    t_gram_accumulate = if (!is.null(prof)) prof["gram_accumulate"] else NA_real_,
    stringsAsFactors = FALSE
  )
}

# ---- Phase 1: Real-world datasets ----
cat("\n==== PHASE 1: Real-World Datasets ====\n\n")

datasets <- list()

# Try to load SeuratData datasets
tryCatch({
  library(SeuratData, quietly = TRUE)
  
  # hcabm40k: 40k cells (largest)
  tryCatch({
    data("hcabm40k")
    mat <- hcabm40k[["RNA"]]$counts
    mat <- as(mat, "dgCMatrix")
    # Filter: keep genes with at least 10 nonzeros
    gene_nnz <- diff(mat@p)  # columns = genes for Seurat (cells x genes transposed)
    # Actually Seurat stores genes x cells, so mat is genes x cells
    # Keep genes (rows) with sufficient expression
    row_nnz <- tabulate(mat@i + 1, nbins = nrow(mat))
    keep_rows <- which(row_nnz >= 10)
    mat <- mat[keep_rows, ]
    datasets[["hcabm40k"]] <- mat
    cat(sprintf("  hcabm40k: %d genes x %d cells, nnz=%d (%.1f%% sparse)\n",
                nrow(mat), ncol(mat), length(mat@x),
                100 * (1 - length(mat@x) / (as.double(nrow(mat)) * ncol(mat)))))
    rm(hcabm40k); gc(verbose = FALSE)
  }, error = function(e) cat(sprintf("  hcabm40k: SKIP (%s)\n", e$message)))
  
  # bmcite: 30k cells
  tryCatch({
    data("bmcite")
    mat <- bmcite[["RNA"]]$counts
    mat <- as(mat, "dgCMatrix")
    row_nnz <- tabulate(mat@i + 1, nbins = nrow(mat))
    keep_rows <- which(row_nnz >= 10)
    mat <- mat[keep_rows, ]
    datasets[["bmcite"]] <- mat
    cat(sprintf("  bmcite:   %d genes x %d cells, nnz=%d (%.1f%% sparse)\n",
                nrow(mat), ncol(mat), length(mat@x),
                100 * (1 - length(mat@x) / (as.double(nrow(mat)) * ncol(mat)))))
    rm(bmcite); gc(verbose = FALSE)
  }, error = function(e) cat(sprintf("  bmcite: SKIP (%s)\n", e$message)))
  
  # pbmcsca: comparison
  tryCatch({
    data("pbmcsca")
    mat <- pbmcsca[["RNA"]]$counts
    mat <- as(mat, "dgCMatrix")
    row_nnz <- tabulate(mat@i + 1, nbins = nrow(mat))
    keep_rows <- which(row_nnz >= 10)
    mat <- mat[keep_rows, ]
    datasets[["pbmcsca"]] <- mat
    cat(sprintf("  pbmcsca:  %d genes x %d cells, nnz=%d (%.1f%% sparse)\n",
                nrow(mat), ncol(mat), length(mat@x),
                100 * (1 - length(mat@x) / (as.double(nrow(mat)) * ncol(mat)))))
    rm(pbmcsca); gc(verbose = FALSE)
  }, error = function(e) cat(sprintf("  pbmcsca: SKIP (%s)\n", e$message)))
  
}, error = function(e) {
  cat(sprintf("  SeuratData not available: %s\n", e$message))
})

# ---- Phase 2: Synthetic matrices for controlled experiments ----
cat("\n==== PHASE 2: Synthetic Matrices ====\n\n")

gen_sparse <- function(m, n, density, seed = 42) {
  set.seed(seed)
  nnz <- as.integer(m * n * density)
  i <- sample.int(m, nnz, replace = TRUE) - 1L
  j <- sample.int(n, nnz, replace = TRUE) - 1L
  x <- runif(nnz, 0.1, 10)
  mat <- sparseMatrix(i = i + 1L, j = j + 1L, x = x, dims = c(m, n))
  as(mat, "dgCMatrix")
}

# Small (baseline), medium, large synthetic
synth_configs <- list(
  list(name = "synth_5k_2k",   m = 5000,  n = 2000,  density = 0.05),
  list(name = "synth_10k_5k",  m = 10000, n = 5000,  density = 0.03),
  list(name = "synth_20k_10k", m = 20000, n = 10000, density = 0.02)
)

for (cfg in synth_configs) {
  mat <- gen_sparse(cfg$m, cfg$n, cfg$density)
  datasets[[cfg$name]] <- mat
  cat(sprintf("  %s: %d x %d, nnz=%d (%.1f%% dense)\n",
              cfg$name, nrow(mat), ncol(mat), length(mat@x),
              100 * length(mat@x) / (as.double(nrow(mat)) * ncol(mat))))
}

# ---- Phase 3: Run the profiling ----
cat("\n==== PHASE 3: Running Profiled Benchmarks ====\n\n")

all_results <- data.frame()

for (ds_name in names(datasets)) {
  mat <- datasets[[ds_name]]
  spz_path <- create_spz(mat, ds_name)
  
  cat(sprintf("\n--- Dataset: %s ---\n", ds_name))
  
  for (k in RANKS) {
    # Skip large ranks for small matrices
    if (k >= min(nrow(mat), ncol(mat))) next
    
    for (rep_id in seq_len(NREPS)) {
      # In-memory
      cat(sprintf("  memory k=%d rep=%d...", k, rep_id))
      r_mem <- tryCatch(
        run_benchmark(mat, k, "memory", ds_name, rep_id),
        error = function(e) {
          cat(sprintf(" ERROR: %s\n", e$message))
          NULL
        }
      )
      if (!is.null(r_mem)) {
        cat(sprintf(" %.2fs\n", r_mem$total_sec))
        all_results <- rbind(all_results, r_mem)
      }
      
      # Streaming
      cat(sprintf("  stream k=%d rep=%d...", k, rep_id))
      r_str <- tryCatch(
        run_benchmark(spz_path, k, "stream", ds_name, rep_id),
        error = function(e) {
          cat(sprintf(" ERROR: %s\n", e$message))
          NULL
        }
      )
      if (!is.null(r_str)) {
        cat(sprintf(" %.2fs\n", r_str$total_sec))
        all_results <- rbind(all_results, r_str)
      }
    }
  }
}

# ---- Phase 4: Analysis ----
cat("\n\n============================================================\n")
cat("ANALYSIS\n")
cat("============================================================\n\n")

# Save raw results
csv_path <- file.path(OUT_DIR, "streaming_profile_raw.csv")
write.csv(all_results, csv_path, row.names = FALSE)
cat(sprintf("Raw results saved to: %s\n\n", csv_path))

# Summary: median across reps
agg <- aggregate(
  cbind(total_sec, t_total_iter, t_chunk_gram, t_chunk_read,
        t_chunk_rhs, t_chunk_nnls, t_gram_accumulate) ~ label + mode + rank,
  data = all_results,
  FUN = median,
  na.rm = TRUE
)

# Speedup comparison
cat("=== STREAMING vs IN-MEMORY SPEEDUP ===\n\n")
cat(sprintf("%-20s %4s %10s %10s %8s\n", "Dataset", "k", "Memory(s)", "Stream(s)", "Ratio"))
cat(paste(rep("-", 60), collapse = ""), "\n")

for (ds in unique(agg$label)) {
  for (k in sort(unique(agg$rank))) {
    mem_row <- agg[agg$label == ds & agg$mode == "memory" & agg$rank == k, ]
    str_row <- agg[agg$label == ds & agg$mode == "stream" & agg$rank == k, ]
    if (nrow(mem_row) > 0 && nrow(str_row) > 0) {
      ratio <- str_row$total_sec / mem_row$total_sec
      cat(sprintf("%-20s %4d %10.2f %10.2f %7.1fx\n",
                  ds, k, mem_row$total_sec, str_row$total_sec, ratio))
    }
  }
}

# Profiling section breakdown for streaming
cat("\n=== STREAMING PROFILING BREAKDOWN (ms, median across reps) ===\n\n")
str_agg <- agg[agg$mode == "stream", ]
if (nrow(str_agg) > 0) {
  cat(sprintf("%-20s %4s %10s %10s %10s %10s %10s %10s\n",
              "Dataset", "k", "Total", "Gram", "ChRead", "ChRHS", "ChNNLS", "GramAcc"))
  cat(paste(rep("-", 90), collapse = ""), "\n")
  for (i in seq_len(nrow(str_agg))) {
    r <- str_agg[i, ]
    cat(sprintf("%-20s %4d %10.0f %10.0f %10.0f %10.0f %10.0f %10.0f\n",
                r$label, r$rank,
                r$t_total_iter, r$t_chunk_gram, r$t_chunk_read,
                r$t_chunk_rhs, r$t_chunk_nnls, r$t_gram_accumulate))
  }
}

# Profiling as percentage of total
cat("\n=== STREAMING TIME DISTRIBUTION (% of total iter time) ===\n\n")
if (nrow(str_agg) > 0) {
  cat(sprintf("%-20s %4s %8s %8s %8s %8s %8s %8s\n",
              "Dataset", "k", "Gram%", "Read%", "RHS%", "NNLS%", "GrAcc%", "Other%"))
  cat(paste(rep("-", 80), collapse = ""), "\n")
  for (i in seq_len(nrow(str_agg))) {
    r <- str_agg[i, ]
    total <- r$t_total_iter
    if (!is.na(total) && total > 0) {
      pct_gram <- 100 * r$t_chunk_gram / total
      pct_read <- 100 * r$t_chunk_read / total
      pct_rhs  <- 100 * r$t_chunk_rhs / total
      pct_nnls <- 100 * r$t_chunk_nnls / total
      pct_gacc <- 100 * r$t_gram_accumulate / total
      pct_other <- 100 - pct_gram - pct_read - pct_rhs - pct_nnls - pct_gacc
      cat(sprintf("%-20s %4d %7.1f%% %7.1f%% %7.1f%% %7.1f%% %7.1f%% %7.1f%%\n",
                  r$label, r$rank, pct_gram, pct_read, pct_rhs, pct_nnls, pct_gacc, pct_other))
    }
  }
}

# Per-iteration timing from in-memory profile
cat("\n=== IN-MEMORY PROFILING BREAKDOWN (ms, median across reps) ===\n\n")
mem_agg <- agg[agg$mode == "memory", ]
if (nrow(mem_agg) > 0) {
  cat(sprintf("%-20s %4s %10s %10s %10s %10s %10s\n",
              "Dataset", "k", "Total", "Gram", "RHS", "NNLS", "GramAcc"))
  cat(paste(rep("-", 75), collapse = ""), "\n")
  for (i in seq_len(nrow(mem_agg))) {
    r <- mem_agg[i, ]
    cat(sprintf("%-20s %4d %10.0f %10.0f %10.0f %10.0f %10.0f\n",
                r$label, r$rank,
                r$t_total_iter, r$t_chunk_gram, r$t_chunk_rhs,
                r$t_chunk_nnls, r$t_gram_accumulate))
  }
}

# Compute-only time comparison (streaming total - streaming read = streaming compute)
cat("\n=== COMPUTE-ONLY COMPARISON (streaming_total - chunk_read) ===\n\n")
cat(sprintf("%-20s %4s %10s %10s %10s %8s\n",
            "Dataset", "k", "Mem(ms)", "StrCompute", "StrRead", "Ratio"))
cat(paste(rep("-", 65), collapse = ""), "\n")
for (ds in unique(agg$label)) {
  for (k in sort(unique(agg$rank))) {
    mem_row <- agg[agg$label == ds & agg$mode == "memory" & agg$rank == k, ]
    str_row <- agg[agg$label == ds & agg$mode == "stream" & agg$rank == k, ]
    if (nrow(mem_row) > 0 && nrow(str_row) > 0) {
      str_compute <- str_row$t_total_iter - str_row$t_chunk_read
      mem_total <- mem_row$t_total_iter
      if (!is.na(str_compute) && !is.na(mem_total) && mem_total > 0) {
        ratio <- str_compute / mem_total
        cat(sprintf("%-20s %4d %10.0f %10.0f %10.0f %7.1fx\n",
                    ds, k, mem_total, str_compute, str_row$t_chunk_read, ratio))
      }
    }
  }
}

cat("\n\nDone. Results in:", OUT_DIR, "\n")
