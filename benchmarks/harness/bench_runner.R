#!/usr/bin/env Rscript
# bench_runner.R — Reusable benchmark timing and reporting utilities
#
# Source this file to get run_bench() and report_bench() for all agents.
#
# Usage:
#   source("benchmarks/harness/bench_runner.R")
#   results <- run_bench("my_test", quote(nmf(A, k=16, tol=1e-10, maxit=20)))
#   report_bench(results)

# ============================================================================
# Machine metadata (cached after first call)
# ============================================================================
.bench_metadata <- NULL

get_bench_metadata <- function() {
  if (!is.null(.bench_metadata)) return(.bench_metadata)

  hostname <- Sys.info()[["nodename"]]
  cpu_info <- tryCatch(
    trimws(system("lscpu | grep 'Model name' | sed 's/.*:\\s*//'", intern = TRUE)),
    error = function(e) "unknown"
  )
  gpu_info <- tryCatch({
    out <- system("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1",
                  intern = TRUE)
    if (length(out) == 0 || nchar(out) == 0) NA_character_ else trimws(out)
  }, error = function(e) NA_character_)

  omp <- as.integer(Sys.getenv("OMP_NUM_THREADS", "1"))
  ram_gb <- tryCatch({
    mem_line <- system("grep MemTotal /proc/meminfo", intern = TRUE)
    round(as.numeric(gsub("[^0-9]", "", mem_line)) / 1024 / 1024, 1)
  }, error = function(e) NA_real_)

  meta <- list(
    node = hostname,
    cpu = cpu_info,
    gpu = gpu_info,
    omp_threads = omp,
    ram_gb = ram_gb
  )
  .bench_metadata <<- meta
  meta
}

# ============================================================================
# Core benchmarking function
# ============================================================================

#' Run a benchmark with warmup, GC, and per-replicate timing
#'
#' @param label   Character label for this benchmark
#' @param expr    An expression to benchmark (use quote() or bquote())
#' @param n_reps  Number of replicates (default 5)
#' @param warmup  Number of warmup runs (default 1)
#' @param envir   Environment in which to evaluate expr
#' @param extra   Named list of extra columns (k, m, n, nnz, backend, loss)
#' @return data.frame with one row per replicate
run_bench <- function(label, expr, n_reps = 5L, warmup = 1L,
                      envir = parent.frame(),
                      extra = list()) {

  # Warmup runs (discarded)
  for (i in seq_len(warmup)) {
    tryCatch(eval(expr, envir = envir), error = function(e) NULL)
  }

  # Timed replicates
  times_ms <- numeric(n_reps)
  for (i in seq_len(n_reps)) {
    gc(verbose = FALSE, full = TRUE)
    t0 <- proc.time()["elapsed"]
    tryCatch(eval(expr, envir = envir), error = function(e) {
      warning(sprintf("Rep %d of '%s' failed: %s", i, label, conditionMessage(e)))
    })
    t1 <- proc.time()["elapsed"]
    times_ms[i] <- (t1 - t0) * 1000
  }

  # Build result data.frame
  timestamp <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  df <- data.frame(
    label = rep(label, n_reps),
    rep = seq_len(n_reps),
    time_ms = times_ms,
    k = if (!is.null(extra$k)) extra$k else NA_integer_,
    m = if (!is.null(extra$m)) extra$m else NA_integer_,
    n = if (!is.null(extra$n)) extra$n else NA_integer_,
    nnz = if (!is.null(extra$nnz)) extra$nnz else NA_integer_,
    backend = if (!is.null(extra$backend)) extra$backend else NA_character_,
    loss = if (!is.null(extra$loss)) extra$loss else NA_character_,
    timestamp = timestamp,
    stringsAsFactors = FALSE
  )
  df
}

# ============================================================================
# Reporting
# ============================================================================

#' Print summary and optionally save CSV
#'
#' @param results   data.frame from run_bench() or rbind of multiple
#' @param baseline  Optional data.frame of baseline results for speedup calc
#' @param save_csv  Logical; if TRUE, save to benchmarks/harness/results/
#' @param tag       Optional tag for the CSV filename
#' @return Invisible summary data.frame
report_bench <- function(results, baseline = NULL, save_csv = TRUE, tag = NULL) {

  # Aggregate per label
  labels <- unique(results$label)
  summary_rows <- lapply(labels, function(lbl) {
    sub <- results[results$label == lbl, ]
    med <- median(sub$time_ms)
    iqr_val <- IQR(sub$time_ms)
    n <- nrow(sub)

    speedup <- NA_real_
    if (!is.null(baseline)) {
      bl_sub <- baseline[baseline$label == lbl, ]
      if (nrow(bl_sub) > 0) {
        speedup <- median(bl_sub$time_ms) / med
      }
    }

    data.frame(
      label = lbl,
      median_ms = round(med, 2),
      iqr_ms = round(iqr_val, 2),
      reps = n,
      speedup = if (is.na(speedup)) NA_real_ else round(speedup, 3),
      stringsAsFactors = FALSE
    )
  })
  summary_df <- do.call(rbind, summary_rows)

  # Print table
  cat(sprintf("\n%-50s %10s %10s %5s %8s\n",
              "Label", "Median(ms)", "IQR(ms)", "Reps", "Speedup"))
  cat(paste(rep("-", 87), collapse = ""), "\n")
  for (i in seq_len(nrow(summary_df))) {
    r <- summary_df[i, ]
    sp_str <- if (is.na(r$speedup)) "   ---" else sprintf("%7.3fx", r$speedup)
    cat(sprintf("%-50s %10.2f %10.2f %5d %8s\n",
                r$label, r$median_ms, r$iqr_ms, r$reps, sp_str))
  }
  cat("\n")

  # Save CSV
  if (save_csv) {
    results_dir <- file.path(
      normalizePath("benchmarks/harness/results", mustWork = FALSE))
    dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

    ts <- format(Sys.time(), "%Y%m%d_%H%M%S")
    label_part <- if (!is.null(tag)) tag else gsub("[^a-zA-Z0-9_]", "", labels[1])
    csv_path <- file.path(results_dir, sprintf("%s_%s.csv", ts, label_part))
    write.csv(results, csv_path, row.names = FALSE)
    cat(sprintf("Saved: %s\n", csv_path))
  }

  invisible(summary_df)
}

# ============================================================================
# Convenience: combine multiple run_bench results
# ============================================================================

#' Combine results from multiple run_bench calls
bench_combine <- function(...) {
  dfs <- list(...)
  do.call(rbind, dfs)
}
