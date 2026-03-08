#!/usr/bin/env Rscript
# run_all.R — Master benchmark harness for RcppML
#
# Usage:
#   Rscript benchmarks/harness/run_all.R               # Run all suites
#   Rscript benchmarks/harness/run_all.R --suite svd_methods  # Single suite
#   Rscript benchmarks/harness/run_all.R --gpu          # Include GPU suites
#
# Must be run on a compute node (never on the login node).

# ============================================================================
# Setup
# ============================================================================

harness_dir <- normalizePath(dirname(sys.frame(1)$ofile %||%
                              "benchmarks/harness/run_all.R"), mustWork = FALSE)
if (!dir.exists(harness_dir)) {
  harness_dir <- file.path(getwd(), "benchmarks", "harness")
}

# Parse command-line args
args <- commandArgs(trailingOnly = TRUE)
single_suite <- NULL
include_gpu <- FALSE
for (i in seq_along(args)) {
  if (args[i] == "--suite" && i < length(args)) single_suite <- args[i + 1]
  if (args[i] == "--gpu") include_gpu <- TRUE
}

# Result output directory
results_dir <- file.path(harness_dir, "results", "current")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

# ============================================================================
# Helpers
# ============================================================================

#' Collect system metadata for reproducibility
collect_metadata <- function() {
  git_commit <- tryCatch(
    trimws(system("git rev-parse --short HEAD", intern = TRUE)),
    error = function(e) "unknown"
  )
  hostname <- Sys.info()[["nodename"]]
  cpu_info <- tryCatch(
    trimws(system("lscpu | grep 'Model name' | sed 's/.*:\\s*//'", intern = TRUE)),
    error = function(e) "unknown"
  )
  gpu_info <- tryCatch(
    trimws(system("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1",
                  intern = TRUE)),
    error = function(e) NULL
  )
  if (length(gpu_info) == 0 || nchar(gpu_info) == 0) gpu_info <- NULL

  list(
    git_commit = git_commit,
    timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
    node = hostname,
    cpu = cpu_info,
    gpu = gpu_info,
    omp_threads = as.integer(Sys.getenv("OMP_NUM_THREADS", "1")),
    r_version = paste0(R.version$major, ".", R.version$minor),
    rcppml_version = tryCatch(
      as.character(packageVersion("RcppML")),
      error = function(e) "unknown"
    )
  )
}

#' Time a single benchmark replicate
bench_time <- function(fn, n_reps = 5) {
  # Warm-up run (not counted)
  tryCatch(fn(), error = function(e) NULL)

  times <- numeric(n_reps)
  results <- vector("list", n_reps)
  for (i in seq_len(n_reps)) {
    gc(verbose = FALSE)
    t0 <- proc.time()["elapsed"]
    results[[i]] <- tryCatch(fn(), error = function(e) e)
    t1 <- proc.time()["elapsed"]
    times[i] <- t1 - t0
  }

  list(
    mean_sec = mean(times),
    sd_sec = sd(times),
    min_sec = min(times),
    max_sec = max(times),
    times = times,
    result = results[[1]]
  )
}

#' Save benchmark results as RDS (YAML-convertible structure)
save_results <- function(suite_name, results, metadata, results_dir) {
  out <- list(metadata = metadata, results = results)
  outfile <- file.path(results_dir, paste0(suite_name, ".rds"))
  saveRDS(out, outfile)
  cat(sprintf("  Saved: %s (%d results)\n", outfile, length(results)))
}

#' Print a summary table of results
print_summary <- function(results) {
  if (length(results) == 0) return(invisible(NULL))
  cat(sprintf("  %-45s %8s %8s\n", "Name", "Mean(s)", "SD(s)"))
  cat(paste(rep("-", 65), collapse = ""), "\n")
  for (r in results) {
    cat(sprintf("  %-45s %8.3f %8.4f\n", r$name, r$mean_sec, r$sd_sec))
  }
}

# ============================================================================
# Define suite order
# ============================================================================

cpu_suites <- c(
  "nmf_cpu_baseline",
  "svd_methods",
  "nnls_crossover",
  "nmf_distributions",
  "nmf_cv",
  "nmf_streaming"
)

gpu_suites <- c(
  "nmf_gpu_baseline"
)

suites_to_run <- if (!is.null(single_suite)) {
  single_suite
} else if (include_gpu) {
  c(cpu_suites, gpu_suites)
} else {
  cpu_suites
}

# ============================================================================
# Load library
# ============================================================================

cat("=== RcppML Benchmark Harness ===\n")
cat(sprintf("Harness dir: %s\n", harness_dir))

suppressPackageStartupMessages({
  library(RcppML)
  library(Matrix)
})

metadata <- collect_metadata()
cat(sprintf("Node: %s | CPU: %s | GPU: %s\n",
            metadata$node, metadata$cpu,
            if (is.null(metadata$gpu)) "none" else metadata$gpu))
cat(sprintf("R %s | RcppML %s | git %s | OMP threads: %d\n",
            metadata$r_version, metadata$rcppml_version,
            metadata$git_commit, metadata$omp_threads))
cat(sprintf("Timestamp: %s\n\n", metadata$timestamp))

# ============================================================================
# Generate datasets if missing
# ============================================================================

dataset_dir <- file.path(harness_dir, "datasets")
sparse_file <- file.path(dataset_dir, "sparse_5k_2k.rds")
dense_file <- file.path(dataset_dir, "dense_1k_500.rds")

if (!file.exists(sparse_file) || !file.exists(dense_file)) {
  cat("Generating benchmark datasets...\n")
  source(file.path(dataset_dir, "generate.R"), local = TRUE)
  cat("Done.\n\n")
}

# Load datasets
datasets <- list(
  sparse_large = readRDS(sparse_file),
  dense_medium = readRDS(dense_file)
)

# ============================================================================
# Run suites
# ============================================================================

total_start <- proc.time()["elapsed"]
all_results <- list()

for (suite in suites_to_run) {
  suite_file <- file.path(harness_dir, "suites", paste0(suite, ".R"))
  if (!file.exists(suite_file)) {
    cat(sprintf("[SKIP] %s — file not found: %s\n", suite, suite_file))
    next
  }

  cat(sprintf("[SUITE] %s\n", suite))
  suite_start <- proc.time()["elapsed"]

  # Source the suite — it must define run_suite(datasets, metadata)
  suite_env <- new.env(parent = globalenv())
  suite_env$bench_time <- bench_time
  suite_env$datasets <- datasets
  suite_env$metadata <- metadata
  source(suite_file, local = suite_env)

  if (exists("run_suite", envir = suite_env)) {
    suite_results <- suite_env$run_suite(datasets, metadata)
    all_results[[suite]] <- suite_results
    save_results(suite, suite_results, metadata, results_dir)
    print_summary(suite_results)
  } else {
    cat("  WARNING: No run_suite() function defined\n")
  }

  suite_elapsed <- proc.time()["elapsed"] - suite_start
  cat(sprintf("  Suite time: %.1f sec\n\n", suite_elapsed))
}

total_elapsed <- proc.time()["elapsed"] - total_start
cat(sprintf("=== Total wall-clock: %.1f sec ===\n", total_elapsed))

# Save combined results
combined_file <- file.path(results_dir, "all_results.rds")
saveRDS(list(metadata = metadata, suites = all_results), combined_file)
cat(sprintf("Combined results: %s\n", combined_file))
