#!/usr/bin/env Rscript
# regression_check.R — Compare current benchmark results against baseline
#
# Usage:
#   Rscript benchmarks/harness/analysis/regression_check.R
#   Rscript benchmarks/harness/analysis/regression_check.R --threshold 0.10
#
# Flags any benchmark where mean time increased by > threshold (default 5%).
# Exit code 1 if any regression detected, 0 otherwise.

harness_dir <- normalizePath(file.path(dirname(sys.frame(1)$ofile %||%
                              "benchmarks/harness/analysis/regression_check.R"),
                              ".."), mustWork = FALSE)
if (!dir.exists(harness_dir)) {
  harness_dir <- file.path(getwd(), "benchmarks", "harness")
}

# Parse args
args <- commandArgs(trailingOnly = TRUE)
threshold <- 0.05
for (i in seq_along(args)) {
  if (args[i] == "--threshold" && i < length(args)) {
    threshold <- as.numeric(args[i + 1])
  }
}

# Load baseline and current results
baseline_dir <- file.path(harness_dir, "results", "baseline")
current_dir <- file.path(harness_dir, "results", "current")

if (!dir.exists(baseline_dir) || length(list.files(baseline_dir, "*.rds")) == 0) {
  cat("No baseline results found. Run benchmarks and freeze baseline first.\n")
  cat("  1. Rscript benchmarks/harness/run_all.R\n")
  cat("  2. Copy results/current/ to results/baseline/\n")
  quit(status = 0)
}

if (!dir.exists(current_dir) || length(list.files(current_dir, "*.rds")) == 0) {
  cat("No current results found. Run benchmarks first:\n")
  cat("  Rscript benchmarks/harness/run_all.R\n")
  quit(status = 0)
}

# Load results
load_suite_results <- function(dir) {
  files <- list.files(dir, pattern = "\\.rds$", full.names = TRUE)
  files <- files[!grepl("all_results\\.rds$", files)]
  all_results <- list()
  for (f in files) {
    data <- readRDS(f)
    for (r in data$results) {
      all_results[[r$name]] <- r
    }
  }
  all_results
}

baseline <- load_suite_results(baseline_dir)
current <- load_suite_results(current_dir)

# Compare
cat(sprintf("Regression threshold: %.0f%%\n", threshold * 100))
cat(sprintf("Baseline benchmarks: %d | Current benchmarks: %d\n\n",
            length(baseline), length(current)))

regressions <- list()
improvements <- list()

common_names <- intersect(names(baseline), names(current))
if (length(common_names) == 0) {
  cat("No common benchmarks found between baseline and current.\n")
  quit(status = 0)
}

cat(sprintf("%-45s %10s %10s %8s %6s\n",
            "Benchmark", "Baseline", "Current", "Change", "Status"))
cat(paste(rep("-", 85), collapse = ""), "\n")

for (name in sort(common_names)) {
  b <- baseline[[name]]
  c <- current[[name]]
  pct_change <- (c$mean_sec - b$mean_sec) / b$mean_sec

  status <- if (pct_change > threshold) {
    regressions[[name]] <- list(baseline = b$mean_sec, current = c$mean_sec, pct = pct_change)
    "REGRESS"
  } else if (pct_change < -threshold) {
    improvements[[name]] <- list(baseline = b$mean_sec, current = c$mean_sec, pct = pct_change)
    "FASTER"
  } else {
    "OK"
  }

  cat(sprintf("%-45s %9.3fs %9.3fs %+7.1f%% %6s\n",
              name, b$mean_sec, c$mean_sec, pct_change * 100, status))
}

cat("\n")

if (length(regressions) > 0) {
  cat(sprintf("REGRESSIONS DETECTED: %d benchmark(s) slower by > %.0f%%\n",
              length(regressions), threshold * 100))
  for (name in names(regressions)) {
    r <- regressions[[name]]
    cat(sprintf("  %s: %.3f -> %.3f (+%.1f%%)\n",
                name, r$baseline, r$current, r$pct * 100))
  }
  quit(status = 1)
} else {
  cat("No regressions detected.\n")
  if (length(improvements) > 0) {
    cat(sprintf("Improvements: %d benchmark(s) faster by > %.0f%%\n",
                length(improvements), threshold * 100))
  }
  quit(status = 0)
}
