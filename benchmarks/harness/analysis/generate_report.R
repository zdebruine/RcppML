#!/usr/bin/env Rscript
# generate_report.R — Generate markdown summary of benchmark results
#
# Usage:
#   Rscript benchmarks/harness/analysis/generate_report.R
#
# Reads results/current/ and produces a markdown table.

harness_dir <- normalizePath(file.path(dirname(sys.frame(1)$ofile %||%
                              "benchmarks/harness/analysis/generate_report.R"),
                              ".."), mustWork = FALSE)
if (!dir.exists(harness_dir)) {
  harness_dir <- file.path(getwd(), "benchmarks", "harness")
}

current_dir <- file.path(harness_dir, "results", "current")
combined_file <- file.path(current_dir, "all_results.rds")

if (!file.exists(combined_file)) {
  cat("No combined results found. Run benchmarks first.\n")
  quit(status = 1)
}

data <- readRDS(combined_file)
meta <- data$metadata

cat("# RcppML Benchmark Report\n\n")
cat(sprintf("- **Date**: %s\n", meta$timestamp))
cat(sprintf("- **Node**: %s\n", meta$node))
cat(sprintf("- **CPU**: %s\n", meta$cpu))
cat(sprintf("- **GPU**: %s\n", if (is.null(meta$gpu)) "none" else meta$gpu))
cat(sprintf("- **R**: %s | **RcppML**: %s\n", meta$r_version, meta$rcppml_version))
cat(sprintf("- **Git**: %s\n", meta$git_commit))
cat(sprintf("- **OMP threads**: %d\n\n", meta$omp_threads))

for (suite_name in names(data$suites)) {
  suite_results <- data$suites[[suite_name]]
  if (length(suite_results) == 0) next

  cat(sprintf("## %s\n\n", suite_name))
  cat("| Benchmark | Mean (s) | SD (s) | Min (s) | Max (s) |\n")
  cat("|-----------|----------|--------|---------|--------|\n")

  for (r in suite_results) {
    cat(sprintf("| %s | %.3f | %.4f | %.3f | %.3f |\n",
                r$name, r$mean_sec, r$sd_sec, r$min_sec, r$max_sec))
  }
  cat("\n")
}
