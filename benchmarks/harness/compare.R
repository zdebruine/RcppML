#!/usr/bin/env Rscript
# compare.R — Compare two CSV benchmark result files
#
# Usage:
#   Rscript benchmarks/harness/compare.R baseline.csv new.csv
#   Rscript benchmarks/harness/compare.R baseline.csv new.csv --threshold 0.10
#
# Reports per-label median time comparison.
# Flags regressions > threshold (default 5%) with WARNING.
# Flags improvements > threshold with CHECK (verify correctness).
# Exit code 1 if any regression detected.

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  cat("Usage: Rscript benchmarks/harness/compare.R <baseline.csv> <new.csv> [--threshold N]\n")
  quit(status = 1)
}

baseline_file <- args[1]
new_file <- args[2]
threshold <- 0.05

for (i in seq_along(args)) {
  if (args[i] == "--threshold" && i < length(args)) {
    threshold <- as.numeric(args[i + 1])
  }
}

if (!file.exists(baseline_file)) {
  cat(sprintf("Baseline file not found: %s\n", baseline_file))
  quit(status = 1)
}
if (!file.exists(new_file)) {
  cat(sprintf("New file not found: %s\n", new_file))
  quit(status = 1)
}

baseline <- read.csv(baseline_file, stringsAsFactors = FALSE)
current <- read.csv(new_file, stringsAsFactors = FALSE)

# Compute per-label medians
agg <- function(df) {
  labels <- unique(df$label)
  result <- data.frame(
    label = labels,
    median_ms = vapply(labels, function(l) median(df$time_ms[df$label == l]), numeric(1)),
    stringsAsFactors = FALSE
  )
  result
}

bl_agg <- agg(baseline)
cur_agg <- agg(current)

# Merge on label
merged <- merge(bl_agg, cur_agg, by = "label", suffixes = c("_baseline", "_current"))

if (nrow(merged) == 0) {
  cat("No common labels found between baseline and current results.\n")
  quit(status = 0)
}

# Compare
cat(sprintf("=== Benchmark Comparison (threshold: %.0f%%) ===\n\n", threshold * 100))
cat(sprintf("Baseline: %s\n", baseline_file))
cat(sprintf("Current:  %s\n\n", new_file))

cat(sprintf("%-50s %10s %10s %8s %8s\n",
            "Label", "Base(ms)", "New(ms)", "Change", "Status"))
cat(paste(rep("-", 90), collapse = ""), "\n")

regressions <- 0L
improvements <- 0L

for (i in seq_len(nrow(merged))) {
  r <- merged[i, ]
  pct <- (r$median_ms_current - r$median_ms_baseline) / r$median_ms_baseline

  status <- if (pct > threshold) {
    regressions <- regressions + 1L
    "WARNING"
  } else if (pct < -threshold) {
    improvements <- improvements + 1L
    "CHECK"
  } else {
    "OK"
  }

  cat(sprintf("%-50s %10.2f %10.2f %+7.1f%% %8s\n",
              r$label, r$median_ms_baseline, r$median_ms_current,
              pct * 100, status))
}

cat("\n")
cat(sprintf("Total: %d labels | %d regressions | %d improvements | %d unchanged\n",
            nrow(merged), regressions, improvements,
            nrow(merged) - regressions - improvements))

if (regressions > 0L) {
  cat(sprintf("\nWARNING: %d benchmark(s) regressed by >%.0f%%\n", regressions, threshold * 100))
  quit(status = 1)
} else {
  cat("\nNo regressions detected.\n")
  quit(status = 0)
}
