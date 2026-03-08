#!/usr/bin/env Rscript
# Run all GPU tests on a GPU node in a single session
# Usage: ssh g051 "cd /mnt/home/debruinz/RcppML-2 && module load r/4.5.2 && Rscript tools/run_gpu_tests.R"

cat("=== GPU Test Suite ===\n")
cat("Working directory:", getwd(), "\n")

# Load from source to get latest code + all data (including pbmc3k)
devtools::load_all()
cat("RcppML loaded via load_all\n")
cat("GPU available:", gpu_available(force_recheck = TRUE), "\n")
cat("pbmc3k available:", !is.null(tryCatch(dim(pbmc3k), error = function(e) NULL)), "\n")

# Run all GPU test files
library(testthat)

gpu_tests <- list.files("tests/testthat", pattern = "test_gpu.*\\.R", full.names = TRUE)
cat("\nTest files:", paste(basename(gpu_tests), collapse = ", "), "\n\n")

results <- list()
for (f in gpu_tests) {
  cat("=== Running:", basename(f), "===\n")
  res <- tryCatch(
    test_file(f, reporter = "summary"),
    error = function(e) e
  )
  results[[basename(f)]] <- res
  cat("\n")
}

# Summary
cat("\n=== SUMMARY ===\n")
for (nm in names(results)) {
  r <- results[[nm]]
  if (inherits(r, "error")) {
    cat(nm, ": ERROR -", r$message, "\n")
  } else {
    df <- as.data.frame(r)
    cat(sprintf("%s : PASS=%d FAIL=%d SKIP=%d WARN=%d\n",
                nm, sum(df$passed), sum(df$failed), sum(df$skipped), sum(df$warning)))
  }
}

totP <- sum(sapply(results, function(r) if(inherits(r,"error")) 0 else sum(as.data.frame(r)$passed)))
totF <- sum(sapply(results, function(r) if(inherits(r,"error")) 0 else sum(as.data.frame(r)$failed)))
totS <- sum(sapply(results, function(r) if(inherits(r,"error")) 0 else sum(as.data.frame(r)$skipped)))
totE <- sum(sapply(results, function(r) inherits(r,"error")))
cat(sprintf("\nTOTAL: PASS=%d FAIL=%d SKIP=%d ERROR=%d\n", totP, totF, totS, totE))
