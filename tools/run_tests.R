#!/usr/bin/env Rscript
# Quick install + test
cat("=== Installing package ===\n")
devtools::install(quick = TRUE, upgrade = "never", args = "--no-lock")
cat("\n=== Running tests ===\n")
res <- devtools::test()
cat("\n=== SUMMARY ===\n")
cat("PASS:", sum(as.data.frame(res)$passed), "\n")
cat("FAIL:", sum(as.data.frame(res)$failed), "\n")
cat("SKIP:", sum(as.data.frame(res)$skipped), "\n")
cat("WARN:", sum(as.data.frame(res)$warning), "\n")
cat("=== DONE ===\n")
