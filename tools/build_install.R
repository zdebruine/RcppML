#!/usr/bin/env Rscript
# Build and install RcppML from source
cat("=== Building and Installing RcppML ===\n")
cat("Hostname:", system("hostname", intern=TRUE), "\n")
cat("Working dir:", getwd(), "\n")

# Remove old lock files
system("rm -rf /mnt/home/debruinz/R/x86_64-pc-linux-gnu-library/4.5/00LOCK-RcppML*")

# Document
cat("\n--- Running devtools::document() ---\n")
devtools::document()

# Fix Rcpp bug
cat("\n--- Fixing Rcpp info bug ---\n")
system("bash tools/fix_rcpp_info_bug.sh")

# Install
cat("\n--- Installing package ---\n")
devtools::install(quick = TRUE)

# Verify
cat("\n--- Verifying installation ---\n")
library(RcppML)
cat("RcppML loaded successfully!\n")
cat("Package version:", as.character(packageVersion("RcppML")), "\n")
