#!/usr/bin/env Rscript
# Quick test to verify sp_* deprecation doesn't break functionality
library(testthat)
library(RcppML)
library(Matrix)

cat("Testing sparsepress roundtrip...\n")
test_file("tests/testthat/test_spz_roundtrip_comprehensive.R", reporter = "summary")
cat("\nDone.\n")
