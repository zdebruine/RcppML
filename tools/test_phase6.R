#!/usr/bin/env Rscript
# Phase 6 test: auto-dispatch
library(RcppML)
library(Matrix)

cat("=== Phase 6 Test: Auto-dispatch ===\n")

# Test 1: RAM detection
ram_mb <- RcppML:::Rcpp_get_available_ram_mb()
cat("Available RAM:", round(ram_mb / 1024, 1), "GB\n")
stopifnot(ram_mb > 0)

# Test 2: .get_available_ram_bytes
ram_bytes <- RcppML:::.get_available_ram_bytes()
cat("Available RAM (bytes):", format(ram_bytes, big.mark=","), "\n")
stopifnot(ram_bytes > 0)

# Test 3: .get_available_vram_bytes (should return 0 on CPU node)
vram_bytes <- RcppML:::.get_available_vram_bytes()
cat("Available VRAM (bytes):", vram_bytes, "\n")

# Test 4: Create a small .spz file and test dispatch
A <- rsparsematrix(500, 200, 0.1)
f <- tempfile(fileext = ".spz")
st_write(A, f, include_transpose = TRUE)
cat("Wrote test .spz file:", f, "\n")

# Test 5: .st_dispatch on small file -> should be IN_CORE_CPU
mode_info <- RcppML:::.st_dispatch(f, k = 5)
cat("Dispatch mode:", mode_info$mode, "\n")
cat("Resource:", mode_info$resource, "\n")
cat("Streaming:", mode_info$streaming, "\n")
stopifnot(mode_info$mode == "IN_CORE_CPU")
stopifnot(mode_info$resource == "cpu")
stopifnot(mode_info$streaming == FALSE)

# Test 6: nmf() with .spz path should auto-dispatch
result <- nmf(f, k = 3, maxit = 5, verbose = TRUE)
cat("NMF result class:", class(result), "\n")
cat("W dims:", dim(result@w), "\n")
cat("H dims:", dim(result@h), "\n")

unlink(f)
cat("\n=== All Phase 6 tests passed! ===\n")
