library(testthat)
library(RcppML)
library(Matrix)

test_that("st_write/st_read round-trip: sparse matrix", {
  set.seed(42)
  A <- Matrix::rsparsematrix(200, 100, density = 0.05, repr = "C")
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))
  st_write(A, f)
  B <- st_read(f)
  expect_equal(as.matrix(A), as.matrix(B), tolerance = 1e-6)
})

test_that("st_info returns correct dimensions for v2 file", {
  A <- Matrix::rsparsematrix(300, 150, density = 0.03, repr = "C")
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))
  st_write(A, f)
  info <- st_info(f)
  expect_equal(info$rows, 300L)
  expect_equal(info$cols, 150L)
  expect_equal(info$version, 2L)
})

test_that("nmf from .spz file works end-to-end", {
  skip(".st_dispatch not yet re-implemented")
  A <- abs(Matrix::rsparsematrix(500, 200, density = 0.05, repr = "C"))
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))
  st_write(A, f, include_transpose = TRUE)
  res <- nmf(f, k = 4, maxit = 5, tol = 1e-3)
  expect_s4_class(res, "nmf")
  expect_equal(nrow(res@w), 500L)
  expect_equal(ncol(res@h), 200L)
})

test_that("chunk_cols parameter accepted by st_write", {
  A <- Matrix::rsparsematrix(100, 200, density = 0.1, repr = "C")
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))
  expect_no_error(st_write(A, f, chunk_cols = 64L))
  info <- st_info(f)
  expect_equal(info$rows, 100L)
})

test_that("st_add_transpose adds transpose successfully", {
  A <- Matrix::rsparsematrix(100, 80, density = 0.1, repr = "C")
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))
  st_write(A, f, include_transpose = FALSE)
  info_before <- st_info(f)
  expect_false(info_before$has_transpose)
  st_add_transpose(f)
  info_after <- st_info(f)
  expect_true(info_after$has_transpose)
})

test_that("st_write_dense round-trips correctly", {
  M <- matrix(rnorm(200), nrow = 20, ncol = 10)
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))
  st_write_dense(M, f)
  M2 <- st_read_dense(f)
  expect_equal(dim(M), dim(M2))
  # float32 precision
  expect_equal(M, M2, tolerance = 1e-6)
})

test_that("auto-dispatch selects IN_CORE_CPU for small files", {
  skip(".st_dispatch not yet re-implemented")
  A <- Matrix::rsparsematrix(50, 30, density = 0.1, repr = "C")
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))
  st_write(A, f)
  mode_info <- RcppML:::.st_dispatch(f, k = 3)
  expect_equal(mode_info$mode, "IN_CORE_CPU")
  expect_false(mode_info$streaming)
})

test_that("Rcpp_get_available_ram_mb returns positive value", {
  ram <- RcppML:::Rcpp_get_available_ram_mb()
  expect_gt(ram, 0)
})
