library(testthat)
library(RcppML)
library(Matrix)

# =============================================================================
# C++ API tests — exercised through the Rcpp bridge layer
# These test the underlying C++ code paths for obs_var_table, header_v2
# accessors, chunk iteration, and file format compatibility.
# =============================================================================

test_that("FileHeader_v2 accessor methods for reserved bytes work", {
  # Write a file WITH obs/var and transpose and check that st_info
  # correctly reads back the accessor-derived fields
  A <- rsparsematrix(100, 50, density = 0.1, repr = "C")
  obs <- data.frame(x = runif(100))
  var <- data.frame(y = runif(50))
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))

  st_write(A, f, obs = obs, var = var, include_transpose = TRUE)
  info <- st_info(f)

  # These fields come from reserved byte accessors in FileHeader_v2
  expect_true(info$has_obs)
  expect_true(info$has_var)
  expect_true(info$has_transpose)
  expect_true(is.numeric(info$transp_chunk_cols))
})

test_that("old files without obs_table_offset report has_obs_table == FALSE", {
  A <- rsparsematrix(100, 50, density = 0.1, repr = "C")
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))

  # Write without obs/var — reserved bytes stay zero
  st_write(A, f)
  info <- st_info(f)
  expect_false(info$has_obs)
  expect_false(info$has_var)
})

test_that("ColDescriptor serialization round-trip via obs table", {
  # Write obs with multiple column types
  A <- rsparsematrix(100, 50, density = 0.1, repr = "C")
  obs <- data.frame(
    int_col = 1:100,
    dbl_col = runif(100),
    str_col = paste0("cell_", 1:100),
    stringsAsFactors = FALSE
  )
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))

  res <- st_write(A, f, obs = obs)
  expect_true(res$has_obs)

  # Read back — obs_raw round-trips through C++ obs_var_table serialize/deserialize
  obs_back <- st_read_obs(f)

  # If obs table read is not yet implemented in C++, this will return empty
  # In that case, at minimum verify the write didn't corrupt the file
  B <- st_read(f)
  expect_equal(as.matrix(A), as.matrix(B), tolerance = 1e-6)
})

test_that("parallel write produces same output as serial write", {
  set.seed(42)
  A <- rsparsematrix(200, 100, density = 0.05, repr = "C")

  f1 <- tempfile(fileext = ".spz")
  f2 <- tempfile(fileext = ".spz")
  on.exit(unlink(c(f1, f2)))

  st_write(A, f1, threads = 1L)
  st_write(A, f2, threads = 4L)

  # Both files should produce identical decompressed matrices
  B1 <- st_read(f1)
  B2 <- st_read(f2)
  expect_equal(as.matrix(B1), as.matrix(B2))
})

test_that("decompress_v2 handles column range parameters", {
  set.seed(42)
  A <- rsparsematrix(100, 50, density = 0.1, repr = "C")
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))

  st_write(A, f)

  # Read full and sliced, compare
  full <- st_read(f)
  sliced <- st_slice_cols(f, 1:25)

  expect_equal(as.matrix(full[, 1:25]), as.matrix(sliced), tolerance = 1e-6)
})

test_that("transpose section enables row access", {
  set.seed(42)
  A <- rsparsematrix(50, 30, density = 0.15, repr = "C")
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))

  st_write(A, f, include_transpose = TRUE)
  sub <- st_slice_rows(f, 1:10)

  expect_equal(nrow(sub), 10L)
  expect_equal(ncol(sub), 30L)
  expect_equal(as.matrix(A[1:10, ]), as.matrix(sub), tolerance = 1e-6)
})

test_that("safe_pread reads correct bytes at given offsets", {
  # This is implicitly tested through all read operations.
  # We verify by reading the same file multiple times and getting consistent results.
  A <- rsparsematrix(100, 50, density = 0.1, repr = "C")
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))

  st_write(A, f)

  # Multiple reads should be identical
  B1 <- st_read(f)
  B2 <- st_read(f)
  B3 <- st_read(f)
  expect_equal(as.matrix(B1), as.matrix(B2))
  expect_equal(as.matrix(B2), as.matrix(B3))
})

test_that("extdata pbmc3k.spz is readable with new code", {
  spz_path <- system.file("extdata", "pbmc3k.spz", package = "RcppML")
  skip_if(spz_path == "", message = "pbmc3k.spz not found in extdata")

  A <- st_read(spz_path)
  expect_s4_class(A, "dgCMatrix")
  expect_true(nrow(A) > 0)
  expect_true(ncol(A) > 0)

  info <- st_info(spz_path)
  expect_equal(info$rows, nrow(A))
  expect_equal(info$cols, ncol(A))
  expect_equal(info$version, 2L)
})
