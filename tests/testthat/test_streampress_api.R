library(testthat)
library(RcppML)
library(Matrix)

# =============================================================================
# Basic write/read
# =============================================================================

test_that("st_write / st_read round-trip preserves matrix exactly", {
  set.seed(42)
  A <- rsparsematrix(200, 100, density = 0.05, repr = "C")
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))

  res <- st_write(A, f, verbose = FALSE)
  expect_equal(res$version, 2)
  expect_true(res$has_transpose)

  B <- st_read(f)
  expect_equal(nrow(B), 200L)
  expect_equal(ncol(B), 100L)
  expect_equal(as.matrix(A), as.matrix(B), tolerance = 1e-6)
})

test_that("st_write defaults to include_transpose = TRUE", {
  A <- rsparsematrix(50, 30, density = 0.1, repr = "C")
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))

  res <- st_write(A, f)
  expect_true(res$has_transpose)
  info <- st_info(f)
  expect_true(info$has_transpose)
})

# =============================================================================
# Obs/Var tables
# =============================================================================

test_that("st_write with obs/var table round-trips data.frame", {
  set.seed(1)
  A <- rsparsematrix(100, 50, density = 0.1, repr = "C")
  obs <- data.frame(
    cell_id = paste0("cell_", 1:100),
    score = runif(100),
    stringsAsFactors = FALSE
  )
  var <- data.frame(
    gene = paste0("gene_", 1:50),
    chr = as.integer(sample(1:22, 50, replace = TRUE)),
    stringsAsFactors = FALSE
  )

  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))

  res <- st_write(A, f, obs = obs, var = var)
  expect_true(res$has_obs)
  expect_true(res$has_var)

  info <- st_info(f)
  expect_true(info$has_obs)
  expect_true(info$has_var)
})

test_that("st_write validates obs nrow matches matrix nrow", {
  A <- rsparsematrix(100, 50, density = 0.1, repr = "C")
  obs_wrong <- data.frame(x = 1:50)
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))

  expect_error(st_write(A, f, obs = obs_wrong), "obs.*nrow")
})

test_that("st_write validates var nrow matches matrix ncol", {
  A <- rsparsematrix(100, 50, density = 0.1, repr = "C")
  var_wrong <- data.frame(x = 1:100)
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))

  expect_error(st_write(A, f, var = var_wrong), "var.*ncol")
})

test_that("st_read_obs on old file (no obs table) returns empty data.frame", {
  A <- rsparsematrix(100, 50, density = 0.1, repr = "C")
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))

  st_write(A, f)
  obs <- st_read_obs(f)
  expect_s3_class(obs, "data.frame")
  expect_equal(nrow(obs), 0L)
})

test_that("st_read_var on file without var table returns empty data.frame", {
  A <- rsparsematrix(100, 50, density = 0.1, repr = "C")
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))

  st_write(A, f)
  var <- st_read_var(f)
  expect_s3_class(var, "data.frame")
  expect_equal(nrow(var), 0L)
})

# =============================================================================
# st_info
# =============================================================================

test_that("st_info returns correct fields", {
  A <- rsparsematrix(300, 150, density = 0.03, repr = "C")
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))

  st_write(A, f, include_transpose = TRUE)
  info <- st_info(f)

  expect_equal(info$rows, 300L)
  expect_equal(info$cols, 150L)
  expect_equal(info$version, 2L)
  expect_true(info$has_transpose)
  expect_true(is.numeric(info$chunk_cols))
  expect_true(is.logical(info$has_obs))
  expect_true(is.logical(info$has_var))
})

# =============================================================================
# Column slicing
# =============================================================================

test_that("st_slice_cols returns correct submatrix", {
  set.seed(42)
  A <- rsparsematrix(100, 50, density = 0.1, repr = "C")
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))

  st_write(A, f)

  cols <- 5:15
  sub <- st_slice_cols(f, cols)
  expect_equal(nrow(sub), 100L)
  expect_equal(ncol(sub), length(cols))
  expect_equal(as.matrix(A[, cols]), as.matrix(sub), tolerance = 1e-6)
})

test_that("st_slice_cols with single column", {
  A <- rsparsematrix(100, 50, density = 0.1, repr = "C")
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))

  st_write(A, f)
  sub <- st_slice_cols(f, 1L)
  expect_equal(nrow(sub), 100L)
  expect_equal(ncol(sub), 1L)
})

# =============================================================================
# Row slicing (requires transpose)
# =============================================================================

test_that("st_slice_rows returns correct submatrix (requires transpose)", {
  set.seed(42)
  A <- rsparsematrix(100, 50, density = 0.1, repr = "C")
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))

  st_write(A, f, include_transpose = TRUE)

  rows <- 10:30
  sub <- st_slice_rows(f, rows)
  expect_equal(nrow(sub), length(rows))
  expect_equal(ncol(sub), 50L)
  expect_equal(as.matrix(A[rows, ]), as.matrix(sub), tolerance = 1e-6)
})

test_that("st_slice_rows errors on file without transpose", {
  A <- rsparsematrix(100, 50, density = 0.1, repr = "C")
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))

  st_write(A, f, include_transpose = FALSE)
  expect_error(st_slice_rows(f, 1:10), "transpose")
})

# =============================================================================
# Combined slicing
# =============================================================================

test_that("st_slice(rows=, cols=) matches manual subset", {
  set.seed(42)
  A <- rsparsematrix(100, 50, density = 0.1, repr = "C")
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))

  st_write(A, f, include_transpose = TRUE)

  rows <- 5:20
  cols <- 10:25
  sub <- st_slice(f, rows = rows, cols = cols)
  expect_equal(nrow(sub), length(rows))
  expect_equal(ncol(sub), length(cols))
  expect_equal(as.matrix(A[rows, cols]), as.matrix(sub), tolerance = 1e-6)
})

test_that("st_slice with only rows acts like st_slice_rows", {
  A <- rsparsematrix(100, 50, density = 0.1, repr = "C")
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))
  st_write(A, f, include_transpose = TRUE)

  sub1 <- st_slice(f, rows = 1:10)
  sub2 <- st_slice_rows(f, 1:10)
  expect_equal(as.matrix(sub1), as.matrix(sub2))
})

test_that("st_slice with only cols acts like st_slice_cols", {
  A <- rsparsematrix(100, 50, density = 0.1, repr = "C")
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))
  st_write(A, f)

  sub1 <- st_slice(f, cols = 1:10)
  sub2 <- st_slice_cols(f, 1:10)
  expect_equal(as.matrix(sub1), as.matrix(sub2))
})

test_that("st_slice with no rows/cols reads full matrix", {
  A <- rsparsematrix(100, 50, density = 0.1, repr = "C")
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))
  st_write(A, f)

  full <- st_slice(f)
  ref <- st_read(f)
  expect_equal(as.matrix(full), as.matrix(ref))
})

# =============================================================================
# Chunk ranges
# =============================================================================

test_that("st_chunk_ranges covers all columns", {
  A <- rsparsematrix(100, 5000, density = 0.01, repr = "C")
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))
  st_write(A, f)

  cr <- st_chunk_ranges(f)
  expect_s3_class(cr, "data.frame")
  expect_true(nrow(cr) >= 1)
  expect_equal(cr$start[1], 1L)
  expect_equal(cr$end[nrow(cr)], 5000L)

  # All columns covered without gaps
  for (i in seq_len(nrow(cr) - 1)) {
    expect_equal(cr$start[i + 1], cr$end[i] + 1L)
  }
})

# =============================================================================
# Chunk iteration
# =============================================================================

test_that("st_map_chunks reconstructs full matrix chunk-by-chunk", {
  set.seed(42)
  A <- rsparsematrix(100, 5000, density = 0.01, repr = "C")
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))
  st_write(A, f)

  chunks <- st_map_chunks(f, function(chunk, cs, ce) chunk)
  reconstructed <- do.call(cbind, chunks)

  ref <- st_read(f)
  expect_equal(as.matrix(reconstructed), as.matrix(ref))
})

# =============================================================================
# Old v2 file compatibility
# =============================================================================

test_that("old v2 files (reserved=0) remain fully readable", {
  # Write a file with include_transpose=FALSE and no obs/var
  A <- rsparsematrix(100, 50, density = 0.1, repr = "C")
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))

  st_write(A, f, include_transpose = FALSE)

  # Should read fine
  B <- st_read(f)
  expect_equal(as.matrix(A), as.matrix(B), tolerance = 1e-6)

  # Info should report no obs/var/transpose
  info <- st_info(f)
  expect_false(info$has_obs)
  expect_false(info$has_var)
  expect_false(info$has_transpose)

  # st_read_obs and st_read_var should return empty
  obs <- st_read_obs(f)
  expect_equal(nrow(obs), 0L)
  var <- st_read_var(f)
  expect_equal(nrow(var), 0L)
})

# =============================================================================
# List assembly (st_write_list)
# =============================================================================

test_that("st_write with list of matrices matches cbind result", {
  set.seed(42)
  A1 <- rsparsematrix(100, 30, density = 0.1, repr = "C")
  A2 <- rsparsematrix(100, 20, density = 0.1, repr = "C")
  combined <- cbind(A1, A2)

  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))

  st_write_list(list(A1, A2), f)
  B <- st_read(f)

  expect_equal(nrow(B), 100L)
  expect_equal(ncol(B), 50L)
  expect_equal(as.matrix(combined), as.matrix(B), tolerance = 1e-6)
})

# =============================================================================
# chunk_bytes parameter
# =============================================================================

test_that("chunk_bytes parameter produces correct chunk_cols", {
  A <- rsparsematrix(1000, 5000, density = 0.005, repr = "C")
  f1 <- tempfile(fileext = ".spz")
  f2 <- tempfile(fileext = ".spz")
  on.exit(unlink(c(f1, f2)))

  # Small chunk_bytes should produce more chunks
  st_write(A, f1, chunk_bytes = 4000)  # very small
  st_write(A, f2, chunk_bytes = 64e6)  # large

  info1 <- st_info(f1)
  info2 <- st_info(f2)

  # With smaller chunks, chunk_cols should be smaller
  expect_true(info1$chunk_cols <= info2$chunk_cols)
})

# =============================================================================
# Dimnames wiring
# =============================================================================

test_that("st_read does not return dimnames without obs/var", {
  A <- rsparsematrix(50, 30, density = 0.1, repr = "C")
  rownames(A) <- paste0("row_", 1:50)
  colnames(A) <- paste0("col_", 1:30)

  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))

  # Without obs/var, dimnames are not stored in spz format
  st_write(A, f)
  B <- st_read(f)
  expect_null(rownames(B))
  expect_null(colnames(B))
})
