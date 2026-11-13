# StreamPress (.spz) reader: round trips and behavior on damaged files.
#
# A truncated or corrupted file must give an R error (or, when a damaged byte
# happens to decode, a structurally valid matrix). It must never crash or read
# out of bounds; CRAN runs these under ASAN/UBSAN.

library(Matrix)

skip_if(.Platform$endian == "big", "StreamPress files are little-endian")

.spz_counts <- function(m = 60L, n = 40L, seed = 11L) {
  set.seed(seed)
  A <- Matrix::rsparsematrix(m, n, density = 0.2,
                             rand.x = function(k) as.numeric(rpois(k, 3) + 1))
  as(A, "CsparseMatrix")
}

.read_bytes <- function(path) readBin(path, "raw", n = file.size(path))

.write_bytes <- function(bytes, path) {
  con <- file(path, "wb")
  on.exit(close(con))
  writeBin(bytes, con)
}

# Zero the CRC32 recorded in the 16-byte footer (bytes 5..8 of the footer).
# A zero CRC means "not recorded", so the reader skips the checksum and the
# structural validation is what has to catch the damage.
.zero_crc <- function(bytes) {
  n <- length(bytes)
  bytes[(n - 16L + 5L):(n - 16L + 8L)] <- as.raw(0)
  bytes
}

# Overwrite `len` bytes at 0-based file offset `off` with `value`.
.poke <- function(bytes, off, value, len = 4L) {
  bytes[(off + 1L):(off + len)] <- as.raw(value)
  bytes
}

.expect_no_crash <- function(path, m, n) {
  res <- tryCatch(st_read(path), error = function(e) NULL)
  if (is.null(res)) {
    succeed()
    return(invisible(NULL))
  }
  expect_s4_class(res, "dgCMatrix")
  # A damaged header may legitimately decode to other dimensions; the
  # result only has to be consistent with the dimensions it reports.
  m <- res@Dim[1]; n <- res@Dim[2]
  expect_equal(length(res@p), n + 1L)
  expect_false(is.unsorted(res@p))
  expect_equal(length(res@i), length(res@x))
  expect_equal(res@p[n + 1L], length(res@i))
  if (length(res@i) > 0) {
    expect_true(all(res@i >= 0L & res@i < m))
  }
  invisible(NULL)
}

# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------

test_that("TEST-REF-SPZ-ROUNDTRIP-V2: sparse v2 file reads back exactly", {
  A <- .spz_counts()
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))
  st_write(A, f, include_transpose = FALSE)
  B <- st_read(f)
  expect_s4_class(B, "dgCMatrix")
  expect_equal(dim(B), dim(A))
  expect_identical(B@p, A@p)
  expect_identical(B@i, A@i)
  expect_equal(B@x, A@x)
})

test_that("TEST-REF-SPZ-ROUNDTRIP-V2-CHUNKS: several chunks and the stored transpose", {
  # Dimnames are not part of the round trip: st_write() does not store them
  # (see "st_read does not return dimnames without obs/var" in
  # test_streampress_api.R), so the matrix is written without names.
  A <- .spz_counts()
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))
  st_write(A, f, include_transpose = TRUE, chunk_cols = 7L, transp_chunk_cols = 9L)
  B <- st_read(f)
  expect_identical(B@p, A@p)
  expect_identical(B@i, A@i)
  expect_equal(B@x, A@x)

  At <- st_read_transpose(f)
  expect_equal(dim(At), rev(dim(A)))
  expect_equal(as.matrix(At), unname(as.matrix(t(A))), ignore_attr = TRUE)
})

test_that("TEST-REF-SPZ-ROUNDTRIP-EMPTY: all-zero columns survive", {
  A <- .spz_counts()
  A[, c(1, 17, 40)] <- 0
  A <- drop0(A)
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))
  st_write(A, f, include_transpose = FALSE, chunk_cols = 8L)
  B <- st_read(f)
  expect_identical(B@p, A@p)
  expect_identical(B@i, A@i)
  expect_equal(B@x, A@x)
})

test_that("TEST-REF-SPZ-ROUNDTRIP-V1: in-memory v1 codec round trips", {
  A <- .spz_counts()
  raw1 <- RcppML:::Rcpp_sp_compress(A)
  B <- RcppML:::Rcpp_sp_decompress(raw1)
  expect_identical(B@p, A@p)
  expect_identical(B@i, A@i)
  expect_equal(B@x, A@x)
})

test_that("TEST-REF-SPZ-ROUNDTRIP-V3: dense v3 file reads back", {
  set.seed(3)
  X <- matrix(runif(30 * 12), 30, 12)
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))
  st_write_dense(X, f, chunk_cols = 5L)
  Y <- st_read_dense(f)
  expect_equal(dim(Y), dim(X))
  expect_equal(Y, X, tolerance = 1e-6, ignore_attr = TRUE)
})

# ---------------------------------------------------------------------------
# Version dispatch
# ---------------------------------------------------------------------------

test_that("TEST-REF-SPZ-DISPATCH: st_read() names the right reader for a dense file", {
  X <- matrix(runif(20), 5, 4)
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))
  st_write_dense(X, f)
  expect_error(st_read(f), "st_read_dense")
})

test_that("TEST-REF-SPZ-DISPATCH: unknown version and non-spz input are errors", {
  A <- .spz_counts()
  f <- tempfile(fileext = ".spz")
  g <- tempfile(fileext = ".spz")
  on.exit(unlink(c(f, g)))
  st_write(A, f, include_transpose = FALSE)
  b <- .read_bytes(f)

  .write_bytes(.poke(b, 4L, c(0x09, 0x00), len = 2L), g)   # version = 9
  expect_error(st_read(g), "version")

  .write_bytes(.poke(b, 0L, c(0x4e, 0x4f, 0x50, 0x45)), g)  # bad magic
  expect_error(st_read(g))

  .write_bytes(as.raw(c(1, 2, 3)), g)                       # far too short
  expect_error(st_read(g))
})

# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------

test_that("TEST-REF-SPZ-TRUNCATE-V2: every truncation of a v2 file is an error", {
  A <- .spz_counts()
  f <- tempfile(fileext = ".spz")
  g <- tempfile(fileext = ".spz")
  on.exit(unlink(c(f, g)))
  st_write(A, f, include_transpose = TRUE, chunk_cols = 7L)
  b <- .read_bytes(f)
  n <- length(b)
  cuts <- unique(c(0L, 5L, 6L, 64L, 127L, 128L, 129L, 200L,
                   n %/% 4L, n %/% 2L, (3L * n) %/% 4L,
                   n - 17L, n - 16L, n - 8L, n - 1L))
  cuts <- cuts[cuts >= 0L & cuts < n]
  for (k in cuts) {
    .write_bytes(b[seq_len(k)], g)
    expect_error(st_read(g), info = paste("truncated to", k, "bytes"))
    expect_error(st_read_transpose(g), info = paste("transpose, truncated to", k))
  }
})

test_that("TEST-REF-SPZ-TRUNCATE-V3: truncated dense file is an error", {
  set.seed(4)
  X <- matrix(runif(30 * 12), 30, 12)
  f <- tempfile(fileext = ".spz")
  g <- tempfile(fileext = ".spz")
  on.exit(unlink(c(f, g)))
  st_write_dense(X, f, chunk_cols = 5L)
  b <- .read_bytes(f)
  n <- length(b)
  for (k in c(7L, 100L, 128L, n %/% 2L, n - 16L, n - 1L)) {
    .write_bytes(b[seq_len(k)], g)
    expect_error(st_read_dense(g), info = paste("truncated to", k, "bytes"))
  }
})

test_that("TEST-REF-SPZ-TRUNCATE-V1: truncated v1 buffer is an error", {
  A <- .spz_counts()
  raw1 <- RcppML:::Rcpp_sp_compress(A)
  n <- length(raw1)
  for (k in c(0L, 10L, 71L, 72L, 73L, n %/% 2L, n - 1L)) {
    expect_error(RcppML:::Rcpp_sp_decompress(raw1[seq_len(k)]),
                 info = paste("truncated to", k, "bytes"))
  }
})

# ---------------------------------------------------------------------------
# Corruption caught by the checksum
# ---------------------------------------------------------------------------

test_that("TEST-REF-SPZ-CRC: a flipped byte anywhere before the footer is an error", {
  A <- .spz_counts()
  f <- tempfile(fileext = ".spz")
  g <- tempfile(fileext = ".spz")
  on.exit(unlink(c(f, g)))
  st_write(A, f, include_transpose = FALSE, chunk_cols = 7L)
  b <- .read_bytes(f)
  n <- length(b)
  set.seed(5)
  pos <- sort(unique(c(1L, 9L, 17L, 29L, 49L, 65L, 129L, 133L, 137L, 141L,
                       sample.int(n - 16L, 40L))))
  for (p in pos) {
    bb <- b
    bb[p] <- xor(bb[p], as.raw(0xff))
    .write_bytes(bb, g)
    expect_error(st_read(g), info = paste("flipped byte", p))
  }

  # v1 buffer and v3 file carry a checksum too
  raw1 <- RcppML:::Rcpp_sp_compress(A)
  r2 <- raw1
  r2[length(r2) %/% 2L] <- xor(r2[length(r2) %/% 2L], as.raw(0xff))
  expect_error(RcppML:::Rcpp_sp_decompress(r2))

  X <- matrix(runif(30 * 12), 30, 12)
  st_write_dense(X, f, chunk_cols = 5L)
  d <- .read_bytes(f)
  d[200L] <- xor(d[200L], as.raw(0xff))
  .write_bytes(d, g)
  expect_error(st_read_dense(g))
})

# ---------------------------------------------------------------------------
# Corruption with the checksum disabled: structural validation must hold
# ---------------------------------------------------------------------------

test_that("TEST-REF-SPZ-FIELDS: implausible header and index fields are errors", {
  A <- .spz_counts()
  f <- tempfile(fileext = ".spz")
  g <- tempfile(fileext = ".spz")
  on.exit(unlink(c(f, g)))
  st_write(A, f, include_transpose = FALSE, chunk_cols = 7L)
  b <- .zero_crc(.read_bytes(f))

  # Sanity: the file still reads with the checksum zeroed
  .write_bytes(b, g)
  expect_equal(dim(st_read(g)), dim(A))

  ff <- as.raw(c(0xff, 0xff, 0xff, 0x7f))
  cases <- list(
    "rows = 1"                 = .poke(b, 8L,  c(1, 0, 0, 0)),
    "cols = 1"                 = .poke(b, 12L, c(1, 0, 0, 0)),
    "cols huge"                = .poke(b, 12L, ff),
    "nnz huge"                 = .poke(b, 16L, ff),
    "nnz = 0"                  = .poke(b, 16L, c(0, 0, 0, 0, 0, 0, 0, 0), len = 8L),
    "num_chunks huge"          = .poke(b, 28L, ff),
    "num_chunks = 1"           = .poke(b, 28L, c(1, 0, 0, 0)),
    "chunk index offset huge"  = .poke(b, 48L, rep(0xff, 8), len = 8L),
    "data offset huge"         = .poke(b, 64L, rep(0xff, 8), len = 8L),
    # First chunk descriptor starts at byte 128: col_start, num_cols, nnz,
    # stream_offset[2], stream_size[2]
    "chunk col_start"          = .poke(b, 128L, c(5, 0, 0, 0)),
    "chunk num_cols huge"      = .poke(b, 132L, ff),
    "chunk nnz huge"           = .poke(b, 136L, ff),
    "chunk gap offset huge"    = .poke(b, 140L, ff),
    "chunk value offset huge"  = .poke(b, 144L, ff),
    "chunk gap size huge"      = .poke(b, 148L, ff),
    "chunk value size huge"    = .poke(b, 152L, ff),
    "chunk gap size zero"      = .poke(b, 148L, c(0, 0, 0, 0))
  )
  for (nm in names(cases)) {
    .write_bytes(cases[[nm]], g)
    expect_error(st_read(g), info = nm)
  }

  # A column range outside the matrix is an error, not an out-of-bounds read
  expect_error(st_read(f, cols = 1000:1001))
})

test_that("TEST-REF-SPZ-FUZZ: random damage never crashes the reader", {
  A <- .spz_counts()
  m <- nrow(A); n <- ncol(A)
  f <- tempfile(fileext = ".spz")
  g <- tempfile(fileext = ".spz")
  on.exit(unlink(c(f, g)))
  st_write(A, f, include_transpose = TRUE, chunk_cols = 7L)
  b <- .zero_crc(.read_bytes(f))
  len <- length(b)

  set.seed(6)
  # Single-byte damage
  for (p in sample.int(len - 16L, 150L)) {
    bb <- b
    bb[p] <- as.raw(sample.int(256L, 1L) - 1L)
    .write_bytes(bb, g)
    .expect_no_crash(g, m, n)
    invisible(tryCatch(st_read_transpose(g), error = function(e) NULL))
  }
  # Runs of zeros and of 0xff
  for (i in seq_len(30L)) {
    p <- sample.int(len - 40L, 1L)
    bb <- b
    bb[p:(p + 7L)] <- as.raw(if (i %% 2L) 0x00 else 0xff)
    .write_bytes(bb, g)
    .expect_no_crash(g, m, n)
    invisible(tryCatch(st_read_transpose(g), error = function(e) NULL))
  }
})

test_that("TEST-REF-SPZ-FUZZ-V3: random damage never crashes the dense reader", {
  set.seed(7)
  X <- matrix(runif(30 * 12), 30, 12)
  f <- tempfile(fileext = ".spz")
  g <- tempfile(fileext = ".spz")
  on.exit(unlink(c(f, g)))
  for (codec in c("raw", "fp16", "fp32_rans")) {
    st_write_dense(X, f, chunk_cols = 5L, codec = codec)
    b <- .zero_crc(.read_bytes(f))
    len <- length(b)
    # Header fields that size allocations (rows at 8, cols at 12) are left
    # intact; everything else is fair game.
    for (p in sample(setdiff(seq_len(len - 16L), 9:16), 60L)) {
      bb <- b
      bb[p] <- as.raw(sample.int(256L, 1L) - 1L)
      .write_bytes(bb, g)
      res <- tryCatch(st_read_dense(g), error = function(e) NULL)
      expect_true(is.null(res) || is.matrix(res), info = paste(codec, "byte", p))
    }
  }
})
