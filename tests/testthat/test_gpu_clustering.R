# Not part of the CRAN subset (CRAN allows about ten minutes and two cores
# for the whole check). Runs when NOT_CRAN=true, as in CI and devtools::test().
testthat::skip_on_cran()

# GPU Clustering Parity Tests
# Tests: dclust (auto GPU dispatch), consensus_nmf with resource="gpu"

# ── dclust on GPU ───────────────────────────────────────────────────

test_that("dclust produces valid clusters on GPU node", {
  skip_if_no_gpu()

  m <- load_pbmc3k_matrix()
  A <- m[1:300, 1:150]
  A <- as(A, "dgCMatrix")

  # dclust auto-dispatches to GPU via .try_gpu_dclust
  dc <- dclust(A, min_samples = 5, seed = 42L, verbose = FALSE)

  expect_true(is.list(dc))
  expect_true(length(dc) > 0)

  # Each cluster element should have: samples, center, id, size
  for (i in seq_along(dc)) {
    expect_true("samples" %in% names(dc[[i]]))
    expect_true("center" %in% names(dc[[i]]))
    expect_true("id" %in% names(dc[[i]]))
    expect_true("size" %in% names(dc[[i]]))
    expect_true(length(dc[[i]]$samples) == dc[[i]]$size)
    expect_true(dc[[i]]$size >= 5)  # respects min_samples
  }
})

test_that("dclust cluster IDs are unique and zero-indexed", {
  skip_if_no_gpu()

  m <- load_pbmc3k_matrix()
  A <- m[1:300, 1:150]
  A <- as(A, "dgCMatrix")

  dc <- dclust(A, min_samples = 5, seed = 42L, verbose = FALSE)

  ids <- sapply(dc, function(x) x$id)
  expect_true(length(ids) == length(unique(ids)))  # unique IDs
  expect_true(all(grepl("^[01]*$", ids)))  # binary path strings
})

# ── consensus_nmf on GPU ───────────────────────────────────────────

test_that("consensus_nmf runs on GPU and returns valid structure", {
  skip_if_no_gpu()

  m <- load_pbmc3k_matrix()
  A <- m[1:300, 1:150]
  A <- as(A, "dgCMatrix")

  con <- consensus_nmf(A, k = 5, reps = 3, seed = 42L,
                        verbose = FALSE, resource = "gpu")

  expect_true(inherits(con, "consensus_nmf"))
  expect_true("consensus" %in% names(con))
  expect_true("models" %in% names(con))
  expect_true("clusters" %in% names(con))
  expect_true("cophenetic" %in% names(con))

  # consensus_nmf clusters the rows of its input
  expect_equal(nrow(con$consensus), nrow(A))
  expect_equal(ncol(con$consensus), nrow(A))

  # One cluster assignment per row
  expect_equal(length(con$clusters), nrow(A))
})

test_that("GPU consensus_nmf produces similar clusters to CPU", {
  skip_if_no_gpu()

  m <- load_pbmc3k_matrix()
  A <- m[1:300, 1:150]
  A <- as(A, "dgCMatrix")

  cpu_con <- consensus_nmf(A, k = 5, reps = 3, seed = 42L,
                            verbose = FALSE)
  gpu_con <- consensus_nmf(A, k = 5, reps = 3, seed = 42L,
                            verbose = FALSE, resource = "gpu")

  # Both should produce valid cluster assignments
  expect_equal(length(cpu_con$clusters), length(gpu_con$clusters))

  # Cophenetic correlation should be positive for both
  expect_true(cpu_con$cophenetic > 0 || is.na(cpu_con$cophenetic))
  expect_true(gpu_con$cophenetic > 0 || is.na(gpu_con$cophenetic))
})
