# Test S4 methods for nmf class
options(RcppML.threads = 1)
options(RcppML.verbose = FALSE)

# Create a small fixture used across tests
make_nmf_model <- function() {
  library(Matrix)
  set.seed(42)
  A <- rsparsematrix(20, 15, 0.3)
  A@x <- abs(A@x)  # ensure non-negative
  nmf(A, k = 3, seed = 1, maxit = 50, tol = 1e-4, verbose = FALSE)
}

# ---------- [[ method ----------
test_that("[[ accesses misc elements by name", {
  model <- make_nmf_model()
  # misc should have at least 'tol' and 'iter'
  expect_true("tol" %in% names(model@misc))
  expect_equal(model[["tol"]], model@misc[["tol"]])
})

test_that("[[ accesses misc elements by numeric index", {
  model <- make_nmf_model()
  expect_equal(model[[1]], model@misc[[1]])
})

test_that("[[ errors on missing name", {
  model <- make_nmf_model()
  expect_error(model[["nonexistent_key"]], "no element")
})

test_that("[[ errors on out-of-bounds index", {
  model <- make_nmf_model()
  expect_error(model[[999]], "out of bounds")
})

# ---------- t() method ----------
test_that("t() transposes w and h", {
  model <- make_nmf_model()
  tmodel <- t(model)
  
  expect_equal(tmodel@w, t(model@h))
  expect_equal(tmodel@h, t(model@w))
  expect_equal(tmodel@d, model@d)
})

test_that("t() preserves misc", {
  model <- make_nmf_model()
  tmodel <- t(model)
  expect_identical(tmodel@misc, model@misc)
})

test_that("t(t(x)) recovers original", {
  model <- make_nmf_model()
  roundtrip <- t(t(model))
  expect_equal(roundtrip@w, model@w)
  expect_equal(roundtrip@h, model@h)
  expect_equal(roundtrip@d, model@d)
})

# ---------- sort() method ----------
test_that("sort() reorders by decreasing d", {
  model <- make_nmf_model()
  smodel <- sort(model)
  
  # d should be sorted in decreasing order
  expect_true(all(diff(smodel@d) <= 0))
  
  # dimensions preserved
  expect_equal(dim(smodel@w), dim(model@w))
  expect_equal(dim(smodel@h), dim(model@h))
})

test_that("sort(decreasing=FALSE) gives increasing order", {
  model <- make_nmf_model()
  smodel <- sort(model, decreasing = FALSE)
  
  expect_true(all(diff(smodel@d) >= 0))
})

# ---------- prod() method ----------
test_that("prod() reconstructs W * diag(d) * H", {
  model <- make_nmf_model()
  reconstructed <- prod(model)
  
  expected <- model@w %*% diag(model@d) %*% model@h
  expect_equal(as.matrix(reconstructed), as.matrix(expected), tolerance = 1e-10)
})

test_that("prod() returns correct dimensions", {
  model <- make_nmf_model()
  reconstructed <- prod(model)
  
  expect_equal(nrow(reconstructed), nrow(model@w))
  expect_equal(ncol(reconstructed), ncol(model@h))
})

# ---------- $ method ----------
test_that("$ accesses w, d, h slots", {
  model <- make_nmf_model()
  expect_equal(model$w, model@w)
  expect_equal(model$d, model@d)
  expect_equal(model$h, model@h)
})

test_that("$ accesses misc slot", {
  model <- make_nmf_model()
  expect_identical(model$misc, model@misc)
})

test_that("$ accesses elements within misc", {
  model <- make_nmf_model()
  if ("tol" %in% names(model@misc)) {
    expect_equal(model$tol, model@misc$tol)
  }
  if ("iter" %in% names(model@misc)) {
    expect_equal(model$iter, model@misc$iter)
  }
})

test_that("$ errors on unknown name", {
  model <- make_nmf_model()
  expect_error(model$nonexistent_slot, "not a slot")
})

# ---------- coerce to list ----------
test_that("as(nmf, 'list') returns list with w, d, h, misc", {
  model <- make_nmf_model()
  lst <- as(model, "list")
  
  expect_true(is.list(lst))
  expect_equal(names(lst), c("w", "d", "h", "misc"))
  expect_equal(lst$w, model@w)
  expect_equal(lst$d, model@d)
  expect_equal(lst$h, model@h)
  expect_identical(lst$misc, model@misc)
})

# ---------- subset / [ methods ----------
test_that("subset() selects specific factors", {
  model <- make_nmf_model()
  sub <- subset(model, 1:2)
  
  expect_equal(ncol(sub@w), 2)
  expect_equal(nrow(sub@h), 2)
  expect_equal(length(sub@d), 2)
})

test_that("[ selects specific factors", {
  model <- make_nmf_model()
  sub <- model[c(1, 3)]
  
  expect_equal(ncol(sub@w), 2)
  expect_equal(nrow(sub@h), 2)
  expect_equal(length(sub@d), 2)
  expect_equal(sub@w, model@w[, c(1, 3)])
  expect_equal(sub@h, model@h[c(1, 3), ])
})

# ---------- dim() method ----------
test_that("dim() returns m, n, k", {
  model <- make_nmf_model()
  d <- dim(model)
  
  expect_equal(d[1], nrow(model@w))   # m
  expect_equal(d[2], ncol(model@h))   # n
  expect_equal(d[3], length(model@d)) # k
})

# ---------- head() / show() method ----------
test_that("head() produces output without error", {
  model <- make_nmf_model()
  # head() prints info and returns invisible(x) 
  expect_output(head(model), "factor model of class \"nmf\"")
})

test_that("show() produces output without error", {
  model <- make_nmf_model()
  expect_output(show(model), "factor model of class \"nmf\"")
})

# ---------- sparsity() method ----------
test_that("sparsity() returns data.frame with correct structure", {
  model <- make_nmf_model()
  sp <- sparsity(model)
  
  expect_true(is.data.frame(sp))
  expect_true(all(c("factor", "sparsity", "model") %in% names(sp)))
  expect_true(all(sp$model %in% c("w", "h")))
  expect_true(all(sp$sparsity >= 0 & sp$sparsity <= 1))
})
