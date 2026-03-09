test_that("nmf() accepts .rds file path (sparse)", {
  skip_on_cran()
  skip_if_not_installed("Matrix")
  data(movielens, package = "RcppML")
  tmp <- tempfile(fileext = ".rds")
  on.exit(unlink(tmp), add = TRUE)
  saveRDS(movielens, tmp)
  
  result <- nmf(tmp, k = 3, tol = 1e-2, maxit = 10, seed = 42, verbose = FALSE)
  expect_s4_class(result, "nmf")
  expect_equal(nrow(result@w), nrow(movielens))
  expect_equal(ncol(result@h), ncol(movielens))
})

test_that("nmf() accepts .rds file path (dense)", {
  set.seed(1)
  A <- matrix(abs(rnorm(60)), nrow = 6, ncol = 10)
  tmp <- tempfile(fileext = ".rds")
  on.exit(unlink(tmp), add = TRUE)
  saveRDS(A, tmp)
  
  result <- nmf(tmp, k = 2, tol = 1e-2, maxit = 10, seed = 42, verbose = FALSE)
  expect_s4_class(result, "nmf")
  expect_equal(nrow(result@w), 6)
  expect_equal(ncol(result@h), 10)
})

test_that("nmf() accepts .mtx file path", {
  skip_if_not_installed("Matrix")
  data(movielens, package = "RcppML")
  tmp <- tempfile(fileext = ".mtx")
  on.exit(unlink(tmp), add = TRUE)
  Matrix::writeMM(movielens, tmp)
  
  result <- nmf(tmp, k = 3, tol = 1e-2, maxit = 10, seed = 42, verbose = FALSE)
  expect_s4_class(result, "nmf")
  expect_equal(nrow(result@w), nrow(movielens))
  expect_equal(ncol(result@h), ncol(movielens))
})

test_that("nmf() accepts .spz file path", {
  skip(".st_dispatch not yet re-implemented")
  skip_if_not_installed("Matrix")
  data(movielens, package = "RcppML")
  tmp <- tempfile(fileext = ".spz")
  on.exit(unlink(tmp), add = TRUE)
  sp_write(movielens, tmp, include_transpose = TRUE)
  
  result <- nmf(tmp, k = 3, tol = 1e-2, maxit = 10, seed = 42, verbose = FALSE)
  expect_s4_class(result, "nmf")
  expect_equal(nrow(result@w), nrow(movielens))
  expect_equal(ncol(result@h), ncol(movielens))
})

test_that("nmf() accepts .csv file path", {
  set.seed(1)
  A <- matrix(abs(rnorm(60)), nrow = 6, ncol = 10)
  rownames(A) <- paste0("gene", 1:6)
  colnames(A) <- paste0("sample", 1:10)
  tmp <- tempfile(fileext = ".csv")
  on.exit(unlink(tmp), add = TRUE)
  utils::write.csv(A, tmp)
  
  result <- nmf(tmp, k = 2, tol = 1e-2, maxit = 10, seed = 42, verbose = FALSE)
  expect_s4_class(result, "nmf")
  expect_equal(nrow(result@w), 6)
  expect_equal(ncol(result@h), 10)
})

test_that("validate_data errors on non-existent file", {
  expect_error(RcppML:::validate_data("nonexistent_file.rds"), "File not found")
})

test_that("validate_data errors on unsupported extension", {
  tmp <- tempfile(fileext = ".xyz")
  file.create(tmp)
  on.exit(unlink(tmp), add = TRUE)
  expect_error(RcppML:::validate_data(tmp), "Unsupported file format")
})

test_that(".load_sparse_file round-trips through .rds", {
  skip_if_not_installed("Matrix")
  data(movielens, package = "RcppML")
  tmp <- tempfile(fileext = ".rds")
  on.exit(unlink(tmp), add = TRUE)
  saveRDS(movielens, tmp)
  
  loaded <- RcppML:::.load_sparse_file(tmp)
  expect_true(inherits(loaded, "dgCMatrix"))
  expect_equal(dim(loaded), dim(movielens))
  expect_equal(Matrix::nnzero(loaded), Matrix::nnzero(movielens))
})

test_that(".load_sparse_file preserves data through .mtx", {
  skip_if_not_installed("Matrix")
  # Create a simple sparse matrix
  A <- Matrix::sparseMatrix(
    i = c(1, 2, 3, 1, 3),
    j = c(1, 2, 3, 3, 1),
    x = c(1.5, 2.5, 3.5, 4.5, 5.5),
    dims = c(3, 3)
  )
  tmp <- tempfile(fileext = ".mtx")
  on.exit(unlink(tmp), add = TRUE)
  Matrix::writeMM(A, tmp)
  
  loaded <- RcppML:::.load_sparse_file(tmp)
  expect_true(inherits(loaded, "dgCMatrix"))
  expect_equal(dim(loaded), dim(A))
  expect_equal(sort(loaded@x), sort(A@x))
})

test_that(".load_sparse_file handles .tsv", {
  set.seed(2)
  A <- matrix(abs(rnorm(20)), nrow = 4, ncol = 5)
  rownames(A) <- paste0("r", 1:4)
  colnames(A) <- paste0("c", 1:5)
  tmp <- tempfile(fileext = ".tsv")
  on.exit(unlink(tmp), add = TRUE)
  utils::write.table(A, tmp, sep = "\t", quote = FALSE)
  
  loaded <- RcppML:::.load_sparse_file(tmp)
  expect_true(is.matrix(loaded))
  expect_equal(dim(loaded), c(4, 5))
})
