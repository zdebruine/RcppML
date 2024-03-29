#' @title Non-negative matrix factorization
#'
#' @description High-performance NMF of the form \eqn{A = wdh} for large dense or sparse matrices, returns an object of class \code{nmf}.
#'
#' @details
#' This fast NMF implementation decomposes a matrix \eqn{A} into lower-rank non-negative matrices \eqn{w} and \eqn{h},
#' with columns of \eqn{w} and rows of \eqn{h} scaled to sum to 1 via multiplication by a diagonal, \eqn{d}: \deqn{A = wdh}
#'
#' The scaling diagonal ensures convex L1 regularization, consistent factor scalings regardless of random initialization, and model symmetry in factorizations of symmetric matrices.
#'
#' The factorization model is randomly initialized.  \eqn{w} and \eqn{h} are updated by alternating least squares.
#'
#' RcppML achieves high performance using the Eigen C++ linear algebra library, OpenMP parallelization, a dedicated Rcpp sparse matrix class, and fast sequential coordinate descent non-negative least squares initialized by Cholesky least squares solutions.
#'
#' Sparse optimization is automatically applied if the input matrix \code{A} is a sparse matrix (i.e. \code{Matrix::dgCMatrix}). There are also specialized back-ends for symmetric, rank-1, and rank-2 factorizations.
#'
#' L1 penalization can be used for increasing the sparsity of factors and assisting interpretability. Penalty values should range from 0 to 1, where 1 gives complete sparsity.
#'
#' Set \code{options(RcppML.verbose = TRUE)} to print model tolerances to the console after each iteration.
#'
#' Parallelization is applied with OpenMP using the number of threads in \code{getOption("RcppML.threads")} and set by \code{option(RcppML.threads = 0)}, for example. \code{0} corresponds to all threads, let OpenMP decide.
#'
#' @section Methods:
#' S4 methods available for the \code{nmf} class:
#' * \code{predict}: project an NMF model (or partial model) onto new samples
#' * \code{evaluate}: calculate mean squared error loss of an NMF model
#' * \code{summary}: \code{data.frame} giving \code{fractional}, \code{total}, or \code{mean} representation of factors in samples or features grouped by some criteria
#' * \code{align}: find an ordering of factors in one \code{nmf} model that best matches those in another \code{nmf} model
#' * \code{prod}: compute the dense approximation of input data
#' * \code{sparsity}: compute the sparsity of each factor in \eqn{w} and \eqn{h}
#' * \code{subset}: subset, reorder, select, or extract factors (same as `[`)
#' * generics such as \code{dim}, \code{dimnames}, \code{t}, \code{show}, \code{head}
#'
#' @param data dense or sparse matrix of features in rows and samples in columns. Prefer \code{matrix} or \code{Matrix::dgCMatrix}, respectively
#' @param k rank
#' @param tol tolerance of the fit
#' @param maxit maximum number of fitting iterations
#' @param L1 LASSO penalties in the range (0, 1], single value or array of length two for \code{c(w, h)}
#' @param L2 Ridge penalties greater than zero, single value or array of length two for \code{c(w, h)}
#' @param seed single initialization seed or array, or a matrix or list of matrices giving initial \code{w}. For multiple initializations, the model with least mean squared error is returned.
#' @param mask dense or sparse matrix of values in \code{data} to handle as missing. Prefer \code{Matrix::dgCMatrix}. Alternatively, specify "\code{zeros}" or "\code{NA}" to mask either all zeros or NA values.
#' @param ... development parameters
#' @return object of class \code{nmf}
#' @importFrom methods is
#' @references
#'
#' DeBruine, ZJ, Melcher, K, and Triche, TJ. (2021). "High-performance non-negative matrix factorization for large single-cell data." BioRXiv.
#'
#' Lin, X, and Boutros, PC (2020). "Optimization and expansion of non-negative matrix factorization." BMC Bioinformatics.
#'
#' Lee, D, and Seung, HS (1999). "Learning the parts of objects by non-negative matrix factorization." Nature.
#'
#' Franc, VC, Hlavac, VC, Navara, M. (2005). "Sequential Coordinate-Wise Algorithm for the Non-negative Least Squares Problem". Proc. Int'l Conf. Computer Analysis of Images and Patterns. Lecture Notes in Computer Science.
#'
#' @author Zach DeBruine
#'
#' @export
#' @seealso \code{\link{project}}, \code{\link{mse}}, \code{\link{nnls}}
#' @md
#' @examples
#' \dontrun{
#' # basic NMF
#' model <- nmf(rsparsematrix(1000, 100, 0.1), k = 10)
#'
#' # compare rank-2 NMF to second left vector in an SVD
#' data(iris)
#' A <- Matrix::as(as.matrix(iris[, 1:4]), "dgCMatrix")
#' nmf_model <- nmf(A, 2, tol = 1e-5)
#' bipartitioning_vector <- apply(nmf_model$w, 1, diff)
#' second_left_svd_vector <- base::svd(A, 2)$u[, 2]
#' abs(cor(bipartitioning_vector, second_left_svd_vector))
#'
#' # compare rank-1 NMF with first singular vector in an SVD
#' abs(cor(nmf(A, 1)$w[, 1], base::svd(A, 2)$u[, 1]))
#'
#' # symmetric NMF
#' A <- crossprod(rsparsematrix(100, 100, 0.02))
#' model <- nmf(A, 10, tol = 1e-5, maxit = 1000)
#' plot(model$w, t(model$h))
#' # see package vignette for more examples
#' }
nmf <- function(data, k, tol = 1e-4, maxit = 100, L1 = c(0, 0), L2 = c(0, 0), seed = NULL, mask = NULL, ...) {
  start_time <- Sys.time()
  # apply defaults to development parameters
  p <- list(...)
  defaults <- list("link_matrix_h" = new("dgCMatrix"), "link_h" = FALSE, "sort_model" = TRUE, "upper_bound" = 0)
  for (i in 1:length(defaults)) {
    if (is.null(p[[names(defaults)[[i]]]])) p[[names(defaults)[[i]]]] <- defaults[[i]]
  }

  if (length(L1) == 1) {
    L1 <- rep(L1, 2)
  } else if (length(L1) != 2) stop("'L1' must be an array of two values, the first for the penalty on 'w', the second for the penalty on 'h'")
  if (max(L1) >= 1 || min(L1) < 0) stop("L1 penalties must be strictly in the range [0,1)")

  if (length(L2) == 1) {
    L2 <- rep(L2, 2)
  } else if (length(L2) != 2) stop("'L2' must be an array of two values, the first for the penalty on 'w', the second for the penalty on 'h'")
  if (min(L2) < 0) stop("L2 penalties must be strictly >= 0")

  # get 'data' in either sparse or dense matrix format and look for NA's
  if (is(data, "sparseMatrix")) {
    if (class(data)[[1]] != "dgCMatrix") data <- as(data, "dgCMatrix")
    if (any(is.na(data@x))) {
      if (!is.null(mask) && mask != "NA") stop("data contains 'NA' values. Either remove these values or specify \"mask = 'NA'\"")
      warning("NA values were detected in the data. Setting \"mask = 'NA'\"")
      mask <- is.na(data)
    }
  } else if (canCoerce(data, "matrix")) {
    data <- as.matrix(data)
    if (!is.numeric(data)) data <- as.numeric(data)
    if (any(is.na(data))) {
      if (!is.null(mask) && mask != "NA") stop("data contains 'NA' values. Either remove these values or specify \"mask = 'NA'\"")
      warning("NA values were detected in the data. Setting \"mask = 'NA'\"")
      mask <- is.na(as(data, "dgCMatrix"))
    }
  } else {
    stop("'data' was not coercible to a matrix")
  }

  if (is.null(mask)) {
    mask_matrix <- new("dgCMatrix")
    mask_zeros <- FALSE
  } else if (class(mask)[[1]] == "character" && mask == "zeros") {
    mask_matrix <- new("dgCMatrix")
    mask_zeros <- TRUE
  } else {
    mask_zeros <- FALSE
    if (!canCoerce(mask, "dgCMatrix")) {
      if (canCoerce(mask, "matrix")) {
        mask <- as.matrix(matrix)
      } else {
        stop("could not coerce the value of 'mask' to a sparse pattern matrix (dgCMatrix)")
      }
    }
    mask_matrix <- as(mask, "dgCMatrix")
  }

  # randomly initialize "w", or check dimensions of provided initialization
  w_init <- list()
  if (is.matrix(seed)) seed <- list(seed)
  if (!is.null(seed)) {
    if (is.matrix(seed[[1]])) {
      for (i in 1:length(seed)) {
        if (ncol(seed[[i]]) == nrow(data) && nrow(seed[[i]]) == k) {
          w_init[[i]] <- seed[[i]]
        } else if (nrow(seed[[i]]) == nrow(data) && ncol(seed[[i]]) == k) {
          w_init[[i]] <- t(seed[[i]])
        } else {
          stop("dimensions of provided initial 'w' matrices in 'seed' were incompatible with dimensions of 'data' and/or 'k'")
        }
      }
    } else if (is.numeric(seed[[1]])) {
      for (i in 1:length(seed)) {
        # randomly select runif or rnorm
        set.seed(seed[[i]])
        if (i == 1 || sample(c(FALSE, TRUE), 1)) {
          # runif
          # randomly set lower and upper bounds from pre-determined array
          # surprisingly, different bounds can affect the best possible discoverable solution from the initialization
          set.seed(seed[[i]])
          bounds <- sample(list(c(0, 1), c(0, 2), c(1, 2), c(1, 10)), 1)[[1]]
          set.seed(seed[[i]])
          w_init[[i]] <- matrix(runif(k * nrow(data), min = bounds[1], max = bounds[2]), k, nrow(data))
        } else {
          # rnorm
          # use rnorm(mean = 2, sd = 1) which does about as well as any other parameter at finding the best solution
          # rnorm often can do better than runif, but on some datatypes it does not, so
          #   run runif for iteration 1 and then possibly rnorm in later iterations
          set.seed(seed[[i]])
          w_init[[i]] <- matrix(rnorm(k * nrow(data), mean = 2, sd = 1), k, nrow(data))
        }
      }
    }
  } else {
    w_init[[1]] <- matrix(runif(k * nrow(data)), k, nrow(data))
  }

  # call C++ routines
  if (class(data)[[1]] == "dgCMatrix") {
    model <- Rcpp_nmf_sparse(data, mask_matrix, tol, maxit, getOption("RcppML.verbose"), L1, L2, getOption("RcppML.threads"), w_init, as(p$link_matrix_h, "dgCMatrix"), mask_zeros, p$link_h, p$sort_model, p$upper_bound)
  } else {
    model <- Rcpp_nmf_dense(data, mask_matrix, tol, maxit, getOption("RcppML.verbose"), L1, L2, getOption("RcppML.threads"), w_init, as(p$link_matrix_h, "dgCMatrix"), mask_zeros, p$link_h, p$sort_model, p$upper_bound)
  }

  # add back dimnames
  colnames(model$w) <- rownames(model$h) <- paste0("nmf", 1:ncol(model$w))
  if (!is.null(rownames(data))) rownames(model$w) <- rownames(data)
  if (!is.null(colnames(data))) colnames(model$h) <- colnames(data)

  misc <- list("tol" = model$tol, "iter" = model$iter, "runtime" = difftime(Sys.time(), start_time, units = "secs"))
  if (model$mse != 0) misc$mse <- model$mse
  if (length(w_init) > 1) {
    misc$w_init <- w_init[[model$best_model + 1]]
  } else {
    misc$w_init <- w_init[[1]]
  }

  new("nmf", w = model$w, d = model$d, h = model$h, misc = misc)
}
