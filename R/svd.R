svd <- function(data, k, tol = 1e-4, maxit = 100, L1 = c(0, 0), L2 = c(0, 0), seed = NULL, mask = NULL, ...) {
  start_time <- Sys.time()
  # apply defaults to development parameters
  p <- list(...)
  defaults <- list("link_matrix_v" = new("dgCMatrix"), "link_v" = FALSE, "sort_model" = TRUE, "upper_bound" = 0)
  for (i in 1:length(defaults)) {
    if (is.null(p[[names(defaults)[[i]]]])) p[[names(defaults)[[i]]]] <- defaults[[i]]
  }

  if (length(L1) == 1) {
    L1 <- rep(L1, 2)
  } else if (length(L1) != 2) stop("'L1' must be an array of two values, the first for the penalty on 'u', the second for the penalty on 'v'")
  if (max(L1) >= 1 || min(L1) < 0) stop("L1 penalties must be strictly in the range [0,1)")

  if (length(L2) == 1) {
    L2 <- rep(L2, 2)
  } else if (length(L2) != 2) stop("'L2' must be an array of two values, the first for the penalty on 'u', the second for the penalty on 'v'")
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

  # randomly initialize "u", or check dimensions of provided initialization
  u_init <- list()
  if (is.matrix(seed)) seed <- list(seed)
  if (!is.null(seed)) {
    if (is.matrix(seed[[1]])) {
      for (i in 1:length(seed)) {
        if (ncol(seed[[i]]) == nrow(data) && nrow(seed[[i]]) == k) {
          u_init[[i]] <- seed[[i]]
        } else if (nrow(seed[[i]]) == nrow(data) && ncol(seed[[i]]) == k) {
          u_init[[i]] <- t(seed[[i]])
        } else {
          stop("dimensions of provided initial 'u' matrices in 'seed' were incompatible with dimensions of 'data' and/or 'k'")
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
          u_init[[i]] <- matrix(runif(k * nrow(data), min = bounds[1], max = bounds[2]), nrow(data), k)
        } else {
          # rnorm
          # use rnorm(mean = 2, sd = 1) which does about as well as any other parameter at finding the best solution
          # rnorm often can do better than runif, but on some datatypes it does not, so
          #   run runif for iteration 1 and then possibly rnorm in later iterations
          set.seed(seed[[i]])
          u_init[[i]] <- matrix(rnorm(k * nrow(data), mean = 2, sd = 1), nrow(data), k)
        }
      }
    }
  } else {
    u_init[[1]] <- matrix(runif(k * nrow(data)), nrow(data), k)
  }

  # call C++ routines
  model <- Rcpp_svd_dense(data, mask_matrix, tol, maxit, getOption("RcppML.verbose"), L1, L2, getOption("RcppML.threads"), u_init, as(p$link_matrix_v, "dgCMatrix"), mask_zeros, p$link_v, p$upper_bound)
#   if (class(data)[[1]] == "dgCMatrix") {
#     model <- Rcpp_svd_sparse(data, mask_matrix, tol, maxit, getOption("RcppML.verbose"), L1, L2, getOption("RcppML.threads"), u_init, as(p$link_matrix_v, "dgCMatrix"), mask_zeros, p$link_v, p$sort_model, p$upper_bound)
#   } else {
#     model <- Rcpp_svd_dense(data, mask_matrix, tol, maxit, getOption("RcppML.verbose"), L1, L2, getOption("RcppML.threads"), u_init, as(p$link_matrix_v, "dgCMatrix"), mask_zeros, p$link_v, p$sort_model, p$upper_bound)
#   }

  # add back dimnames
  # colnames(model$u) <- rownames(model$v) <- paste0("svd", 1:ncol(model$u))
  # if (!is.null(rownames(data))) rownames(model$u) <- rownames(data)
  # if (!is.null(colnames(data))) colnames(model$v) <- colnames(data)

  misc <- list("tol" = model$tol, "iter" = model$iter, "runtime" = difftime(Sys.time(), start_time, units = "secs"))
  if (model$mse != 0) misc$mse <- model$mse
  if (length(u_init) > 1) {
    misc$u_init <- u_init[[model$best_model + 1]]
  } else {
    misc$u_init <- u_init[[1]]
  }

  new("svd", u = model$u, v = model$v, d = model$d, misc = misc)
}
