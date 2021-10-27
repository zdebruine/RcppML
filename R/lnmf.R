#' @title Linked non-negative matrix factorization
#'
#' @description Run lNMF on a list of datasets to separate shared and unique signals.
#'
#' @details
#' Detailed documentation to come
#'
#' @inheritParams nmf
#' @param data list of dense or sparse matrices giving features in rows and samples in columns. Rows in all matrices must correspond exactly. Prefer \code{matrix} or \code{Matrix::dgCMatrix}, respectively
#' @param k_wh rank of the shared factorization
#' @param k_uv ranks of the unique factorizations, an array corresponding to each dataset provided
#' @param L1 LASSO penalties in the range (0, 1], single value or array of length two for \code{c(w & u, h & v)}
#' @param L2 Ridge penalties greater than zero, single value or array of length two for \code{c(w & u, h & v)}
#' @param seed single initialization seed or array of seeds. If multiple seeds are provided, the model with least mean squared error is returned.
#' @param mask list of dense or sparse matrices indicating values in \code{data} to handle as missing. Prefer \code{Matrix::ngCMatrix}. Alternatively, specify a string "\code{zeros}" or "\code{NA}" to mask either all zeros or NA values.
#' @return an object of class \code{lnmf}
#' @references
#' 
#' DeBruine, ZJ, Melcher, K, and Triche, TJ. (2021). "High-performance non-negative matrix factorization for large single-cell data." BioRXiv.
#' 
#' @author Zach DeBruine
#'
#' @export
#' @seealso \code{\link{nmf}}, \code{\link{predict.nmf}}, \code{\link{evaluate.nmf}}, \code{\link{nnls}}
#' @md
lnmf <- function(data, k_wh, k_uv, tol = 1e-4, maxit = 100, L1 = c(0, 0), L2 = c(0, 0), nonneg = TRUE, seed = NULL, mask = NULL) {

  if (length(k_uv) != length(data)) stop("number of ranks specified in 'k_uv' must equal the length of the list of datasets in 'data'")
  if (length(data) == 1) stop("only one dataset was provided, linked NMF is only useful for multiple datasets")

  # define initial "h", the linking matrix
  set_pointers <- c(sapply(data, function(x) ncol(x)))
  for (i in 2:length(set_pointers))
    set_pointers[i] <- set_pointers[i] + set_pointers[i - 1]
  set_pointers <- c(0, set_pointers)
  k_pointers <- c(k_wh, k_uv)
  for (i in 2:length(k_pointers))
    k_pointers[i] <- k_pointers[i] + k_pointers[i - 1]
  n_samples <- sum(sapply(data, function(x) ncol(x)))
  link_matrix <- matrix(0, sum(k_wh, k_uv), n_samples)
  link_matrix[1:k_wh, 1:n_samples] <- 1

  for (i in 1:length(k_uv))
    link_matrix[(k_pointers[i] + 1):k_pointers[i + 1], (set_pointers[i] + 1):set_pointers[i + 1]] <- 1

  # combine data into a single matrix
  if (!all(sapply(data, function(x) class(x)) == class(data[[1]]))) stop("'data' contains items of different classes")
  if (!all(sapply(data, function(x) nrow(x)) == nrow(data[[1]]))) stop("'data' contains items with different numbers of rows")
  data <- do.call(cbind, data)

  # combine all mask matrices into a single matrix and check dimensions against "A"
  if (!is.null(mask)) {
    if (is.list(mask)) {
      if (length(mask) != length(k_uv)) stop("'mask' was a list, but not of the same length as 'data' and 'k_uv'")
      if (class(mask)[[1]] == "character") {
        if (!all(sapply(mask, function(x) x == mask[[1]]))) stop("'mask' was a list of characters, but not all items were equal. Since datasets are factorized jointly, only one type of masking may be specified.")
        mask <- mask[[1]]
      } else {
        if (!all(sapply(mask, function(x) class(x)) == class(mask[[1]]))) stop("'mask' contains items of different classes")
        if (!all(sapply(mask, function(x) nrow(x)) == nrow(mask[[1]]))) stop("'mask' contains items with different numbers of rows")
        mask <- do.call(cbind, mask)
        if (!all(dim(mask) == dim(data))) stop("dimensions of all 'mask' and 'data' items are not equivalent")
      }
    }
  }

  link_matrix <- as(link_matrix, "ngCMatrix")

  model <- nmf(data, nrow(link_matrix), tol, maxit, L1, L2, nonneg, seed, mask, link_h = TRUE, link_matrix_h = link_matrix, sort_model = FALSE)
  diag_order_wh <- order(model@d[1:k_wh], decreasing = TRUE)
  w <- model@w[, diag_order_wh]
  rownames(model@h) <- paste0("h", 1:nrow(model@h))
  colnames(w) <- paste0("w", 1:ncol(w))
  u <- v <- h <- d_wh <- d_uv <- list()
  for (i in 1:length(k_uv)) {
    u[[i]] <- as.matrix(model@w[, (k_pointers[i] + 1):k_pointers[i + 1]])
    d_uv[[i]] <- model@d[(k_pointers[i] + 1):k_pointers[i + 1]]
    diag_order_uv <- order(d_uv[[i]], decreasing = TRUE)
    d_uv[[i]] <- d_uv[[i]][diag_order_uv]
    v[[i]] <- as.matrix(model@h[(k_pointers[i] + 1):k_pointers[i + 1], (set_pointers[i] + 1):set_pointers[i + 1]])
    if(ncol(v[[i]]) == 1) v[[i]] <- t(v[[i]])
    if(nrow(v[[i]]) > 1){
      v[[i]] <- v[[i]][diag_order_uv,]
      u[[i]] <- u[[i]][, diag_order_uv]
    }
    h[[i]] <- as.matrix(model@h[diag_order_wh, (set_pointers[i] + 1):set_pointers[i + 1]])
    if(ncol(h[[i]]) == 1) h[[i]] <- t(h[[i]])
    d_wh[[i]] <- model@d[diag_order_wh]
    scale_h <- rowSums(h[[i]])
    h[[i]] <- apply(h[[i]], 2, function(x) x / scale_h)
    d_wh[[i]] <- d_wh[[i]] * scale_h
    names(d_wh[[i]]) <- NULL
    colnames(u[[i]]) <- paste0("u", i, ".", 1:ncol(u[[i]]))
    rownames(v[[i]]) <- paste0("v", i, ".", 1:nrow(v[[i]]))
    rownames(u[[i]]) <- rownames(w)
    colnames(v[[i]]) <- colnames(h[[i]])
  }
  return(new("lnmf", w = w, u = u, v = v, h = h, d_wh = d_wh, d_uv = d_uv, misc = model@misc))
}