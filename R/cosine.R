#' Cosine similarity
#'
#' Column-by-column Euclidean norm cosine similarity for a matrix, pair of matrices, pair of vectors, or pair of a vector and matrix. Supports sparse matrices.
#'
#' This function takes advantage of extremely fast vector operations and is able to handle very large datasets.
#'
#' \code{cosine} applies a Euclidean norm to provide very similar results to Pearson correlation. Note that negative values may be returned due to the use of Euclidean normalization when all associations are largely random.
#'
#' This function adopts the sparse matrix computational strategy applied by \code{qlcMatrix::cosSparse}, and extends it to any combination of single and/or pair of sparse matrix and/or dense vector.
#' 
#' @param x matrix or vector of, or coercible to, class "dgCMatrix" or "sparseVector"
#' @param y (optional) matrix or vector of, or coercible to, class "dgCMatrix" or "sparseVector"
#' @returns dense matrix, vector, or value giving cosine distances
#' @export
#'
cosine <- function(x, y = NULL) {
  if (grepl("atrix", class(x)[1])) {
    colnames(x) <- rownames(x) <- NULL
    if (class(x)[1] != "dgCMatrix") x <- as(x, "dgCMatrix")
    if (!is.null(y)) {
      if (grepl("atrix", class(y)[1])) {
        # x is a matrix, y is a matrix
        colnames(y) <- rownames(y) <- NULL
        if (class(y)[1] != "dgCMatrix") y <- as(y, "dgCMatrix")
        res <- crossprod(tcrossprod(x, Diagonal(x = as.vector(crossprod(x ^ 2, rep(1, x@Dim[1]))) ^ -0.5)), tcrossprod(y, Diagonal(x = as.vector(crossprod(y ^ 2, rep(1, x@Dim[1]))) ^ -0.5)))
        return(as.matrix(res))
      } else {
        # x is a matrix, y is a vector
        return(as.vector(crossprod(tcrossprod(x, Diagonal(x = as.vector(crossprod(x ^ 2, rep(1, x@Dim[1]))) ^ -0.5)), tcrossprod(y, crossprod(y ^ 2, rep(1, x@Dim[1])) ^ -0.5))))
      }
    } else {
      # x is a matrix, y is NULL
      res <- crossprod(tcrossprod(x, Diagonal(x = as.vector(crossprod(x ^ 2, rep(1, x@Dim[1]))) ^ -0.5)))
      return(as.matrix(res))
    }
  } else {
    if (is.null(y)) stop("x is a vector and y is NULL")
    if (grepl("atrix", class(y)[1])) {
      # x is a vector, y is a matrix
      if (class(y)[1] != "dgCMatrix") y <- as(y, "dgCMatrix")
      colnames(y) <- rownames(y) <- NULL
      return(as.vector(crossprod(tcrossprod(x, crossprod(x ^ 2, rep(1, length(x)))), tcrossprod(y, Diagonal(x = as.vector(crossprod(y ^ 2, rep(1, length(x)))) ^ -0.5)))))
    } else {
      # x is a vector, y is a vector
      return(as.vector(crossprod(tcrossprod(x, crossprod(x ^ 2, rep(1, length(x))) ^ -0.5), tcrossprod(y, crossprod(y ^ 2, rep(1, length(x))) ^ -0.5))))
    }
  }
}