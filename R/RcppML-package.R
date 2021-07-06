#' RcppML: Machine Learning in Rcpp
#'
#' @description
#' Constrained and regularized least squares and matrix factorization model projection for dense and sparse matrices.
#'
#' @details 
#' A library of machine learning methods in Rcpp being actively developed using a very fast active-set/coordinate descent constrained least squares algorithm with optional L0 regularization.
#'
#' @import knitr Matrix RcppEigen
#' @importFrom Rcpp evalCpp
#' @importFrom methods as canCoerce
#' @useDynLib RcppML, .registration = TRUE
#' @docType package
#' @name RcppML
#' @author Zach DeBruine
#' @aliases RcppML-package
#' @md
#'
"_PACKAGE"