% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/nmf_methods.R
\name{evaluate}
\alias{evaluate}
\alias{evaluate,nmf-method}
\title{Evaluate an NMF model}
\usage{
evaluate(x, ...)

\S4method{evaluate}{nmf}(x, data, mask = NULL, missing_only = FALSE, ...)
}
\arguments{
\item{x}{fitted model, class \code{nmf}, generally the result of calling \code{nmf}, with models of equal dimensions as \code{data}}

\item{...}{development parameters}

\item{data}{dense or sparse matrix of features in rows and samples in columns. Prefer \code{matrix} or \code{Matrix::dgCMatrix}, respectively}

\item{mask}{dense or sparse matrix of values in \code{data} to handle as missing. Prefer \code{Matrix::dgCMatrix}. Alternatively, specify "\code{zeros}" or "\code{NA}" to mask either all zeros or NA values.}

\item{missing_only}{calculate mean squared error only for missing values specified as a matrix in \code{mask}}
}
\description{
Calculate mean squared error for an NMF model, accounting for any masking schemes requested during fitting.
}
