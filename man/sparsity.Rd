% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/nmf_methods.R
\name{sparsity}
\alias{sparsity}
\alias{sparsity,nmf-method}
\title{Compute the sparsity of each NMF factor}
\usage{
sparsity(object, ...)

\S4method{sparsity}{nmf}(object, ...)
}
\arguments{
\item{object}{object of class \code{nmf}.}

\item{...}{additional parameters}
}
\description{
Compute the sparsity of each NMF factor
}
\details{
For \code{\link{nmf}} models, the sparsity of each factor is computed and summarized
or \eqn{w} and \eqn{h} matrices. A long \code{data.frame} with columns \code{factor}, \code{sparsity}, and \code{model} is returned.
}
