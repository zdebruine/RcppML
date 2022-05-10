#' Random distributions and samples
#' 
#' @description \code{r_sample} is just like \code{base::sample}, only faster. 
#' \code{r_sample} takes a sample of the specified size from the elements of \code{x} using replacement if indicated.
#' 
#' @param x either a positive integer giving the number of items to choose from, or a vector of elements to shuffle or from which to choose. See 'Details'.
#' @param size an integer giving the number of items to choose. If \code{NULL}, \code{x} is shuffled and all values are kept.
#' @param replace should sampling be with replacement?
#' @rdname random
#' @export
#' @examples 
#' # draw all integers from 1 to 10 in a random order
#' sample(10)
#' 
#' # shuffle a vector of values
#' v <- r_unif(3)
#' v
#' v_ <- sample(v)
#' v_
#' 
#' # draw values from a vector
#' sample(r_unif(100), 3) 
#' 
#' # draw some integers between 1 and 1000
#' sample(1000, 3)
#' 
r_sample <- function(x, size = NULL, replace = FALSE){
  if(length(x) == 1 && x %% 1 == 0){
    if(is.null(size)) size <- x
    v <- c_sample(x, size, replace, .Random.seed[3], .Random.seed[4]) + 1
  } else {
    if(is.null(size)) size <- length(x)
    i <- c_sample(length(x), size, replace, .Random.seed[3], .Random.seed[4]) + 1
    v <- x[i]
  }
  set.seed(.Random.seed[4])
  v
}

#' Generate random distributions
#' 
#' @description These functions generate random distributions (uniform, normal, or binomial) just like their base R counterparts (\code{runif}, \code{rnorm}, and \code{rbinom}), but faster.
#' 
#' @details 
#' All RNGs make use of Marsaglia's xorshift method to generate random integers.
#' 
#' * \code{r_binom} takes the modulus of the random integer (inverse probability) and returns a count if the modulus is zero
#' * \code{r_unif} takes the random integer and divides it by the seed and returns the floating decimal portion of the result.
#' * \code{r_norm} maps the random uniform distribution to a Gaussian distribution using Marsaglia's polar method.
#'
#' @md
#' 
#' @param n number of observations
#' @param min finite lower limit of the uniform distribution
#' @param max finite upper limit of the uniform distribution
#' @seealso \code{\link{r_matrix}}, \code{\link{r_sparsematrix}}
#' @export
#' @rdname random
#' @examples
#' # simulate a uniform distribution
#' v <- r_unif(10000)
#' plot(density(v))
#' 
#' # simulate a normal distribution
#' v <- r_norm(10000)
#' plot(density(v))
#' 
#' # simulate a binomial distribution
#' v <- r_binom(10000, 100, inv_prob = 10)
#' hist(v)
#' sum(v) / length(v)
#' # ~10 because 100 trials at 10 percent success odds
#' #   is about 10 successes per element
#' 
#' # get successful trials in a bernoulli distribution
#' v <- r_binom(100, 1, 20)
#' successful_trials <- slot(as(v, "nsparseVector"), "i")
#' successful_trials
#' 
r_unif <- function(n, min = 0, max = 1){
  v <- c_runif(n, min, max, .Random.seed[3], .Random.seed[4])
  set.seed(.Random.seed[3])
  v
}

#' @rdname random
#' @param size number of trials (one or more)
#' @param inv_prob inverse probability of success for each trial, must be integral (e.g. 50 percent success = 2, 10 percent success = 10)
#' @export
r_binom <- function(n, size = 1, inv_prob = 2){
  s <- .Random.seed[3]
  v <- c_binom(n, size, inv_prob, s)
  set.seed(.Random.seed[3], .Random.seed[4])
  v
}

#' Random transpose-identical dense/sparse matrix
#' 
#' @description Generate a random sparse matrix, just like \code{Matrix::rsparsematrix}, but much faster. Generation of transpose-identical matrices is also supported without additional computational cost.
#'
#' @description Generate a matrix initialized with random uniform values, just like \code{(matrix(runif(nrow * ncol), nrow, ncol)}, but much faster. 
#' Generation of transpose-identical matrices is also supported without additional computational cost.
#'
#' @param nrow number of rows
#' @param ncol number of columns
#' @param t_identical should the matrix be transpose-identical?
#'
#' @seealso \code{\link{r_unif}}
#' @rdname random_matrix
#' @export
#' @seealso \code{\link{r_matrix}}, \code{\link{r_unif}}, \code{\link{r_binom}}
#' @examples 
#' # generate a simple random matrix
#' A <- r_matrix(10, 10)
#' 
#' # generate two matrices that are transpose identical
#' set.seed(123)
#' A1 <- r_matrix(10, 100, symmetric = TRUE)
#' set.seed(123)
#' A2 <- r_matrix(100, 10, symmetric = TRUE)
#' all.equal(t(A2), A1)
#' 
#' # generate a transpose-identical pair of speckled matrices
#' set.seed(123)
#' A <- r_sparsematrix(10, 100, inv_density = 10, symmetric = TRUE)
#' set.seed(123)
#' A <- r_sparsematrix(100, 10, inv_density = 10, symmetric = TRUE)
#' all.equal(t(A), A)
#' isSymmetric(A[1:10,1:10])
#' heatmap(as.matrix(A), scale = "none", Rowv = NA, Colv = NA)
#' 
#' # note that density is probabilistic, not absolute
#' A <- replicate(1000, r_sparsematrix(100, 100, 10))
#' densities <- sapply(A, function(x) length(x@i) / prod(dim(x)))
#' plot(density(densities)) # normal distribution centered at 0.100
#' range(densities)
#' 
r_matrix <- function(nrow, ncol, transpose_identical = FALSE){
  if(transpose_identical){
    v <- c_rtimatrix(nrow, ncol, .Random.seed[3])
  } else {
    v <- c_rmatrix(nrow, ncol, .Random.seed[3])
  }
  set.seed(.Random.seed[3])
  v
}

#' @rdname random_matrix
#' @param inv_density an integer giving the inverse density of the matrix (i.e. 10 percent density corresponds to \code{inv_density = 10}). Density is probabilistic, not exact. See examples.
#' @param pattern should a pattern matrix (\code{Matrix::ngCMatrix}) be returned? If not, a \code{Matrix::dgCMatrix} with random uniform values will be returned.
#' @export
r_sparsematrix <- function(nrow, ncol, inv_density, t_identical = FALSE, pattern = FALSE){
  require(Matrix)
  if(t_identical){
    v <- c_rtisparsematrix(nrow, ncol, inv_density, pattern, .Random.seed[3])
  } else {
    v <- c_rsparsematrix(nrow, ncol, inv_density, pattern, .Random.seed[3])
  }
  set.seed(.Random.seed[4])
  v
}
