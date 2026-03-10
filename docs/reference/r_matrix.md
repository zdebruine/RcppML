# Random transpose-identical dense/sparse matrix

Generate a random sparse matrix, just like
[`Matrix::rsparsematrix`](https://rdrr.io/pkg/Matrix/man/rsparsematrix.html)
or `(matrix(runif(nrow * ncol), nrow,))`, but much faster. Generation of
transpose-identical matrices is also supported without additional
computational cost.

## Usage

``` r
r_matrix(nrow, ncol, transpose_identical = FALSE)

r_sparsematrix(
  nrow,
  ncol,
  inv_density,
  transpose_identical = FALSE,
  pattern = FALSE
)
```

## Arguments

- nrow:

  number of rows

- ncol:

  number of columns

- transpose_identical:

  should the matrix be transpose-identical?

- inv_density:

  an integer giving the inverse density of the matrix (i.e. 10 percent
  density corresponds to `inv_density = 10`). Density is probabilistic,
  not exact. See examples.

- pattern:

  should a pattern matrix (`Matrix::ngCMatrix`) be returned? If not, a
  `Matrix::dgCMatrix` with random uniform values will be returned.

## Value

A dense numeric matrix of dimensions `nrow` by `ncol` with random
uniform values.

A sparse matrix: `ngCMatrix` if `pattern = TRUE`, otherwise `dgCMatrix`.

## See also

[`r_unif`](https://zdebruine.github.io/RcppML/reference/random.md),
[`r_binom`](https://zdebruine.github.io/RcppML/reference/random.md)

## Examples

``` r
if (FALSE) { # \dontrun{
# generate a simple random matrix
A <- r_matrix(10, 10)

# generate two matrices that are transpose identical
set.seed(123)
A1 <- r_matrix(10, 100, transpose_identical = TRUE)
set.seed(123)
A2 <- r_matrix(100, 10, transpose_identical = TRUE)
all.equal(t(A2), A1)

# generate a transpose-identical pair of speckled matrices
set.seed(123)
A <- r_sparsematrix(10, 100, inv_density = 10, transpose_identical = TRUE)
set.seed(123)
A <- r_sparsematrix(100, 10, inv_density = 10, transpose_identical = TRUE)
all.equal(t(A), A)
Matrix::isSymmetric(A[1:10, 1:10])
heatmap(as.matrix(A), scale = "none", Rowv = NA, Colv = NA)

# note that density is probabilistic, not absolute
A <- replicate(1000, r_sparsematrix(100, 100, 10))
densities <- sapply(A, function(x) length(x@i) / prod(dim(x)))
plot(density(densities)) # normal distribution centered at 0.100
range(densities)
} # }
```
