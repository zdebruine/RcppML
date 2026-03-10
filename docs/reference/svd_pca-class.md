# svd_pca S4 Class

S4 class for deflation SVD/PCA results.

Computes scores for new samples by projecting `newdata` onto the right
singular vectors. Equivalent to PCA "out-of-sample prediction".

## Usage

``` r
# S4 method for class 'svd_pca,ANY,ANY,ANY'
x[i]

# S4 method for class 'svd_pca'
head(x, n = getOption("digits"), ...)

# S4 method for class 'svd_pca'
show(object)

# S4 method for class 'svd_pca'
dim(x)

reconstruct(object, ...)

# S4 method for class 'svd_pca'
reconstruct(object, ...)

# S4 method for class 'svd_pca'
predict(object, newdata, ...)

variance_explained(object, ...)

# S4 method for class 'svd_pca'
variance_explained(object, ...)
```

## Arguments

- x:

  object of class `svd_pca`

- i:

  indices for subsetting factors

- n:

  number of rows/columns to show

- ...:

  Ignored.

- object:

  An `svd_pca` object

- newdata:

  A numeric matrix of new samples (rows = samples, columns = features).
  Must have the same number of features as the original data
  (`ncol(newdata) == nrow(object@v)`).

## Value

A subsetted `svd_pca` object containing only the selected factors.

Invisibly returns the `svd_pca` object `x`.

Invisibly returns the `svd_pca` object.

Integer vector of length 3: `c(m, n, k)` where m is the number of rows,
n the number of columns, and k the rank.

Dense matrix: \\U \cdot diag(d) \cdot V'\\ (plus row means if centered)

A numeric matrix of scores with `nrow(newdata)` rows and
`length(object@d)` columns (i.e., same \\k\\ as the model). Each row is
the projection of one new sample onto the singular space.

Numeric vector of proportion of variance explained by each factor

## Slots

- `u`:

  left singular vectors (scores), m x k matrix

- `d`:

  singular values, numeric vector of length k

- `v`:

  right singular vectors (loadings), n x k matrix

- `misc`:

  list containing metadata: centered, row_means, test_loss,
  iters_per_factor, wall_time_ms, auto_rank

## Examples

``` r
# \donttest{
A <- matrix(rnorm(200), 20, 10)
s <- RcppML::svd(A, k = 3)
Ahat <- reconstruct(s)
dim(Ahat)
#> [1] 20 10
# }
# \donttest{
A <- matrix(rnorm(200), 20, 10)
s <- RcppML::svd(A, k = 3)
variance_explained(s)
#> [1] 0.2509394 0.2046180 0.1462364
# }
```
