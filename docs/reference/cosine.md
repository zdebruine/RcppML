# Cosine similarity

Column-by-column Euclidean norm cosine similarity for a matrix, pair of
matrices, pair of vectors, or pair of a vector and matrix. Supports
sparse matrices.

## Usage

``` r
cosine(x, y = NULL)
```

## Arguments

- x:

  matrix or vector of, or coercible to, class "dgCMatrix" or
  "sparseVector"

- y:

  (optional) matrix or vector of, or coercible to, class "dgCMatrix" or
  "sparseVector"

## Value

dense matrix, vector, or value giving cosine distances

## Details

This function takes advantage of extremely fast vector operations and is
able to handle very large datasets.

`cosine` applies a Euclidean norm to provide very similar results to
Pearson correlation. Note that negative values may be returned due to
the use of Euclidean normalization when all associations are largely
random.

This function adopts the sparse matrix computational strategy applied by
`qlcMatrix::cosSparse`, and extends it to any combination of single
and/or pair of sparse matrix and/or dense vector.

## See also

[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md),
[`bipartiteMatch`](https://zdebruine.github.io/RcppML/reference/bipartiteMatch.md)

## Examples

``` r
x <- matrix(runif(20), 4, 5)
cosine(x)         # self-similarity: 5x5 matrix
#>           [,1]      [,2]      [,3]      [,4]      [,5]
#> [1,] 1.0000000 0.8777736 0.7470947 0.9657964 0.8113463
#> [2,] 0.8777736 1.0000000 0.9345580 0.9212170 0.9422615
#> [3,] 0.7470947 0.9345580 1.0000000 0.7500746 0.7806535
#> [4,] 0.9657964 0.9212170 0.7500746 1.0000000 0.9248622
#> [5,] 0.8113463 0.9422615 0.7806535 0.9248622 1.0000000
cosine(x, x[,1])  # similarity of columns to first column
#> [1] 1.0000000 0.8777736 0.7470947 0.9657964 0.8113463
```
