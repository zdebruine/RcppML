# PCA (centered SVD)

Convenience wrapper around
[`svd`](https://zdebruine.github.io/RcppML/reference/svd.md) with
`center = TRUE`.

## Usage

``` r
pca(A, k = 10, ...)
```

## Arguments

- A:

  Input matrix. May be dense (`matrix`), sparse (`dgCMatrix`), or a path
  to a `.spz` file for out-of-core streaming SVD.

- k:

  Number of factors (rank). Use `"auto"` for automatic rank selection
  via cross-validation. Default: 10.

- ...:

  Additional arguments passed to
  [`svd`](https://zdebruine.github.io/RcppML/reference/svd.md).

## Value

An S4 object of class `svd_pca` (see
[`svd`](https://zdebruine.github.io/RcppML/reference/svd.md)).

## See also

[`svd`](https://zdebruine.github.io/RcppML/reference/svd.md),
[`sparse_pca`](https://zdebruine.github.io/RcppML/reference/svd.md),
[`nn_pca`](https://zdebruine.github.io/RcppML/reference/svd.md)

## Examples

``` r
# \donttest{
library(Matrix)
data(aml)
result <- pca(aml, k = 5)
result
#> 824x5 rank-5 PCA model of class "svd_pca"
#>   sigma range: [10.55, 29.15]
#>   wall time: 41.7 ms
# }
```
