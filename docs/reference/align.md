# Align two NMF models

Align two NMF models

## Usage

``` r
align(object, ...)

# S4 method for class 'nmf'
align(object, ref, method = "cosine", ...)
```

## Arguments

- object:

  nmf model to be aligned to `ref`

- ...:

  arguments passed to or from other methods

- ref:

  reference nmf model to which `object` will be aligned

- method:

  either `cosine` or `cor`

## Value

An `nmf` object with factors reordered to best match `ref`.

## Details

For [`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md) models,
factors in `object` are reordered to minimize the cost of bipartite
matching (see
[`bipartiteMatch`](https://zdebruine.github.io/RcppML/reference/bipartiteMatch.md))
on a [`cosine`](https://zdebruine.github.io/RcppML/reference/cosine.md)
or correlation distance matrix. The \\w\\ matrix is used for matching,
and must be equidimensional in `object` and `ref`.

## See also

[`bipartiteMatch`](https://zdebruine.github.io/RcppML/reference/bipartiteMatch.md),
[`cosine`](https://zdebruine.github.io/RcppML/reference/cosine.md),
[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md)

## Examples

``` r
# \donttest{
data <- simulateNMF(50, 30, k = 3, seed = 1)
m1 <- nmf(data$A, 3, seed = 1, maxit = 50)
m2 <- nmf(data$A, 3, seed = 2, maxit = 50)
aligned <- align(m2, m1)  # reorder m2 factors to match m1
# }
```
