# Bipartite graph matching

Hungarian algorithm for matching samples in a bipartite graph from a
distance ("cost") matrix

## Usage

``` r
bipartiteMatch(x)
```

## Arguments

- x:

  symmetric matrix giving the cost of every possible pairing

## Value

List with elements `cost` (total matching cost) and `assignment`
(0-indexed column assignment for each row).

## Details

This implementation was adapted from RcppHungarian, an Rcpp wrapper for
the original C++ implementation by Cong Ma (2016).

## See also

[`align`](https://zdebruine.github.io/RcppML/reference/align.md),
[`consensus_nmf`](https://zdebruine.github.io/RcppML/reference/consensus_nmf.md)

## Examples

``` r
cost <- matrix(c(1, 0.5, 0.2,
                  0.5, 1, 0.3,
                  0.2, 0.3, 1), nrow = 3, byrow = TRUE)
result <- bipartiteMatch(cost)
result$cost
#> [1] 1
result$assignment
#> [1] 2 0 1
```
