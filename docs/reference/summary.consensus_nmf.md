# Summary for Consensus NMF

Summary for Consensus NMF

## Usage

``` r
# S3 method for class 'consensus_nmf'
summary(object, ...)
```

## Arguments

- object:

  consensus_nmf object

- ...:

  additional arguments (unused)

## Value

Invisibly returns the `consensus_nmf` object. Summary statistics are
printed to the console.

## See also

[`consensus_nmf`](https://zdebruine.github.io/RcppML/reference/consensus_nmf.md),
[`plot.consensus_nmf`](https://zdebruine.github.io/RcppML/reference/plot.consensus_nmf.md)

## Examples

``` r
# \donttest{
library(Matrix)
A <- rsparsematrix(50, 30, 0.3)
cons <- consensus_nmf(A, k = 3, reps = 5, seed = 42)
summary(cons)
#> Consensus NMF Results
#> =====================
#> Rank (k): 3 
#> Replicates: 5 
#> Samples: 50 
#> Cophenetic correlation: 0.8538 
#> 
#> Cluster sizes:
#> 
#>  1  2  3 
#> 21 14 15 
#> 
#> Consensus summary:
#>   Min: 0 
#>   Median: 0.4 
#>   Mean: 0.332 
#>   Max: 1 
# }
```
