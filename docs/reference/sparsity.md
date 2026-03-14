# Compute the sparsity of each NMF factor

Compute the sparsity of each NMF factor

## Usage

``` r
sparsity(object, ...)

# S4 method for class 'nmf'
sparsity(object, ...)
```

## Arguments

- object:

  object of class `nmf`.

- ...:

  additional parameters

## Value

A `data.frame` with columns `factor`, `sparsity`, and `model` ("w" or
"h").

## Details

For [`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md) models,
the sparsity of each factor is computed and summarized or \\w\\ and
\\h\\ matrices. A long `data.frame` with columns `factor`, `sparsity`,
and `model` is returned.

## See also

[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md),
[`summary,nmf-method`](https://zdebruine.github.io/RcppML/reference/summary-nmf-method.md)

## Examples

``` r
# \donttest{
data <- simulateNMF(50, 30, k = 3, seed = 1)
model <- nmf(data$A, 3, seed = 1, maxit = 50)
sparsity(model)
#>   factor   sparsity model
#> 1   nmf1 0.04000000     w
#> 2   nmf2 0.06000000     w
#> 3   nmf3 0.08000000     w
#> 4   nmf1 0.06666667     h
#> 5   nmf2 0.06666667     h
#> 6   nmf3 0.06666667     h
# }
```
