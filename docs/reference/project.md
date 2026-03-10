# Project model onto data (DEPRECATED)

**Deprecated**. Use [`predict`](https://rdrr.io/r/stats/predict.html)
instead.

`project` has been replaced by
[`predict`](https://rdrr.io/r/stats/predict.html) (the S4 method for
class [`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md)).

## Usage

``` r
project(w, data, ...)
```

## Arguments

- w:

  feature factor matrix, or an `nmf` model object

- data:

  dense or sparse input matrix (features x samples)

- ...:

  additional arguments passed to
  [`predict`](https://rdrr.io/r/stats/predict.html) or
  [`nnls`](https://zdebruine.github.io/RcppML/reference/nnls.md)

## Value

matrix `h` of dimension `(k, ncol(data))`

## See also

[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md),
[`nnls`](https://zdebruine.github.io/RcppML/reference/nnls.md)
