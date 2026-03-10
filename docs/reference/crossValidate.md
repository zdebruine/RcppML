# Cross-validate NMF (DEPRECATED)

**Deprecated**. Use
[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md) with
`test_fraction` instead.

`crossValidate` has been replaced by passing a vector of ranks and a
`test_fraction` to
[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md).

## Usage

``` r
crossValidate(A, k, ...)
```

## Arguments

- A:

  input data matrix (features x samples)

- k:

  integer vector of ranks to test

- ...:

  additional arguments passed to
  [`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md)

## Value

An `nmfCrossValidate` data frame

## See also

[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md)
