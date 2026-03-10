# Convert a dgCMatrix to SparsePress format (in-memory)

Convert a dgCMatrix to SparsePress format (in-memory)

## Usage

``` r
Rcpp_sp_compress(A, use_delta = TRUE, use_value_pred = FALSE)
```

## Arguments

- A:

  A sparse matrix (dgCMatrix)

- use_delta:

  Use delta prediction

- use_value_pred:

  Use value prediction

## Value

A raw vector containing the compressed data
