# Write a dgCMatrix to a SparsePress (.spz) file

Write a dgCMatrix to a SparsePress (.spz) file

## Usage

``` r
Rcpp_sp_write(
  A,
  path,
  use_delta = TRUE,
  use_value_pred = FALSE,
  verbose = FALSE,
  precision = "auto",
  row_sort = FALSE,
  include_transpose = FALSE,
  chunk_cols = 2048L,
  obs_raw = NULL,
  var_raw = NULL
)
```

## Arguments

- A:

  A sparse matrix (dgCMatrix)

- path:

  Output file path (.spz)

- use_delta:

  Use delta prediction (better for structured data)

- use_value_pred:

  Use value prediction (for integer data)

- verbose:

  Print compression stats

- precision:

  Value precision: "auto", "fp32", "fp16", "quant8", "fp64"

- row_sort:

  Sort rows by nnz for better compression

- include_transpose:

  Also store CSC(A^T) in file

## Value

A list with compression statistics
