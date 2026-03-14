# Read a StreamPress file into a dgCMatrix

Read a StreamPress file into a dgCMatrix

## Usage

``` r
st_read(path, cols = NULL, reorder = TRUE, threads = NULL)
```

## Arguments

- path:

  Path to a `.spz` file.

- cols:

  Optional integer vector of column indices to read (1-indexed).

- reorder:

  Logical; if `TRUE` (default), undo any row permutation.

- threads:

  Integer or NULL; threads for parallel decompression. `NULL` (default)
  enables automatic selection: 1 thread for files smaller than 50 MB
  (where threading overhead exceeds the benefit), and all available
  threads for larger files. Use `0` to always request all available
  threads, or a positive integer to fix the count.

## Value

A `dgCMatrix` sparse matrix with dimnames if available.

## See also

[`st_write`](https://zdebruine.github.io/RcppML/reference/st_write.md),
[`st_info`](https://zdebruine.github.io/RcppML/reference/st_info.md)

## Examples

``` r
# \donttest{
library(Matrix)
A <- rsparsematrix(100, 50, 0.1)
f <- tempfile(fileext = ".spz")
st_write(A, f)
B <- st_read(f)
all.equal(A, B)
#> [1] "Mean relative difference: 2.357902e-08"
unlink(f)
# }
```
