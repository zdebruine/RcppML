# Get metadata from a StreamPress file

Reads only the header – no decompression is performed.

## Usage

``` r
st_info(path)
```

## Arguments

- path:

  Path to a `.spz` file.

## Value

A list with fields including `rows`, `cols`, `nnz`, `density_pct`,
`file_bytes`, `raw_bytes`, `ratio`, `version`, `has_obs`, `has_var`,
`has_transpose`, `transp_chunk_cols`.

## See also

[`st_read`](https://zdebruine.github.io/RcppML/reference/st_read.md),
[`st_write`](https://zdebruine.github.io/RcppML/reference/st_write.md)

## Examples

``` r
# \donttest{
library(Matrix)
A <- rsparsematrix(100, 50, 0.1)
f <- tempfile(fileext = ".spz")
st_write(A, f)
info <- st_info(f)
cat(sprintf("Matrix: %d x %d, nnz=%d\n", info$rows, info$cols, info$nnz))
#> Matrix: 100 x 50, nnz=500
unlink(f)
# }
```
