# SparsePress I/O (Deprecated)

These functions are deprecated. Use the `st_*` equivalents from the
[`streampress`](https://zdebruine.github.io/RcppML/reference/streampress.md)
API instead.

## Usage

``` r
sp_write(
  x,
  path,
  delta = TRUE,
  value_pred = FALSE,
  verbose = FALSE,
  precision = "auto",
  row_sort = FALSE,
  include_transpose = FALSE,
  chunk_cols = 2048L
)

sp_read(path, cols = NULL, reorder = TRUE)

sp_read_transpose(path)

sp_info(path)

sp_compress(x, delta = TRUE, value_pred = FALSE)

sp_decompress(x)

sp_write_dense(
  x,
  path,
  include_transpose = FALSE,
  chunk_cols = 2048L,
  codec = "raw",
  delta = FALSE,
  verbose = FALSE
)

sp_read_dense(path)

sp_convert(
  input,
  output,
  precision = "auto",
  include_transpose = FALSE,
  row_sort = TRUE,
  verbose = 1L
)
```

## See also

[`streampress`](https://zdebruine.github.io/RcppML/reference/streampress.md),
[`st_write`](https://zdebruine.github.io/RcppML/reference/st_write.md),
[`st_read`](https://zdebruine.github.io/RcppML/reference/st_read.md),
[`st_info`](https://zdebruine.github.io/RcppML/reference/st_info.md),
[`st_write_dense`](https://zdebruine.github.io/RcppML/reference/st_write_dense.md),
[`st_read_dense`](https://zdebruine.github.io/RcppML/reference/st_read_dense.md)
