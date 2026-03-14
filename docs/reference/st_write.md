# Write a sparse matrix to a StreamPress file

Write a sparse matrix to a StreamPress file

## Usage

``` r
st_write(
  x,
  path,
  obs = NULL,
  var = NULL,
  delta = TRUE,
  value_pred = FALSE,
  verbose = FALSE,
  precision = "auto",
  row_sort = FALSE,
  include_transpose = TRUE,
  chunk_cols = NULL,
  chunk_bytes = 8e+06,
  transp_chunk_cols = NULL,
  transp_chunk_bytes = NULL,
  threads = 0L
)
```

## Arguments

- x:

  A sparse matrix (`dgCMatrix`) or object coercible to one.

- path:

  Output file path. Extension `.spz` is recommended.

- obs:

  Optional data.frame of observation (row/cell) metadata. Must have
  `nrow(obs) == nrow(x)`.

- var:

  Optional data.frame of variable (column/gene) metadata. Must have
  `nrow(var) == ncol(x)`.

- delta:

  Logical; use density-based delta prediction for structure. Default
  `TRUE`.

- value_pred:

  Logical; use value prediction for integer-valued data. Default
  `FALSE`.

- verbose:

  Logical; print compression statistics. Default `FALSE`.

- precision:

  Value precision: `"auto"` (default), `"fp32"`, `"fp16"`, `"quant8"`,
  `"fp64"`.

- row_sort:

  Logical; sort rows by nnz for better compression.

- include_transpose:

  Logical; store CSC(A^T) in the file. Default `TRUE`.

- chunk_cols:

  Integer or NULL; columns per chunk. If NULL, computed from
  `chunk_bytes`.

- chunk_bytes:

  Numeric; target bytes per chunk when `chunk_cols` is NULL. Default 8
  MB, which yields ~50 columns per chunk for typical scRNA-seq matrices
  (~38 k rows). Smaller chunks create more parallel work during reading;
  larger chunks compress slightly better.

- transp_chunk_cols:

  Integer or NULL; columns per transpose chunk.

- transp_chunk_bytes:

  Numeric or NULL; target bytes per transpose chunk.

- threads:

  Integer; number of threads for parallel compression (0 = all
  available). Default 0.

## Value

Invisibly returns a list with compression statistics.

## See also

[`st_read`](https://zdebruine.github.io/RcppML/reference/st_read.md),
[`st_info`](https://zdebruine.github.io/RcppML/reference/st_info.md)

## Examples

``` r
# \donttest{
library(Matrix)
A <- rsparsematrix(1000, 500, 0.05)
f <- tempfile(fileext = ".spz")
st_write(A, f)
B <- st_read(f)
all.equal(A, B)  # TRUE
#> [1] "Mean relative difference: 2.431235e-08"
unlink(f)
# }
```
