# Write a Dense Matrix to StreamPress v3 Format

Write a Dense Matrix to StreamPress v3 Format

## Usage

``` r
st_write_dense(
  x,
  path,
  include_transpose = FALSE,
  chunk_cols = 2048L,
  codec = "raw",
  delta = FALSE,
  verbose = FALSE
)
```

## Arguments

- x:

  A numeric matrix.

- path:

  Output file path. Extension `.spz` is recommended.

- include_transpose:

  Logical; store transposed panels. Default `FALSE`.

- chunk_cols:

  Integer; columns per chunk. Default 2048.

- codec:

  Compression codec: `"raw"`, `"fp16"`, `"quant8"`, `"fp16_rans"`,
  `"fp32_rans"`. Default `"raw"`.

- delta:

  Logical; apply XOR-delta encoding. Default `FALSE`.

- verbose:

  Logical; print write statistics. Default `FALSE`.

## Value

Invisibly returns a list with write statistics.

## See also

[`st_read_dense`](https://zdebruine.github.io/RcppML/reference/st_read_dense.md),
[`st_write`](https://zdebruine.github.io/RcppML/reference/st_write.md)

## Examples

``` r
# \donttest{
A <- matrix(rnorm(1000), 50, 20)
f <- tempfile(fileext = ".spz")
st_write_dense(A, f, include_transpose = TRUE)
B <- st_read_dense(f)
max(abs(A - B))
#> [1] 1.164056e-07
unlink(f)
# }
```
