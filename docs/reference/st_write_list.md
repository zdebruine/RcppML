# Write a List of Matrices as a Single StreamPress File

Column-concatenates a list of matrices and writes them as a single
`.spz` file. All matrices must have the same number of rows.

## Usage

``` r
st_write_list(
  x,
  path,
  obs = NULL,
  var = NULL,
  chunk_bytes = 6.4e+07,
  chunk_cols = NULL,
  include_transpose = TRUE,
  precision = "auto",
  threads = 0L,
  verbose = FALSE
)
```

## Arguments

- x:

  A list of `dgCMatrix` objects (or coercible). All must have identical
  `nrow`.

- path:

  Output `.spz` path.

- obs:

  Optional data.frame of cell metadata (`nrow` == total cols).

- var:

  Optional data.frame of gene metadata (`nrow` == nrow of mats).

- chunk_bytes:

  Target bytes per chunk. Default 64 MB.

- chunk_cols:

  Explicit column count per chunk. Overrides `chunk_bytes`.

- include_transpose:

  Logical. Default `TRUE`.

- precision:

  Value precision. Default `"auto"`.

- threads:

  Integer. 0 = all threads.

- verbose:

  Logical.

## Value

Invisibly, compression statistics.

## See also

[`st_write`](https://zdebruine.github.io/RcppML/reference/st_write.md)

## Examples

``` r
if (FALSE) { # \dontrun{
mats <- list(mat1, mat2)
st_write_list(mats, "combined.spz")
} # }
```
