# Slice Rows from a StreamPress File

Read a subset of rows from a `.spz` file using the pre-stored transpose
section. Requires that the file was written with
`include_transpose = TRUE`.

## Usage

``` r
st_slice_rows(path, rows, threads = 0L)
```

## Arguments

- path:

  Path to a `.spz` file.

- rows:

  Integer vector of row indices (1-indexed).

- threads:

  Integer; number of threads (0 = all available). Default 0.

## Value

A `dgCMatrix` sparse matrix.

## See also

[`st_slice_cols`](https://zdebruine.github.io/RcppML/reference/st_slice_cols.md),
[`st_slice`](https://zdebruine.github.io/RcppML/reference/st_slice.md)

## Examples

``` r
if (FALSE) { # \dontrun{
mat <- st_slice_rows("data.spz", rows = 1:100)
dim(mat)
} # }
```
