# Slice Rows and/or Columns from a StreamPress File

Convenience function combining row and column slicing.

## Usage

``` r
st_slice(path, rows = NULL, cols = NULL, threads = 0L)
```

## Arguments

- path:

  Path to a `.spz` file.

- rows:

  Optional integer vector of row indices (1-indexed).

- cols:

  Optional integer vector of column indices (1-indexed).

- threads:

  Integer; number of threads (0 = all available). Default 0.

## Value

A `dgCMatrix` sparse matrix.

## See also

[`st_slice_cols`](https://zdebruine.github.io/RcppML/reference/st_slice_cols.md),
[`st_slice_rows`](https://zdebruine.github.io/RcppML/reference/st_slice_rows.md)

## Examples

``` r
if (FALSE) { # \dontrun{
mat <- st_slice("data.spz", rows = 1:100, cols = 1:10)
dim(mat)
} # }
```
