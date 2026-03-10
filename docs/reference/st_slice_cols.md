# Slice Columns from a StreamPress File

Read a subset of columns from a `.spz` file without decompressing the
entire matrix.

## Usage

``` r
st_slice_cols(path, cols, threads = 0L)
```

## Arguments

- path:

  Path to a `.spz` file.

- cols:

  Integer vector of column indices (1-indexed).

- threads:

  Integer; number of threads (0 = all available). Default 0.

## Value

A `dgCMatrix` sparse matrix.

## See also

[`st_slice_rows`](https://zdebruine.github.io/RcppML/reference/st_slice_rows.md),
[`st_slice`](https://zdebruine.github.io/RcppML/reference/st_slice.md)

## Examples

``` r
if (FALSE) { # \dontrun{
mat <- st_slice_cols("data.spz", cols = 1:10)
dim(mat)
} # }
```
