# Slice Columns Matching Variable Metadata Filter

Slice Columns Matching Variable Metadata Filter

## Usage

``` r
st_filter_cols(path, ..., threads = 0L)
```

## Arguments

- path:

  Path to a `.spz` file.

- ...:

  Filter expression on var columns (passed to
  [`subset()`](https://rdrr.io/r/base/subset.html)).

- threads:

  Integer decode threads. 0 = all.

## Value

A `dgCMatrix` sparse matrix.

## See also

[`st_filter_rows`](https://zdebruine.github.io/RcppML/reference/st_filter_rows.md),
[`st_read_var`](https://zdebruine.github.io/RcppML/reference/st_read_var.md)

## Examples

``` r
if (FALSE) { # \dontrun{
mat <- st_filter_cols("data.spz", highly_variable == TRUE)
dim(mat)
} # }
```
