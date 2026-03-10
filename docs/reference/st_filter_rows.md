# Slice Rows Matching Observation Metadata Filter

Slice Rows Matching Observation Metadata Filter

## Usage

``` r
st_filter_rows(path, ..., threads = 0L)
```

## Arguments

- path:

  Path to a `.spz` file.

- ...:

  Filter expression on obs columns (passed to
  [`subset()`](https://rdrr.io/r/base/subset.html)).

- threads:

  Integer decode threads. 0 = all.

## Value

A `dgCMatrix` sparse matrix.

## See also

[`st_filter_cols`](https://zdebruine.github.io/RcppML/reference/st_filter_cols.md),
[`st_obs_indices`](https://zdebruine.github.io/RcppML/reference/st_obs_indices.md)

## Examples

``` r
if (FALSE) { # \dontrun{
mat <- st_filter_rows("data.spz", cell_type == "B cell")
dim(mat)
} # }
```
