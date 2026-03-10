# Get Row Indices Matching Observation Metadata Filter

Reads the obs table, applies a filter expression via
[`subset()`](https://rdrr.io/r/base/subset.html), and returns matching
row indices.

## Usage

``` r
st_obs_indices(path, ...)
```

## Arguments

- path:

  Path to a `.spz` file.

- ...:

  Filter expressions passed to
  [`subset()`](https://rdrr.io/r/base/subset.html) on the obs table.

## Value

Integer vector of matching row indices (1-based).

## See also

[`st_filter_rows`](https://zdebruine.github.io/RcppML/reference/st_filter_rows.md),
[`st_read_obs`](https://zdebruine.github.io/RcppML/reference/st_read_obs.md)

## Examples

``` r
if (FALSE) { # \dontrun{
idx <- st_obs_indices("data.spz", cell_type == "B cell")
length(idx)
} # }
```
