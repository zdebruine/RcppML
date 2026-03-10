# Read Variable (Column) Metadata from a StreamPress File

Reads the embedded var table from a v2 `.spz` file without decompressing
the matrix data. Returns an empty data.frame if no var table was stored.

## Usage

``` r
st_read_var(path)
```

## Arguments

- path:

  Path to a `.spz` file.

## Value

A `data.frame` with variable metadata, or an empty data.frame if no var
table is present.

## See also

[`st_read_obs`](https://zdebruine.github.io/RcppML/reference/st_read_obs.md),
[`st_write`](https://zdebruine.github.io/RcppML/reference/st_write.md)

## Examples

``` r
if (FALSE) { # \dontrun{
var <- st_read_var("data.spz")
head(var)
} # }
```
