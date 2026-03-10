# Read Observation (Row) Metadata from a StreamPress File

Reads the embedded obs table from a v2 `.spz` file without decompressing
the matrix data. Returns an empty data.frame if no obs table was stored.

## Usage

``` r
st_read_obs(path)
```

## Arguments

- path:

  Path to a `.spz` file.

## Value

A `data.frame` with observation metadata, or an empty data.frame if no
obs table is present.

## See also

[`st_read_var`](https://zdebruine.github.io/RcppML/reference/st_read_var.md),
[`st_write`](https://zdebruine.github.io/RcppML/reference/st_write.md)

## Examples

``` r
if (FALSE) { # \dontrun{
obs <- st_read_obs("data.spz")
head(obs)
} # }
```
