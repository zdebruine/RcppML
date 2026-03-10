# Get Column Ranges for Each Chunk in a StreamPress File

Returns the column ranges (1-indexed, inclusive) for each chunk without
decompressing any data.

## Usage

``` r
st_chunk_ranges(path)
```

## Arguments

- path:

  Path to a `.spz` file.

## Value

A `data.frame` with columns `start` and `end`.

## See also

[`st_map_chunks`](https://zdebruine.github.io/RcppML/reference/st_map_chunks.md)

## Examples

``` r
if (FALSE) { # \dontrun{
ranges <- st_chunk_ranges("data.spz")
print(ranges)
} # }
```
