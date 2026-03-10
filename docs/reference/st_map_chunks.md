# Apply a Function to Every Chunk in a StreamPress File

Sequentially reads and decodes each chunk, applies `fn`, and collects
results.

## Usage

``` r
st_map_chunks(path, fn, transpose = FALSE, threads = 0L)
```

## Arguments

- path:

  Path to a `.spz` file.

- fn:

  Function taking `(chunk, col_start, col_end)` where `chunk` is a
  `dgCMatrix`.

- transpose:

  Logical; if `TRUE`, iterate over transpose chunks (row chunks of the
  original matrix). Default `FALSE`.

- threads:

  Integer; decode threads per chunk. Default 0 (all threads).

## Value

Invisible list of results from `fn`.

## See also

[`st_chunk_ranges`](https://zdebruine.github.io/RcppML/reference/st_chunk_ranges.md)

## Examples

``` r
if (FALSE) { # \dontrun{
# Compute column sums per chunk
st_map_chunks("data.spz", function(chunk, s, e) Matrix::colSums(chunk))
} # }
```
