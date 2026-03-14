# Add Transpose Section to an Existing StreamPress File

Add Transpose Section to an Existing StreamPress File

## Usage

``` r
st_add_transpose(path, verbose = TRUE)
```

## Arguments

- path:

  Path to a `.spz` file.

- verbose:

  Logical; print progress. Default TRUE.

## Value

Invisibly returns the path.

## See also

[`st_write`](https://zdebruine.github.io/RcppML/reference/st_write.md),
[`st_read`](https://zdebruine.github.io/RcppML/reference/st_read.md)

## Examples

``` r
# \donttest{
library(Matrix)
m <- rsparsematrix(50, 20, 0.3)
tmp <- tempfile(fileext = ".spz")
st_write(m, tmp, include_transpose = FALSE)
st_add_transpose(tmp)
#> [transpose] Reading 50 x 20 matrix (nnz=300)...
#> [transpose] Recompressing with transpose...
#> [transpose] Done. Transpose added (1 chunks).
info <- st_info(tmp)
unlink(tmp)
# }
```
