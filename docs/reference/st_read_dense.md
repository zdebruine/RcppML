# Read a Dense Matrix from StreamPress v3 Format

Read a Dense Matrix from StreamPress v3 Format

## Usage

``` r
st_read_dense(path)
```

## Arguments

- path:

  Path to a StreamPress v3 `.spz` file.

## Value

A numeric matrix.

## See also

[`st_write_dense`](https://zdebruine.github.io/RcppML/reference/st_write_dense.md),
[`st_read`](https://zdebruine.github.io/RcppML/reference/st_read.md)

## Examples

``` r
# \donttest{
A <- matrix(rnorm(1000), 50, 20)
f <- tempfile(fileext = ".spz")
st_write_dense(A, f)
B <- st_read_dense(f)
max(abs(A - B))
#> [1] 1.149582e-07
unlink(f)
# }
```
