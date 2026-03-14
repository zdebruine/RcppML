# Free GPU-Resident Sparse Matrix

Explicitly frees CUDA device memory held by a `gpu_sparse_matrix`
object. This is optional — the memory will be freed automatically when
the object is garbage-collected.

## Usage

``` r
st_free_gpu(x)
```

## Arguments

- x:

  A `gpu_sparse_matrix` object from
  [`st_read_gpu()`](https://zdebruine.github.io/RcppML/reference/st_read_gpu.md).

## Value

Invisibly returns `NULL`.

## See also

[`st_read_gpu`](https://zdebruine.github.io/RcppML/reference/st_read_gpu.md)

## Examples

``` r
if (FALSE) { # \dontrun{
gpu_mat <- st_read_gpu("matrix.spz")
st_free_gpu(gpu_mat)
} # }
```
