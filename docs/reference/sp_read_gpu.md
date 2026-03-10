# Read SparsePress File Directly to GPU Memory

Reads a `.spz` v2 file and decodes it directly on the GPU, returning an
opaque GPU-resident CSC matrix. This avoids the CPU-to-GPU transfer that
occurs when using
[`sp_read()`](https://zdebruine.github.io/RcppML/reference/sparsepress-deprecated.md)
followed by `nmf(data, gpu = TRUE)`.

The returned object is an external reference — the matrix data lives
entirely in GPU device memory. Pass it directly to
[`nmf()`](https://zdebruine.github.io/RcppML/reference/nmf.md) for
zero-copy GPU NMF.

## Usage

``` r
sp_read_gpu(path, device = 0L)
```

## Arguments

- path:

  Path to a `.spz` file (v2 format required).

- device:

  Integer; CUDA device ID (default 0).

## Value

An object of class `"gpu_sparse_matrix"` with fields:

- m:

  Number of rows

- n:

  Number of columns

- nnz:

  Number of non-zeros

- device:

  CUDA device ID

- .col_ptr:

  Opaque device pointer (numeric)

- .row_idx:

  Opaque device pointer (numeric)

- .values:

  Opaque device pointer (numeric)

## Details

Only `.spz` v2 format is supported for GPU decode. Use
[`sp_convert()`](https://zdebruine.github.io/RcppML/reference/sparsepress-deprecated.md)
to convert other formats to v2.

The returned object has a finalizer that automatically frees GPU memory
when the R object is garbage-collected. You can also free it manually
with
[`sp_free_gpu()`](https://zdebruine.github.io/RcppML/reference/sp_free_gpu.md).

## See also

[`sp_read`](https://zdebruine.github.io/RcppML/reference/sparsepress-deprecated.md),
[`sp_free_gpu`](https://zdebruine.github.io/RcppML/reference/sp_free_gpu.md),
[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md)

## Examples

``` r
if (FALSE) { # \dontrun{
# Read directly to GPU
gpu_data <- sp_read_gpu("data.spz")

# Run NMF on GPU-resident data (zero-copy)
result <- nmf(gpu_data, k = 10)

# Clean up (optional — GC will do this automatically)
sp_free_gpu(gpu_data)
} # }
```
