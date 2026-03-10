# Methods for gpu_sparse_matrix objects

S3 methods for the `gpu_sparse_matrix` class returned by
[`sp_read_gpu`](https://zdebruine.github.io/RcppML/reference/sp_read_gpu.md).

## Usage

``` r
# S3 method for class 'gpu_sparse_matrix'
print(x, ...)

# S3 method for class 'gpu_sparse_matrix'
dim(x)

# S3 method for class 'gpu_sparse_matrix'
nrow(x)

# S3 method for class 'gpu_sparse_matrix'
ncol(x)
```

## Arguments

- x:

  a `gpu_sparse_matrix` object

- ...:

  additional arguments (unused)

## Value

- `print`:

  Invisibly returns `x`, prints summary to console.

- `dim`:

  Integer vector of length 2: `c(nrow, ncol)`.

- `nrow`:

  Number of rows (integer).

- `ncol`:

  Number of columns (integer).

## See also

[`sp_read_gpu`](https://zdebruine.github.io/RcppML/reference/sp_read_gpu.md),
[`sp_free_gpu`](https://zdebruine.github.io/RcppML/reference/sp_free_gpu.md)

## Examples

``` r
if (FALSE) { # \dontrun{
gpu_mat <- sp_read_gpu("data.spz")
print(gpu_mat)
dim(gpu_mat)
nrow(gpu_mat)
ncol(gpu_mat)
sp_free_gpu(gpu_mat)
} # }
```
