# Convert Any Supported Format to StreamPress (.spz)

One-time migration tool: read a matrix from any supported format and
write it as a `.spz` v2 file.

## Usage

``` r
.extract_h5ad_metadata(path)
```

## Arguments

- input:

  Path to input file or an in-memory matrix.

- output:

  Path for the output `.spz` file.

- precision:

  Value precision: `"auto"`, `"fp32"`, `"fp16"`, `"quant8"`, `"fp64"`.

- include_transpose:

  Logical; store CSC(A^T).

- row_sort:

  Logical; sort rows by nnz. Default `TRUE`.

- verbose:

  Integer verbosity level (0=silent, 1=summary, 2=detailed).

## Value

Invisibly returns a list with conversion statistics.

## See also

[`st_read`](https://zdebruine.github.io/RcppML/reference/st_read.md),
[`st_write`](https://zdebruine.github.io/RcppML/reference/st_write.md),
[`st_info`](https://zdebruine.github.io/RcppML/reference/st_info.md)
