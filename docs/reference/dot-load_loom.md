# Load matrix from Loom file (.loom)

Loom files store the main matrix in /matrix (genes x cells, row-major).
Gene names may be in /row_attrs/Gene, cell IDs in /col_attrs/CellID.

## Usage

``` r
.load_loom(path)
```

## Arguments

- path:

  Path to .loom file

## Value

A dgCMatrix (transposed to features x samples)
