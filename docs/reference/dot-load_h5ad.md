# Load matrix from AnnData HDF5 (.h5ad)

Reads the X matrix from an AnnData-format HDF5 file. Supports both dense
and sparse (CSC/CSR) X matrices.

## Usage

``` r
.load_h5ad(path)
```

## Arguments

- path:

  Path to .h5ad file

## Value

A matrix or dgCMatrix
