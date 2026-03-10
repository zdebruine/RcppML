# Load matrix from generic HDF5 file

Looks for datasets named "X", "matrix", or "data". Falls back to the
first 2-D dataset found.

## Usage

``` r
.load_h5(path)
```

## Arguments

- path:

  Path to .h5 file

## Value

A matrix or dgCMatrix
