# Load sparse matrix from HDF5 group (10x-style CSC)

Load sparse matrix from HDF5 group (10x-style CSC)

## Usage

``` r
.load_h5_sparse_group(grp)
```

## Arguments

- grp:

  An hdf5r H5Group with data/indices/indptr/shape

## Value

A dgCMatrix
