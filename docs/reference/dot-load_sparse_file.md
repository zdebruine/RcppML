# Load a sparse matrix from a file path

Auto-detects format from file extension and loads the matrix. Supported
formats: .spz, .rds, .mtx, .mtx.gz, .csv, .csv.gz, .tsv, .tsv.gz, .h5,
.hdf5, .h5ad

## Usage

``` r
.load_sparse_file(path)
```

## Arguments

- path:

  Character string giving path to the file.

## Value

A matrix (dense or dgCMatrix).
