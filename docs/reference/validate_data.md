# Validate and prepare input data

Accepts a matrix (dense or sparse), or a file path (character string).
File paths are auto-loaded based on extension:

- `.spz`: StreamPress compressed format

- `.rds`: R serialized object (must contain a matrix)

- `.mtx`, `.mtx.gz`: Matrix Market format (requires Matrix package)

- `.csv`, `.csv.gz`, `.tsv`, `.tsv.gz`: Delimited text

- `.h5`, `.hdf5`: HDF5 (requires hdf5r package)

- `.h5ad`: AnnData HDF5 (requires hdf5r package)

## Usage

``` r
validate_data(data)
```

## Arguments

- data:

  A matrix, sparse matrix, or file path (character string).

## Value

A list with `data`, `is_sparse`, and `has_na`.
