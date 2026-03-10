# Read a SparsePress (.spz) file into a dgCMatrix

Read a SparsePress (.spz) file into a dgCMatrix

## Usage

``` r
Rcpp_sp_read(path, cols = NULL, reorder = TRUE)
```

## Arguments

- path:

  Input file path (.spz)

- cols:

  Optional integer vector of column indices to read (1-indexed)

- reorder:

  Whether to undo row permutation (default TRUE)

## Value

A sparse matrix (dgCMatrix)
