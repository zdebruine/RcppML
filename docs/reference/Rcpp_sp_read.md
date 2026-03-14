# Read a SparsePress (.spz) file into a dgCMatrix

Read a SparsePress (.spz) file into a dgCMatrix

## Usage

``` r
Rcpp_sp_read(path, cols = NULL, reorder = TRUE, threads = 0L)
```

## Arguments

- path:

  Input file path (.spz)

- cols:

  Optional integer vector of column indices to read (1-indexed)

- reorder:

  Whether to undo row permutation (default TRUE)

- threads:

  Number of threads for parallel decompression (0 = all available,
  default 0)

## Value

A sparse matrix (dgCMatrix)
