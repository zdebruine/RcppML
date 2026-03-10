# PBMC 3k Single-Cell RNA-seq Dataset (StreamPress Compressed)

A representative subset of the 10x Genomics PBMC 3k single-cell RNA-seq
dataset, shipped as StreamPress-compressed raw bytes to meet CRAN
tarball size limits. Contains the 8,000 most variable genes across 500
cells.

## Usage

``` r
pbmc3k
```

## Format

A `raw` vector containing StreamPress (.spz) compressed bytes. To obtain
the sparse matrix, write the bytes to a temporary file and read with
[`st_read`](https://zdebruine.github.io/RcppML/reference/st_read.md):

      data(pbmc3k)
      tmp <- tempfile(fileext = ".spz")
      writeBin(pbmc3k, tmp)
      counts <- st_read(tmp)
      # counts is a dgCMatrix: 8,000 genes x 500 cells
      

## Source

10x Genomics PBMC 3k dataset, filtered, processed, and subsetted for
size.

## Details

The underlying matrix is a `dgCMatrix` with 8,000 rows (genes) and 500
columns (cells), containing 412,180 non-zero entries (integer UMI
counts). Genes were selected by variance from the full 13,714-gene
panel. The data was compressed with StreamPress, which is lossless for
integer count data.

This dataset is commonly used for demonstrating single-cell analysis
workflows including distribution-aware NMF and zero-inflation
diagnostics.

## Examples

``` r
# \donttest{
# Load the compressed bytes
data(pbmc3k)

# Decompress to sparse matrix
tmp <- tempfile(fileext = ".spz")
writeBin(pbmc3k, tmp)
counts <- st_read(tmp)
dim(counts)  # 8000 x 500
#> [1] 8000  500
# }
```
