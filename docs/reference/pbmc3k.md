# PBMC 3k Single-Cell RNA-seq Dataset (StreamPress Compressed)

The full 10x Genomics PBMC 3k single-cell RNA-seq dataset with Seurat
cell type annotations, shipped as StreamPress-compressed raw bytes.
Contains 13,714 genes across 2,638 cells with 9 annotated cell types.

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
      # counts is a dgCMatrix: 13,714 genes x 2,638 cells
      

Cell type annotations are embedded in the .spz file as column (var)
metadata:

      cell_types <- st_read_var(tmp)$cell_type
      table(cell_types)
      

## Source

10x Genomics PBMC 3k dataset, processed with Seurat
(SeuratData::pbmc3k.final).

## Details

The underlying matrix is a `dgCMatrix` with 13,714 rows (genes) and
2,638 columns (cells), containing 2,238,732 non-zero entries (integer
UMI counts). Cell type annotations (9 types: Naive CD4 T, Memory CD4 T,
CD14+ Mono, B, CD8 T, FCGR3A+ Mono, NK, DC, Platelet) were obtained from
the Seurat `pbmc3k.final` reference object via the SeuratData package
and stored as StreamPress column metadata.

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
dim(counts)  # 13714 x 2638
#> [1] 13714  2638

# Access cell type annotations
cell_types <- st_read_var(tmp)$cell_type
table(cell_types)
#> cell_types
#> Memory CD4 T            B   CD14+ Mono           NK        CD8 T  Naive CD4 T 
#>          483          344          480          155          271          697 
#> FCGR3A+ Mono           DC     Platelet 
#>          162           32           14 
# }
```
