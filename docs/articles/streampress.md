# StreamPress: High-Performance Sparse Matrix I/O

## Motivation

R’s built-in serialization formats (`.rds`, `.rda`) are general-purpose
but inefficient for sparse matrices. They serialize the entire CSC
structure including all slot metadata, which inflates file sizes and
slows I/O for large datasets.

StreamPress (`.spz`) is a columnar compressed format designed
specifically for sparse matrices. It uses rANS entropy coding with
delta-encoded indices to achieve 2–20× compression over `.rds` files
while supporting random column access and streaming reads. For dense
matrices, StreamPress v3 provides optional FP16, QUANT8, and rANS
compression codecs.

## API Reference

### Writing Sparse Matrices

[`st_write()`](https://zdebruine.github.io/RcppML/reference/st_write.md)
compresses a sparse `dgCMatrix` to `.spz` format:

``` r
stats <- st_write(x, path,
  precision = "auto",         # "fp64", "fp32", "fp16", "quant8", or "auto"
  include_transpose = TRUE,   # store CSC(A^T) for fast row access
  delta = TRUE,               # delta prediction for row indices
  value_pred = FALSE,         # value prediction for integer data
  chunk_cols = NULL,          # columns per chunk (NULL = auto from chunk_bytes)
  chunk_bytes = 64e6,         # target bytes per chunk
  verbose = FALSE             # print compression statistics
)
```

Returns an invisible list with compression statistics: `raw_bytes`,
`compressed_bytes`, `ratio`, `compress_ms`.

### Reading Sparse Matrices

[`st_read()`](https://zdebruine.github.io/RcppML/reference/st_read.md)
decompresses a `.spz` file to a `dgCMatrix`:

``` r
mat <- st_read(path,
  cols = NULL,      # column indices to read (NULL = all)
  reorder = TRUE,   # undo any row permutation
  threads = 0       # number of threads (0 = all)
)
```

The `cols` parameter enables partial reads — load only the columns you
need without decompressing the entire file.

### Inspecting Files

[`st_info()`](https://zdebruine.github.io/RcppML/reference/st_info.md)
reads only the header, with no decompression:

``` r
info <- st_info(path)
# Returns: rows, cols, nnz, density_pct, file_bytes, raw_bytes, ratio,
#          version, value_type, chunk_cols, num_chunks, ...
```

### Dense Matrix Support

StreamPress v3 handles dense matrices with optional compression codecs:

``` r
st_write_dense(x, path,
  codec = "raw",              # "raw", "fp16", "quant8", "fp16_rans", "fp32_rans"
  include_transpose = FALSE,  # store transposed panels
  chunk_cols = 2048L          # columns per chunk
)
mat <- st_read_dense(path)
```

## Theory

### Columnar Storage

StreamPress stores each column independently in a CSC-like layout. This
design enables:

- **Random column access**: Read column $j$ without decompressing
  columns 1 through $j - 1$.
- **Streaming I/O**: Process columns in chunks for out-of-core NMF on
  datasets larger than RAM.
- **Parallel decompression**: Independent columns can be decoded
  concurrently.

### Compression Pipeline

For each column, StreamPress applies:

1.  **Delta encoding** of row indices: Store differences between
    consecutive indices rather than absolute values. For dense rows,
    deltas are mostly 1 — tiny integers that compress extremely well.
2.  **rANS entropy coding**: An asymmetric numeral system that
    approaches the Shannon entropy bound, yielding near-optimal
    compression.
3.  **Precision reduction** (optional): Convert 64-bit doubles to 32-bit
    floats, 16-bit half-precision, or 8-bit quantized values before
    entropy coding.

### Precision Hierarchy

| Precision | Bytes/value | Exact for integers up to | Typical use case           |
|:----------|:------------|:-------------------------|:---------------------------|
| `fp64`    | 8           | $2^{53}$                 | Lossless (default)         |
| `fp32`    | 4           | $2^{24}$ (16.7 million)  | Count data, most use cases |
| `fp16`    | 2           | $2^{11}$ (2,048)         | Small counts, ratings      |
| `quant8`  | 1           | 256 buckets              | Maximum compression        |

For integer count data (UMI counts, ratings), `fp32` or `fp16` is
lossless up to the limits shown. `quant8` maps values to 256
equally-spaced buckets, introducing quantization error proportional to
the data range.

## Worked Examples

### Example 1: Sparse Matrix Round-Trip

We write the MovieLens ratings matrix to StreamPress and verify lossless
recovery.

``` r
data(movielens)

# Write to StreamPress
spz_file <- tempfile(fileext = ".spz")
stats <- st_write(movielens, spz_file, include_transpose = FALSE)

# Inspect the file
info <- st_info(spz_file)

# Read back
ml_back <- st_read(spz_file)

# Compare file sizes
rds_file <- tempfile(fileext = ".rds")
saveRDS(movielens, rds_file)

size_df <- data.frame(
  Format = c("R object (in memory)", ".rds file", ".spz file"),
  `Size (KB)` = c(
    round(as.numeric(object.size(movielens)) / 1024, 1),
    round(file.info(rds_file)$size / 1024, 1),
    round(file.info(spz_file)$size / 1024, 1)
  ),
  check.names = FALSE
)

knitr::kable(size_df, align = "lr",
             caption = "MovieLens storage: R object vs. .rds vs. .spz")
```

| Format               | Size (KB) |
|:---------------------|----------:|
| R object (in memory) |    1664.6 |
| .rds file            |     271.6 |
| .spz file            |      68.2 |

MovieLens storage: R object vs. .rds vs. .spz

``` r
# Verify values are identical (dimnames may differ since SPZ doesn't store them)
cat("Values identical:", identical(movielens@x, ml_back@x), "\n")
#> Values identical: TRUE
cat("Structure identical:", identical(movielens@i, ml_back@i) &&
      identical(movielens@p, ml_back@p), "\n")
#> Structure identical: TRUE
```

The `.spz` format preserves all numeric values and the CSC structure
exactly. Dimension names (row/column labels) are not stored in the
`.spz` file itself; use R attributes or sidecar metadata for these.

### Example 2: Dense Matrix Round-Trip

StreamPress v3 handles dense matrices with optional compression codecs.

``` r
data(aml)

# Write dense matrix
dense_file <- tempfile(fileext = ".spz")
st_write_dense(aml, dense_file)

# Read back
aml_back <- st_read_dense(dense_file)

# Compare sizes
rds_dense <- tempfile(fileext = ".rds")
saveRDS(aml, rds_dense)

dense_df <- data.frame(
  Format = c("R object (in memory)", ".rds file", ".spz v3 file"),
  `Size (KB)` = c(
    round(as.numeric(object.size(aml)) / 1024, 1),
    round(file.info(rds_dense)$size / 1024, 1),
    round(file.info(dense_file)$size / 1024, 1)
  ),
  check.names = FALSE
)
knitr::kable(dense_df, align = "lr",
             caption = "AML dense matrix storage comparison")
```

| Format               | Size (KB) |
|:---------------------|----------:|
| R object (in memory) |     953.6 |
| .rds file            |     825.6 |
| .spz v3 file         |     434.7 |

AML dense matrix storage comparison

``` r
cat("Max absolute difference:", max(abs(aml - aml_back)), "\n")
#> Max absolute difference: 2.98015e-08
```

### Example 3: Precision Mode Comparison

Different precision modes trade file size for numerical accuracy. We
write the MovieLens data at each precision level and measure the error
introduced.

``` r
precisions <- c("fp64", "fp32", "fp16", "quant8")
prec_results <- data.frame(
  Precision = precisions,
  `File Size (KB)` = NA_real_,
  `Max Abs Error` = NA_real_,
  Lossless = NA_character_,
  check.names = FALSE,
  stringsAsFactors = FALSE
)

for (i in seq_along(precisions)) {
  f <- tempfile(fileext = ".spz")
  st_write(movielens, f, precision = precisions[i], include_transpose = FALSE)
  back <- st_read(f)
  err <- max(abs(movielens@x - back@x))
  prec_results$`File Size (KB)`[i] <- round(file.info(f)$size / 1024, 1)
  prec_results$`Max Abs Error`[i] <- round(err, 6)
  prec_results$Lossless[i] <- if (err == 0) "Yes" else "No"
}

knitr::kable(prec_results, align = "lrrr",
             caption = "Precision vs. compression tradeoff (MovieLens ratings)")
```

| Precision | File Size (KB) | Max Abs Error | Lossless |
|:----------|---------------:|--------------:|---------:|
| fp64      |           74.0 |      0.000000 |      Yes |
| fp32      |           68.2 |      0.000000 |      Yes |
| fp16      |           69.2 |      0.000000 |      Yes |
| quant8    |           68.7 |      0.007843 |       No |

Precision vs. compression tradeoff (MovieLens ratings)

For integer rating data (values 1–5), `fp64`, `fp32`, and `fp16` are all
lossless because the values fit within the exact representation range of
each format. Only `quant8` introduces quantization error, mapping the
continuous value range into 256 discrete buckets.

### Example 4: Real-World Dataset — pbmc3k Single-Cell RNA-seq

The `pbmc3k` dataset ships as StreamPress-compressed raw bytes — a
representative subset (8,000 genes × 500 cells) of the 10x Genomics PBMC
3k dataset, demonstrating how SPZ enables shipping sparse datasets
within CRAN’s tarball size limits.

``` r
# Load compressed bytes (not a matrix — raw bytes)
data(pbmc3k)

# Write to temp file and inspect
pbmc3k_file <- tempfile(fileext = ".spz")
writeBin(pbmc3k, pbmc3k_file)
pbmc_info <- st_info(pbmc3k_file)

pbmc_df <- data.frame(
  Property = c("Rows (genes)", "Columns (cells)", "Non-zeros",
               "Density", "SPZ file size", "Compression ratio"),
  Value = c(
    format(pbmc_info$rows, big.mark = ","),
    format(pbmc_info$cols, big.mark = ","),
    format(pbmc_info$nnz, big.mark = ","),
    paste0(round(pbmc_info$density_pct, 1), "%"),
    paste0(round(pbmc_info$file_bytes / 1024, 0), " KB"),
    paste0(round(pbmc_info$ratio, 1), "x")
  ),
  stringsAsFactors = FALSE
)
knitr::kable(pbmc_df, align = "lr", caption = "pbmc3k StreamPress file summary")
```

| Property          |   Value |
|:------------------|--------:|
| Rows (genes)      |   8,000 |
| Columns (cells)   |     500 |
| Non-zeros         | 412,180 |
| Density           |   10.3% |
| SPZ file size     |  663 KB |
| Compression ratio |    3.6x |

pbmc3k StreamPress file summary

``` r
# Load into R and run a quick NMF on a subset
counts <- st_read(pbmc3k_file)

# Subset for speed: 500 most variable genes × all cells
# Compute row variance efficiently: Var = E[x^2] - E[x]^2
n <- ncol(counts)
row_means <- Matrix::rowMeans(counts)
row_sq_means <- Matrix::rowMeans(counts^2)
gene_var <- row_sq_means - row_means^2
top_genes <- order(gene_var, decreasing = TRUE)[1:500]
counts_sub <- counts[top_genes, ]

model <- nmf(counts_sub, k = 5, seed = 42, tol = 1e-4, maxit = 50,
             verbose = FALSE)
cat("NMF complete: k =", ncol(model@w), ", loss =",
    round(model@misc$loss, 2), "\n")
#> NMF complete: k = 5 , loss = 3287246
```

### Example 5: Out-of-Core Streaming NMF (Conceptual)

For datasets too large to fit in memory, StreamPress enables streaming
NMF that processes the matrix in chunks directly from disk:

``` r
# Stream NMF from a large .spz file
# The matrix is never fully loaded into RAM
model <- nmf("/path/to/huge_matrix.spz", k = 20, streaming = TRUE,
             maxit = 50, seed = 42)
```

Streaming reads chunks of columns from the `.spz` file, updates the
factor matrices incrementally, and discards each chunk before loading
the next. This enables NMF on datasets that are 10–100× larger than
available RAM.

## Key Takeaways

1.  **StreamPress achieves substantial compression** over both R objects
    and `.rds` files, with near-instantaneous decompression for random
    column access.
2.  **Precision modes** offer a clear tradeoff: `fp64` for lossless
    preservation, `fp32`/`fp16` for lossless on integer data, `quant8`
    for maximum compression with controlled error.
3.  **The pbmc3k shipping pattern** (`raw` bytes in `.rda` → `writeBin`
    → `st_read`) enables distributing large datasets within CRAN’s
    tarball size limits.
4.  **Streaming NMF** from `.spz` files enables analysis of datasets
    larger than available RAM.

*See the [GPU
Acceleration](https://zdebruine.github.io/RcppML/articles/gpu-acceleration.md)
vignette for GPU-direct reads from `.spz`, and the [NMF
Fundamentals](https://zdebruine.github.io/RcppML/articles/nmf-fundamentals.md)
vignette for fitting NMF models on loaded data.*
