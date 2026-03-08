# SparsePress (SPZ) Format

## Overview

SparsePress is a column-oriented compressed sparse matrix format designed for fast I/O on large-scale sparse data (scRNA-seq, genomic count matrices). SPZ v2 uses chunked compression with rANS entropy coding, achieving **10–20× compression** while supporting random-access decompression at the chunk level.

---

## SPZ v2 File Layout

```
[FileHeader_v2: 128 bytes]
├── Magic: "SPRZ" (4 bytes)
├── Version: 2 (u16)
├── Dimensions: m (rows), n (cols), nnz (u64)
├── Chunk config: chunk_cols=256, num_chunks, table_log=11
├── Value type (UINT8/16/32, FLOAT32/16/64, QUANT8)
├── Section offsets: chunk_index, tables, data, transpose, metadata
└── Density, max_value, reserved

[ChunkDescriptor_v2 × num_chunks: 48 bytes each]
├── col_start, num_cols, nnz
├── stream_offset[2], stream_size[2] (gap & value streams)
├── decoded_gap_bytes, decoded_value_bytes
└── quant_scale, quant_offset (for QUANT8)

[Data Section: concatenated chunk data]
├── All gap streams (rANS-encoded row indices)
└── All value streams (rANS or byte-shuffled)

[Transpose Section (optional): CSC(A^T) pre-stored]
├── num_transpose_chunks (u32)
├── TransposeChunkDescriptor × num_transpose_chunks
└── Transpose chunk data

[Metadata: variable-length]
├── Row names, column names, row_permutation (if row_sort=true)
└── Self-describing key-value format

[Footer_v2: 16 bytes]
├── metadata_size (u32)
├── file_crc32 (u32)
├── total_chunks (u32)
└── Magic: "SPEN" (4 bytes)
```

---

## Encoding Methods

### Gap Encoding (Row Indices)

Sparse column data is represented by row indices of non-zero entries. These are encoded as **delta gaps** — the difference between consecutive row indices minus 1, reset at each column boundary.

**Pipeline**: Row indices → Per-column delta gaps → rANS entropy coding

```
Per chunk:
[col_counts: varint]         ← number of nonzeros per column
[rANS_table][rANS_data]     ← entropy-coded gaps (symbols 0–254)
[overflow: varint]           ← gaps ≥ 255 (escaped from rANS)
```

- **Symbols 0–254**: Encoded directly by the rANS codec
- **Symbol 255**: Escape flag; actual gap value stored in the overflow section as a VarInt
- **Probability table**: 14-bit scale ($2^{14} = 16384$ entries)

### Value Encoding

Value encoding depends on the data type:

| Type | Encoding Strategy |
|---|---|
| **UINT8/16/32** | Direct rANS with varint escape |
| **FLOAT32** | Byte-shuffle into 4 streams → per-stream rANS |
| **FLOAT16** | fp32→fp16 conversion → byte-shuffle (2 streams) → rANS |
| **FLOAT64** | Byte-shuffle into 8 streams → per-stream rANS |
| **QUANT8** | Affine quantization (min/max per chunk) → rANS |

**Byte-shuffling** separates the bytes of each float into independent streams. Since the exponent bytes and mantissa bytes of floating-point numbers have very different entropy distributions, encoding them separately achieves much better compression.

### rANS Entropy Coding

rANS (range Asymmetric Numeral Systems) is a fast, near-optimal entropy coder that approaches the Shannon limit. Each symbol is encoded using a frequency table derived from the chunk data.

- Encoding: $O(1)$ per symbol (table lookup + integer operations)
- Decoding: $O(1)$ per symbol (state update)
- Table size: 16KB (14-bit precision)

---

## Streaming Decompression

### Architecture

The `SpzLoader<Scalar>` class provides panel-at-a-time decompression:

```cpp
SpzLoader<float> loader(path, require_transpose = true);

// Iterate over panels (forward: columns)
loader.reset_forward();
Chunk<float> chunk;
while (loader.next_forward(chunk)) {
    // chunk.matrix: Eigen::SparseMatrix<float, ColMajor>
    // chunk.col_start: first column index (global)
    // chunk.num_cols: columns in this chunk (≤ 256)
    // Process panel...
}

// Iterate over transpose panels (rows as columns of A^T)
loader.reset_transpose();
while (loader.next_transpose(chunk)) {
    // For W-update in NMF
}
```

### Decompression Steps (Per Chunk)

1. **Read chunk descriptor** from the index: column range, nnz, stream offsets
2. **Decode gap stream**: Parse column counts (varint) → decode rANS gaps → reconstruct row indices
3. **Decode value stream**: Decode rANS → de-shuffle bytes (for floats) → cast to output type
4. **Build Eigen sparse matrix**: Assemble CSC (col_ptr, row_idx, values)

### Memory Model

During streaming NMF:

| Component | Memory | Notes |
|---|---|---|
| Compressed file | $O(\text{file\_size})$ | Entire file read into memory |
| Decompressed chunk | $O(\text{max\_chunk\_nnz})$ | Largest panel |
| NMF factors | $O(k(m + n))$ | $W$, $d$, $H$ |
| **Total** | ~150 MB for 2 GB uncompressed | 10–15× memory reduction |

---

## Streaming NMF Integration

### Panel-at-a-Time Iteration

```
for iter = 1 to maxit:

    // H update (from forward panels)
    loader.reset_forward()
    while loader.next_forward(chunk):
        G = W^T W                          // global Gram
        B_chunk = W^T · chunk.matrix       // local RHS
        H[:, col_range] = nnls(G, B_chunk) // update panel of H

    // W update (from transpose panels)
    loader.reset_transpose()
    while loader.next_transpose(chunk):
        G' = H H^T                         // global Gram
        B_chunk = H · chunk.matrix         // local RHS
        W[row_range, :] = nnls(G', B_chunk)

    // Loss computation and convergence check
```

### Key Properties

- **Panel boundaries are transparent**: The lazy CV mask, IRLS weights, and ZI posteriors all work identically for streaming and in-memory
- **Non-MSE losses supported**: Streaming paths pass `weight_zeros=true` to IRLS, computing correct distribution-derived weights at zero entries
- **GPU streaming**: Chunks decompress on CPU, transfer to GPU device memory, compute on GPU, transfer results back

---

## Compression Options

| Option | Effect |
|---|---|
| `row_sort = TRUE` | Reorder rows by nnz for better gap entropy (+10–15% compression) |
| `include_transpose = TRUE` | Pre-store CSC($A^T$) — eliminates runtime transpose, ~2× file size |
| `precision = "fp16"` | Half-precision float storage — good for count data |
| `precision = "quant8"` | 8-bit affine quantization per chunk — maximum compression |
| `delta = TRUE` | Density-based gap prediction (default) |

---

## Compression Performance

### Typical Results (scRNA-seq Data)

| Dataset | Dimensions | Density | Uncompressed | SPZ | Ratio |
|---|---|---|---|---|---|
| PBMC 33K | 33K × 5.2K | 0.5% | 660 MB | 40 MB | 16.5× |
| Lung 63K | 63K × 36K | 0.2% | 5.1 GB | 290 MB | 17.6× |

### Decompression Speed

- Full matrix: ~50–200 MB/s (CPU, depends on chunk size)
- Single chunk: ~100–300 MB/s (parallelizable with OpenMP)
- Faster than reading uncompressed `.rds` files on typical NFS filesystems

---

## R Interface

```r
# Write sparse matrix to SPZ file
sp_write(A, "data.spz",
         delta = TRUE,              # gap prediction
         precision = "auto",        # value type selection
         row_sort = FALSE,          # reorder rows by nnz
         include_transpose = FALSE) # pre-store A^T

# Read SPZ file to sparse matrix
A <- sp_read("data.spz",
             cols = NULL,           # partial read (e.g., 1:500)
             reorder = TRUE)        # undo row permutation

# Inspect header without decompression
info <- sp_info("data.spz")
# Returns: rows, cols, nnz, density_pct, file_bytes, raw_bytes, ratio, version

# In-memory compression/decompression
blob <- sp_compress(A, delta = TRUE)
A_reconstructed <- sp_decompress(blob)
```

---

## References

1. Duda, J. (2013). Asymmetric numeral systems: entropy coding combining speed of Huffman coding with compression rate of arithmetic coding. *arXiv:1311.2540*.
