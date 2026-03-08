# FactorNet I/O Layer

## Overview

FactorNet provides a `DataLoader` abstraction for chunked matrix access, enabling both in-memory and out-of-core (streaming) NMF. The I/O layer lives in `inst/include/FactorNet/io/` and interfaces with the SPZ v2 compressed file format from the SparsePress library.

## DataLoader Interface

**Header:** `#include <FactorNet/io/loader.hpp>`

All loaders implement the same interface — NMF algorithms accept any `DataLoader<Scalar>` without knowing the data source.

```cpp
template<typename Scalar>
class DataLoader {
public:
    virtual uint32_t rows() const = 0;              // m
    virtual uint32_t cols() const = 0;              // n
    virtual uint64_t nnz() const = 0;

    virtual uint32_t num_forward_chunks() const = 0;
    virtual uint32_t num_transpose_chunks() const = 0;

    // Sequential forward iteration (column-panels of A)
    virtual bool next_forward(Chunk<Scalar>& out) = 0;

    // Sequential transpose iteration (column-panels of A^T)
    virtual bool next_transpose(Chunk<Scalar>& out) = 0;

    virtual void reset_forward() = 0;
    virtual void reset_transpose() = 0;
};
```

### Chunk

A `Chunk` represents one column-panel of the matrix:

```cpp
template<typename Scalar>
struct Chunk {
    uint32_t col_start;   // first column index in the full matrix
    uint32_t num_cols;    // number of columns in this chunk
    uint32_t num_rows;    // m (full matrix row count)
    uint64_t nnz;         // nonzeros in this chunk
    Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int> matrix;
};
```

Chunks are fixed-size column-panels (default 256 columns, matching SPZ chunk size). The last chunk may be smaller.

## Implementations

### InMemoryLoader

**Header:** `#include <FactorNet/io/in_memory.hpp>`

Zero-copy chunked views of an in-memory Eigen sparse matrix. The forward matrix is borrowed (non-owning reference); the transpose is computed and owned.

```cpp
template<typename Scalar>
class InMemoryLoader : public DataLoader<Scalar> {
public:
    InMemoryLoader(const SpMat& A, uint32_t chunk_cols = 256);

    // Direct access (for init/loss routines)
    const SpMat& forward_matrix() const;
    const SpMat& transpose_matrix() const;
};
```

**Memory:** O(nnz) for the pre-computed transpose. Forward chunks reference contiguous CSC blocks of the original matrix (no copy).

**Use case:** Standard in-memory NMF that uses the chunked path (e.g., when the matrix is large enough to benefit from chunked processing but still fits in RAM).

### SpzLoader

**Header:** `#include <FactorNet/io/spz_loader.hpp>`

Streaming decompression of `.spz` v2 files. The compressed file is memory-mapped or read into a buffer; chunks are decompressed on demand (one at a time).

```cpp
template<typename Scalar>
class SpzLoader : public DataLoader<Scalar> {
public:
    explicit SpzLoader(const std::string& path, bool require_transpose = true);

    // Raw file access (for init routines)
    const uint8_t* file_data() const;
    size_t file_size() const;
    const sparsepress::v2::FileHeader_v2& header() const;
    bool has_transpose() const;
};
```

**Memory:** O(file_size + max_chunk_nnz). The full decompressed matrix is never held in memory — only one chunk at a time is decompressed.

**Transpose requirement:** SPZ v2 files may optionally include a transpose section. NMF requires both forward and transpose passes (for H-update and W-update respectively). If `require_transpose = true` (default), the constructor throws if the file lacks a transpose section.

## Streaming NMF

**Header:** `#include <FactorNet/nmf/fit_streaming_spz.hpp>`

```cpp
template<typename Scalar = float>
NMFResult<Scalar> nmf_streaming_spz(
    const std::string& path,
    const NMFConfig<Scalar>& config);
```

### Workflow

1. **Create loader**: `SpzLoader(path, require_transpose=true)`
   - Reads the `.spz` file into memory
   - Parses v2 header and chunk descriptors
   - Verifies transpose section exists

2. **Initialization**:
   - **Random** (`init_mode = 0`): Generate W directly — no full matrix needed
   - **SVD-based** (`init_mode = 1`): Temporarily decompress all forward chunks into a full `SparseMatrix`, run Lanczos/IRLBA, then discard the full matrix

3. **Dispatch to chunked NMF**:
   - GPU available → `nmf_chunked_gpu<Scalar>(loader, config, W_init, H_init)`
   - CPU fallback → `nmf_chunked<Scalar>(loader, config, W_init, H_init)`

### Chunked NMF Algorithm

Each ALS iteration processes the matrix in chunks rather than all at once:

```
for iter = 0 to max_iter:
    // H-update: iterate forward chunks of A
    G_W = gram(W)
    loader.reset_forward()
    while loader.next_forward(chunk):
        B_chunk = rhs(chunk.matrix, W)
        nnls_batch(G_W, B_chunk, H[:, chunk.col_start:...])

    // W-update: iterate transpose chunks of A^T
    G_H = gram(H)
    loader.reset_transpose()
    while loader.next_transpose(chunk):
        B_chunk = rhs(chunk.matrix, H)
        nnls_batch(G_H, B_chunk, W[:, chunk.col_start:...])

    // Convergence check (accumulate loss across chunks)
```

### Memory Budget

| Component | Size | Notes |
|-----------|------|-------|
| SPZ file buffer | file_size | Compressed data in RAM |
| W | m × k × sizeof(float) | Persistent |
| H | n × k × sizeof(float) | Persistent |
| G (Gram) | k × k × sizeof(float) | Recomputed |
| B (RHS chunk) | k × chunk_cols × sizeof(float) | Per-chunk |
| Decompressed chunk | chunk_nnz elements | One at a time |

Total: O(file_size + (m+n)·k + chunk_size). For a 10 GB uncompressed matrix with 10× compression, the streaming path uses ~1 GB file buffer + factor matrices.

## R Interface

```r
# Write SPZ file
write_spz(sparse_matrix, "data.spz")

# Streaming NMF (never loads full matrix)
result <- nmf("data.spz", k = 20, maxit = 100)

# The path string triggers the streaming code path automatically
```

## SPZ v2 Format Summary

The SparsePress v2 format stores sparse matrices in compressed column-panels:

```
┌──────────────────┐
│  128-byte Header │  magic, version, m, n, nnz, chunk info
├──────────────────┤
│  Chunk Descs     │  per-chunk: col_start, num_cols, nnz, offset, size
├──────────────────┤
│  Forward Data    │  compressed column-panels (delta + rANS coded)
├──────────────────┤
│  Transpose Data  │  compressed row-panels (optional)
├──────────────────┤
│  Footer/Checksum │  integrity verification
└──────────────────┘
```

Each chunk is independently decompressible — the loader can jump to any chunk by offset. See [algorithms/sparsepress.md](../algorithms/sparsepress.md) for the full format specification.
