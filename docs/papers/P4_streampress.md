# P4: StreamPress — A Streaming Format for Out-of-Core Matrix Factorization

**Target Venue**: SoftwareX  
**Type**: Software/format paper  
**Estimated Length**: 10–14 pages  

---

## Abstract (Draft)

We present StreamPress (SPZ), a column-oriented compressed sparse matrix format
designed for streaming out-of-core matrix factorization. SPZ v2 uses chunked
rANS entropy coding to achieve 10–20× compression over standard formats (MatrixMarket,
HDF5) while enabling random-access decompression at the column-chunk level.
This architecture allows NMF to process matrices far exceeding available RAM by
streaming column chunks through a fixed-memory processing pipeline. We describe
the format specification (v2 sparse with gap+value encoding, v3 dense with
byte-shuffled float compression), the auto-dispatch loader that selects optimal
decompression and transposition strategies at runtime, and the column-chunked
streaming NMF algorithm that maintains exact equivalence to in-memory factorization.
Our flagship benchmark demonstrates `nmf(geo_spz_path, k=64, loss="nb")` on a
reprocessed GEO single-cell corpus running on HPC without user configuration,
achieving near-linear scaling with corpus size.

---

## Format Specification Summary

### SPZ v2: Compressed Sparse Matrices

**File Layout**:
- 128-byte header: magic, version, dimensions (m, n, nnz), chunk config, section offsets
- Chunk descriptors: 48 bytes each (col_start, num_cols, nnz, stream offsets/sizes)
- Data section: concatenated gap + value streams per chunk
- Optional transpose section: pre-stored CSC(A^T) for transpose-heavy operations
- Metadata: row/column names, permutation arrays
- Footer: CRC32 checksum, chunk count, magic sentinel

**Encoding Pipeline**:
1. Row indices → per-column delta gaps → rANS entropy coding (symbols 0–254, overflow escaping)
2. Values → type-dependent encoding:
   - Integers (UINT8/16/32): direct rANS
   - Floats (FLOAT32/16/64): byte-shuffle → per-stream rANS
   - Quantized (QUANT8): affine quantization (min/max per chunk) → rANS

**Compression Ratios** (typical scRNA-seq):

| Format | Size | Compression vs MM |
|--------|------|-------------------|
| MatrixMarket (.mtx.gz) | 100% (baseline) | 1× |
| HDF5 (gzip) | ~60% | 1.7× |
| SPZ v2 (UINT8) | ~5–10% | 10–20× |
| SPZ v2 (QUANT8) | ~3–5% | 20–33× |

### SPZ v3: Dense Matrix Compression (Planned)

- Byte-shuffled float encoding for dense matrices
- Block-based tiling for cache-efficient decompression
- Same chunked random-access semantics as v2

---

## Loader Architecture

### Auto-Dispatch Algorithm

The SPZ loader automates format detection and decompression strategy:

```
load(path):
  1. Read header → detect version (v2/v3), value type, dimensions
  2. Choose decompression target:
     - If matrix fits in RAM: full decode to Eigen::SparseMatrix (CSC)
     - If matrix > RAM but streaming supported: return lazy-load handle
     - If transpose needed: check for pre-stored transpose section
  3. Choose value conversion:
     - UINT8/16/32 → native integer or float32 cast
     - QUANT8 → dequantize with per-chunk affine parameters
     - FLOAT16 → fp16→fp32 widening
  4. Return matrix or streaming handle
```

### Streaming Handle

The streaming handle exposes a column-chunk iterator:

```cpp
class SPZStreamReader {
  void open(const std::string& path);
  ChunkData next_chunk();   // decode next chunk of ~256 columns
  bool has_more() const;
  void reset();             // seek back to first chunk
};
```

Each `next_chunk()` call:
1. Reads the next chunk descriptor
2. Decompresses gap + value streams (rANS decode)
3. Reconstructs CSC arrays for the chunk
4. Returns column pointers, row indices, values

Memory usage: O(chunk_size × m) regardless of total matrix size.

---

## Column-Chunked Streaming NMF

### Algorithm

```
StreamingNMF(spz_path, k, maxit):

1. Open SPZ stream reader
2. Initialize H randomly (k × n)
3. For iter = 1, ..., maxit:
   a. H-update pass (fix W, solve for H):
      - Compute Gram: G_W = W^T W    [k×k, in-memory]
      - Stream through columns:
        For each chunk (cols j1..j2):
          A_chunk = stream.next_chunk()
          B_chunk = W^T A_chunk        [compute RHS for chunk]
          For each col j in chunk:
            Solve NNLS: G_W h_j = b_j  [per-column NNLS]
        stream.reset()
   b. W-update pass (fix H, solve for W):
      - Compute Gram: G_H = H H^T    [k×k, in-memory]
      - Stream through columns (need A^T, use transpose section or re-stream):
        For each chunk:
          A_chunk = stream.next_chunk()
          Accumulate: B_W += A_chunk H_chunk^T  [k×m partial RHS]
        Solve NNLS batch for W using accumulated B_W and G_H
        stream.reset()
   c. Normalize: d = colnorms(W); W = W/d
   d. Compute loss (requires another streaming pass, or use Gram trick)
```

### Key Properties

- **Exact equivalence**: Produces the same result as in-memory NMF (no approximation)
- **Memory bounded**: O(mk + nk + chunk_size × m) — independent of total matrix nnz
- **I/O efficient**: Sequential chunk reads; no random access to data section
- **Distribution support**: IRLS weights computed per-chunk, accumulated into Gram

### Loss Computation (Gram Trick)

Instead of materializing the full reconstruction, loss is:

$$\|A - WdH\|_F^2 = \text{tr}(A^T A) - 2 \cdot \text{tr}(B_H^T H) + \text{tr}(G_W \cdot H H^T)$$

- $\text{tr}(A^T A)$: precomputed during first streaming pass (sum of squared values)
- $\text{tr}(B_H^T H)$: accumulated as RHS is computed
- $\text{tr}(G_W \cdot H H^T)$: O(k²) from in-memory matrices

No additional streaming pass needed for loss.

---

## Flagship Benchmark

### GEO Reprocessed Single-Cell Corpus

**Goal**: Demonstrate `nmf(geo_spz_path, k=64, loss="nb")` on a large GEO corpus.

**Dataset**: Concatenated scRNA-seq datasets from GEO, reprocessed into a single
SPZ file. Target: 30,000 genes × 1,000,000 cells (30B potential entries,
~5% nonzero = 1.5B nnz).

**Setup**:
- SPZ file size: ~15 GB (with UINT8 quantization)
- Available RAM: 128 GB (bigmem node) — matrix doesn't fit in-memory as dense
- `nmf(geo_spz_path, k=64, loss="nb", maxit=50)`
- No user configuration needed — auto-detection of streaming mode

**Expected Results**:
- Wall time: ~2–4 hours for 50 iterations
- Peak memory: < 4 GB (bounded by W, H, and one chunk)
- Convergence: comparable to in-memory baseline on smaller datasets

### Scaling Benchmarks

**Benchmark 1: Streaming vs In-Memory**
- Dataset sizes: 10k, 50k, 100k, 500k, 1M cells
- Compare: in-memory (up to RAM limit) vs streaming
- Metric: wall time, peak RSS, final loss

**Benchmark 2: Compression Ratio vs Load Speed**
- Value types: UINT8, UINT16, FLOAT32, QUANT8
- Metric: file size, decompression throughput (GB/s), NMF wall time

**Benchmark 3: Chunk Size Sensitivity**
- chunk_cols ∈ {64, 128, 256, 512, 1024}
- Metric: wall time (I/O vs compute tradeoff)

---

## Figure List

1. **Figure 1**: SPZ v2 file layout diagram (header, chunks, data, transpose, footer)
2. **Figure 2**: Encoding pipeline (raw indices → gaps → rANS)
3. **Figure 3**: Compression ratio comparison (SPZ vs MTX.gz vs HDF5 vs Parquet)
4. **Figure 4**: Streaming NMF memory trace (flat line vs in-memory growth)
5. **Figure 5**: Scaling: wall time vs number of cells (streaming vs in-memory)
6. **Figure 6**: Chunk size sensitivity analysis
7. **Table 1**: Compression ratios across value types
8. **Table 2**: Streaming NMF performance on GEO corpus

---

## Paper Outline

### 1. Introduction (1.5 pages)
- Growth of single-cell datasets: millions of cells
- Limitations of in-memory factorization
- Contribution: streaming format + out-of-core NMF

### 2. SPZ Format Specification (3 pages)
- File layout and encoding methods
- rANS entropy coding primer
- Compression analysis

### 3. Loader and Auto-Dispatch (1.5 pages)
- Format detection and strategy selection
- Streaming handle API
- Transpose handling

### 4. Streaming NMF Algorithm (2.5 pages)
- Column-chunked update derivation
- Gram trick for loss
- Distribution support via weighted streaming

### 5. Benchmarks (3 pages)
- Flagship GEO benchmark
- Scaling, compression, chunk sensitivity

### 6. Discussion (1.5 pages)
- Comparison with other streaming approaches (mini-batch, online NMF)
- Future: v3 dense format, GPU streaming, distributed MPI streaming

---

## Reproducibility

- Code: `benchmarks/harness/suites/nmf_streaming.R`
- Format tools: `sp_read()`, `sp_write()`, `sp_info()`
- GEO corpus: Download script + SPZ conversion pipeline provided
