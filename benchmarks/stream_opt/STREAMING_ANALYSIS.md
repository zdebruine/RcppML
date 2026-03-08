# Streaming NMF Performance Analysis

**Date**: 2026-03-07  
**Node**: c004 (cpu partition), 4 OMP threads  
**R**: 4.5.2, RcppML with StreamTimer instrumentation

## Architectural Principles

The streaming path exists for datasets **too large to fit in memory**. Design:

1. **Compute-bound, I/O-hidden**: CPU handles I/O (SPZ decompression, staging),
   GPU handles compute (NNLS, SpMM, Gram). Prefetch overlaps I/O with compute
   so the system is always compute-bound with I/O hidden behind it.

2. **Both orientations pre-built**: The SPZ file stores both forward (A) and
   transpose (A^T) chunks via `sp_write(x, path, include_transpose=TRUE)`.
   A distributed-transpose streaming operation produces these before NMF begins.
   Either orientation can be streamed on demand (H-update uses forward,
   W-update uses transpose).

3. **Adaptive chunk sizing**: Chunk size should be auto-tuned to fill a good
   proportion of GPU memory with a single chunk. This maximizes compute
   efficiency per kernel launch and amortizes PCIe transfer overhead. Currently
   SPZ files use a fixed 256-column chunk size — this needs to become adaptive
   based on available GPU VRAM (or CPU memory budget).

4. **CPU orchestrates, GPU computes**: CPU reads/decompresses SPZ chunks and
   stages them for GPU transfer. GPU performs all heavy linear algebra: SpMM
   for RHS, Cholesky/CD for NNLS, rank-k Gram updates. W stays device-resident.

5. **No in-memory caching**: Never assume data fits in RAM. Working memory is
   O(m×k + n×k + max_chunk_nnz), not O(total_nnz).

## Executive Summary

Streaming NMF currently runs **5–16x slower** than in-memory NMF. The overhead
is dominated by two bottlenecks whose relative weight shifts with rank:

| Rank (k) | I/O % | Compute % | Loss % | Overhead |
|----------|-------|-----------|--------|----------|
| 8        | 71%   | 15%       | 10%    | 15–16x  |
| 16       | 51%   | 27%       | 17%    | 10–12x  |
| 32       | 17%   | 45%       | 31%    | 7–9x    |

At low rank, **I/O dominates** (chunk reads from SPZ). At high rank, **compute
dominates** (RHS accumulation and loss evaluation), revealing that the streaming
compute kernels are ~2–3x slower than the in-memory fused kernels.

## Critical Bug Found & Fixed

**`config.loss.robust_delta` was set to `huber_delta` (default 1.0) instead of 0**
in `Rcpp_nmf_streaming_spz()`. This made `requires_irls()` return `true` for
all streaming calls, forcing the **per-column** NNLS path (with Gram correction
per column) instead of the **batch** path. Result: 464x overhead → 3–16x after fix.

## Detailed Profile Breakdown

### pbmc3k (13,714 × 2,700, 2.3M nnz, 6.2% dense)

| Section          | k=8 (ms) | %     | k=16 (ms) | %     | k=32 (ms) | %     |
|------------------|----------|-------|-----------|-------|-----------|-------|
| total_iter       | 1090     | 100   | 1383      | 100   | 2220      | 100   |
| w_transpose_read | 329      | 30.2  | 232       | 16.8  | 52        | 2.3   |
| chunk_read       | 139      | 12.7  | 51        | 3.7   | 53        | 2.4   |
| **I/O total**    | **468**  | **42.9** | **283** | **20.5** | **105**  | **4.7** |
| chunk_rhs        | 170      | 15.6  | 295       | 21.3  | 604       | 27.2  |
| w_rhs            | 176      | 16.1  | 301       | 21.8  | 529       | 23.8  |
| **RHS total**    | **346**  | **31.8** | **596** | **43.1** | **1133** | **51.0** |
| loss             | 216      | 19.8  | 383       | 27.7  | 818       | 36.8  |
| chunk_nnls       | 2        | 0.2   | 4         | 0.3   | 11        | 0.5   |
| w_nnls           | 9        | 0.8   | 21        | 1.5   | 63        | 2.8   |
| chunk_gram       | 2        | 0.2   | 6         | 0.4   | 12        | 0.5   |
| gram_accumulate  | 0.3      | 0.0   | 1         | 0.1   | 3         | 0.1   |
| scaling          | 0.5      | 0.0   | 1         | 0.1   | 2         | 0.1   |

### synth_20k (10,000 × 20,000, 10M nnz, 5% dense)

| Section          | k=8 (ms) | %     | k=16 (ms) | %     | k=32 (ms) | %     |
|------------------|----------|-------|-----------|-------|-----------|-------|
| total_iter       | 9911     | 100   | 9817      | 100   | 11833     | 100   |
| w_transpose_read | 4142     | 41.8  | 3454      | 35.2  | 1899      | 16.0  |
| chunk_read       | 3064     | 30.9  | 1778      | 18.1  | 79        | 0.7   |
| **I/O total**    | **7206** | **72.7** | **5232** | **53.3** | **1978** | **16.7** |
| chunk_rhs        | 741      | 7.5   | 1281      | 13.1  | 2821      | 23.8  |
| w_rhs            | 775      | 7.8   | 1458      | 14.9  | 3063      | 25.9  |
| **RHS total**    | **1516** | **15.3** | **2739** | **27.9** | **5884** | **49.7** |
| loss             | 977      | 9.9   | 1688      | 17.2  | 3723      | 31.5  |
| chunk_nnls       | 14       | 0.1   | 30        | 0.3   | 73        | 0.6   |
| w_nnls           | 7        | 0.1   | 15        | 0.2   | 44        | 0.4   |

### synth_10k (5,000 × 10,000, 1.5M nnz, 3% dense)

| Section          | k=8 (ms) | %     | k=16 (ms) | %     | k=32 (ms) | %     |
|------------------|----------|-------|-----------|-------|-----------|-------|
| total_iter       | 1562     | 100   | 1637      | 100   | 1865      | 100   |
| w_transpose_read | 638      | 40.9  | 558       | 34.1  | 375       | 20.1  |
| chunk_read       | 482      | 30.9  | 284       | 17.3  | 28        | 1.5   |
| **I/O total**    | **1120** | **71.7** | **841** | **51.4** | **403** | **21.6** |
| chunk_rhs        | 113      | 7.2   | 200       | 12.2  | 371       | 19.9  |
| w_rhs            | 117      | 7.5   | 209       | 12.8  | 399       | 21.4  |
| **RHS total**    | **230**  | **14.7** | **409** | **25.0** | **770** | **41.3** |
| loss             | 157      | 10.0  | 282       | 17.2  | 536       | 28.7  |

### In-memory reference (synth_20k)

| Section          | k=8 (ms) | %     | k=16 (ms) | %     | k=32 (ms) | %     |
|------------------|----------|-------|-----------|-------|-----------|-------|
| total            | 337      | 100   | 590       | 100   | 1093      | 100   |
| fused_rhs_nnls_H | 121      | 35.8  | 192       | 32.6  | 344       | 31.5  |
| fused_rhs_nnls_W | 108      | 32.1  | 192       | 32.6  | 347       | 31.8  |
| loss             | 101      | 30.1  | 185       | 31.4  | 341       | 31.2  |
| scaling          | 4        | 1.2   | 12        | 2.1   | 38        | 3.5   |
| gram_*           | 3        | 0.9   | 8         | 1.4   | 23        | 2.1   |

## Key Observations

### 1. I/O dominates at low rank, diminishes at high rank

At k=8, I/O is 71–73% of streaming time. By k=32, it drops to 5–22%.

**Why**: Forward and transpose chunk reads are O(nnz) regardless of k.
RHS and loss computation are O(nnz × k). As k grows, compute time grows
linearly while I/O stays constant. The double-buffered prefetch overlaps
I/O with compute — at k=32 the prefetch completes before the compute kernel
finishes, making chunk_read nearly free (0.7% for synth_20k at k=32).

**Implication**: I/O optimization matters most for low-rank factorizations
(k ≤ 16). At high rank, the compute kernels are the bottleneck.

### 2. Transpose read is 2× more expensive than forward read

`w_transpose_read` is consistently ~1.3–2× more expensive than `chunk_read`
for the same dataset. This is because the SPZ transpose store must be
decompressed separately and may have different compression characteristics
(row-major vs column-major access patterns).

### 3. Loss computation is unexpectedly expensive (10–37%)

The streaming path computes loss at every nonzero per iteration (O(nnz × k)).
In-memory NMF also pays this cost (~30% of time), but the streaming loss
is **not overlapped** with I/O — it runs after all forward chunks are
processed but requires the full H matrix to be assembled.

### 4. RHS kernels are ~2x slower than in-memory fused kernels

For synth_20k at k=32:
- In-memory: fused_rhs_nnls_H = 344ms, fused_rhs_nnls_W = 347ms
- Streaming: chunk_rhs = 2821ms, w_rhs = 3063ms (plus loss = 3723ms)

The in-memory "fused" kernels combine RHS accumulation + NNLS solve in one
pass. The streaming path does them separately. The streaming RHS also computes
column-by-column with SpMat iterators, while in-memory uses dense Eigen
matrix operations.

However, the streaming chunk_rhs+chunk_nnls (2894ms) vs in-memory
fused_rhs_nnls_H (344ms) is an **8.4x** gap, which is larger than just
fused vs unfused. The streaming path accumulates RHS in parallel over
columns within a chunk, but each chunk is small — overhead from launching
OMP parallel regions for each small chunk may dominate.

### 5. NNLS solve is <3% of time (Cholesky is fast)

The actual NNLS solve (Cholesky clip batch) takes negligible time even at
k=32. The bottleneck is RHS computation (W'A for H-update, A'H for
W-update), not the solve itself.

### 6. Overhead decreases with rank

| Dataset   | k=8    | k=16   | k=32   |
|-----------|--------|--------|--------|
| pbmc3k    | 4.9x   | 7.1x   | 6.3x   |
| synth_20k | 16.5x  | 11.0x  | 8.7x   |
| synth_10k | 15.4x  | 12.4x  | 7.5x   |

For pbmc3k the overhead increases slightly from k=8 to k=16 because I/O is
already small; the compute overhead dominates. For larger datasets, the
overhead reliably decreases with k because I/O is amortized.

## Optimization Hypotheses

### H1: Overlap loss computation with W-update I/O (est. 10–37% reduction)

**Observation**: Loss is computed after ALL forward chunks, then W-update
starts with transpose I/O. Loss could be computed per-chunk during the
forward pass and accumulated, overlapping the loss computation with the
first transpose chunk reads.

**Current state**: Loss IS computed per-chunk already. But the loss timer
shows it as a large separate cost. The issue is that loss is computed
inline during the forward pass — it's not overlapping with anything.

**Hypothesis**: Move loss computation to a separate thread that processes
completed H_panel chunks while the next chunk is prefetched and the NNLS
solve runs. This creates a 3-stage pipeline: read → solve → loss.

### H2: Fuse RHS computation with iteration (est. 30–50% reduction)

**Observation**: chunk_rhs and w_rhs iterate over sparse nonzeros to
compute W'A and A'H. These are the same nonzero entries visited by loss.

**Hypothesis**: Fuse the loss computation into the RHS kernel by computing
both the RHS vector and the reconstruction error in the same sparse
iteration. This eliminates the redundant nnz traversal. For the forward
pass, compute both B_panel = W_T * A_panel (RHS) and loss at the same time.

### H3: Reduce transpose read cost via CSR caching (est. 15–42% reduction)

**Observation**: w_transpose_read is the single most expensive section at
low rank (30–42% of total). Each iteration re-reads and decompresses
the same transpose data from SPZ.

**Hypothesis**: Cache decompressed transpose chunks in memory after the
first iteration. If total decompressed size fits in a memory budget
(e.g., 2× the current chunk buffer), re-use cached chunks on subsequent
iterations. This would make w_transpose_read ~0 after iteration 1.

Alternatively: use the forward chunks and transpose in-memory rather than
reading a separate transpose store. For an m×n matrix, transposing the
CSC chunk to CSR is O(nnz_chunk) and may be cheaper than decompressing
a separate SPZ transpose file.

### H4: Increase chunk size to amortize OMP overhead (est. 10–20% reduction)

**Observation**: Many small chunks means many OMP parallel region launches.
Each launch has overhead (thread wake, barrier sync). If chunk_cols is too
small, OMP overhead per nnz is high.

**Hypothesis**: Increase default chunk_cols to reduce the number of chunks.
For synth_20k with 20k columns, if chunks are 1024 cols, that's ~20 chunks.
Increasing to 4096 would give ~5 chunks, reducing OMP launch overhead 4×.

### H5: ~~Eliminate per-iteration SPZ re-reads~~ — **REJECTED**

**Observation**: Every NMF iteration re-reads ALL chunks from disk.

**Hypothesis (rejected)**: Cache all chunks in memory after the first read.

**Why rejected**: Caching all chunks in RAM directly contradicts the core
purpose of streaming NMF — handling datasets too large for memory. If the
data fits in RAM, use in-memory NMF. The streaming path must remain truly
out-of-core with O(max_chunk_nnz) working memory, not O(total_nnz).

**Correct approach**: Make I/O invisible via **double-buffered async prefetch**
(already implemented in `fit_chunked.hpp`). When compute time per chunk
exceeds I/O time (which happens naturally at higher rank), the prefetch
completes before the compute kernel finishes, making I/O effectively free.
At low rank where I/O dominates, the solution is larger chunks (H4) to
increase compute-per-chunk, not caching. CachingLoader has been removed.

## Prioritized Action Plan

1. **H2 (Gram-trick loss)**: Highest impact — eliminates the separate O(nnz×k) loss
   traversal by using O(k²) Gram-trick: `loss = trAtA - 2*cross_term + recon_norm`.
   `trAtA` is O(nnz) but computed once (constant). `cross_term` piggybacks on
   the RHS kernel. `recon_norm` is O(k²) from Gram matrices already available.
   Expected to eliminate ~30% of iteration time.

2. **H4 (Larger chunks)**: Increases compute-per-chunk, making the async
   prefetch overlap more effective and amortizing OMP launch overhead.
   Test with 2048 and 4096 chunk sizes.

3. **H3 (In-memory transpose)**: Transpose forward chunks in-memory instead
   of reading a separate SPZ transpose store. O(nnz_chunk) per chunk but
   avoids the expensive transpose decompression (30–42% of time at low k).

4. **H1 (Pipeline stages)**: Lower priority if H2 is implemented, since
   loss computation becomes nearly free (O(k²) instead of O(nnz×k)).

## Optimization Results

### Optimizations Applied (cumulative)

1. ~~**Chunk caching (H5)**~~: **Removed.** Caching all chunks in RAM defeats
   the purpose of streaming. I/O overlap is handled by double-buffered
   async prefetch in `fit_chunked.hpp`.

2. **W_T_d pre-scaling**: Pre-compute `W_T_d = diag(d) * W_T` once per
   iteration instead of multiplying per-nnz in the loss kernel.

3. **Eigen-vectorized compute kernels (H2 partial)**: Replaced manual scalar
   pointer loops in chunk_rhs, w_rhs, and loss with Eigen expressions:
   - `b_col.noalias() += val * W_T.col(row)` (SIMD-vectorized column accumulation)
   - `bw_col.noalias() += val * H.col(row)` (same)
   - `W_T_d.col(row).dot(h_col)` (SIMD-vectorized dot product)

4. **Gram-trick loss (H2 full)**: Replaced O(nnz×k) per-iteration loss with
   O(k²) Gram-trick: `loss = trAtA - 2*cross_term + recon_norm`. `trAtA`
   computed once (first iter), `cross_term` piggybacks on RHS, `recon_norm`
   from Gram matrices already available.

### Overhead Progression (streaming time / in-memory time)

| Dataset   | k  | Pre-Gram-Trick | Post-Gram-Trick (no cache) |
|-----------|-----|---------------|---------------------------|
| pbmc3k    | 8   | 4.89x         | **4.71x**                 |
| pbmc3k    | 16  | 7.10x         | **5.97x**                 |
| pbmc3k    | 32  | 6.28x         | **3.44x**                 |
| synth_20k | 8   | 16.46x        | **16.90x**                |
| synth_20k | 16  | 10.95x        | **12.16x**                |
| synth_20k | 32  | 8.69x         | **7.62x**                 |
| synth_10k | 8   | 15.41x        | **15.55x**                |
| synth_10k | 16  | 12.38x        | **11.90x**                |
| synth_10k | 32  | 7.45x         | **6.94x**                 |

*Note: Pre-Gram-Trick had caching enabled (which hid I/O after iter 1).
Post-Gram-Trick has NO caching — true streaming. The Gram-trick loss
is nearly free (0.1–0.4%) but I/O now dominates since it's not cached.*

### Post-Gram-Trick Profile — No Caching (synth_20k)

| Section          | k=8 (ms)  | %     | k=16 (ms) | %     | k=32 (ms) | %     |
|------------------|-----------|-------|-----------|-------|-----------|-------|
| total_iter       | 9931      | 100   | 9831      | 100   | 9898      | 100   |
| w_transpose_read | 4535      | 45.7  | 4369      | 44.4  | 3812      | 38.5  |
| chunk_read       | 4363      | 43.9  | 4101      | 41.7  | 3614      | 36.5  |
| **I/O total**    | **8898**  | **89.6** | **8469** | **86.1** | **7426** | **75.0** |
| chunk_rhs        | 349       | 3.5   | 596       | 6.1   | 1020      | 10.3  |
| w_rhs            | 369       | 3.7   | 575       | 5.8   | 1155      | 11.7  |
| **Compute total**| **718**   | **7.2** | **1171** | **11.9** | **2175** | **22.0** |
| loss             | 15        | 0.1   | 19        | 0.2   | 23        | 0.2   |
| chunk_nnls       | 14        | 0.1   | 33        | 0.3   | 80        | 0.8   |
| w_nnls           | 7         | 0.1   | 17        | 0.2   | 46        | 0.5   |

### Post-Gram-Trick Profile — No Caching (pbmc3k)

| Section          | k=8 (ms)  | %     | k=16 (ms) | %     | k=32 (ms) | %     |
|------------------|-----------|-------|-----------|-------|-----------|-------|
| total_iter       | 1119      | 100   | 1189      | 100   | 1209      | 100   |
| w_transpose_read | 446       | 39.9  | 446       | 37.5  | 397       | 32.8  |
| chunk_read       | 435       | 38.9  | 378       | 31.8  | 243       | 20.1  |
| **I/O total**    | **881**   | **78.7** | **824** | **69.3** | **640** | **53.0** |
| chunk_rhs        | 85        | 7.6   | 137       | 11.5  | 197       | 16.3  |
| w_rhs            | 84        | 7.5   | 98        | 8.2   | 143       | 11.8  |
| **Compute total**| **169**   | **15.1** | **235** | **19.8** | **340** | **28.1** |
| loss             | 3         | 0.3   | 4         | 0.3   | 7         | 0.6   |

### In-Memory Reference (synth_20k)

| Section          | k=8 (ms) | %     | k=16 (ms) | %     | k=32 (ms) | %     |
|------------------|----------|-------|-----------|-------|-----------|-------|
| total            | 590      | 100   | 810       | 100   | 1300      | 100   |
| fused_rhs_nnls_H | 112      | 35.0  | 165       | 30.8  | 320       | 30.9  |
| fused_rhs_nnls_W | 101      | 31.7  | 178       | 33.2  | 328       | 31.6  |
| loss             | 100      | 31.3  | 173       | 32.4  | 329       | 31.7  |
| scaling          | 4        | 1.1   | 11        | 2.0   | 37        | 3.6   |
| gram_*           | 3        | 0.6   | 8         | 1.5   | 23        | 1.8   |

### Analysis: Gram-Trick Loss Success, I/O Becomes The Bottleneck

**Gram-trick loss eliminated loss computation as a bottleneck completely:**
- Before: loss was 10–37% of streaming time (O(nnz×k) per iteration)
- After: loss is 0.1–0.6% of streaming time (O(k²) per iteration)

**The remaining gap is almost entirely I/O** (SPZ decompression):

| Dataset   | k  | I/O %  | Compute % | I/O Time | Compute Time |
|-----------|-----|--------|-----------|----------|--------------|
| synth_20k | 8   | 89.6%  | 7.2%      | 8898 ms  | 718 ms       |
| synth_20k | 16  | 86.1%  | 11.9%     | 8469 ms  | 1171 ms      |
| synth_20k | 32  | 75.0%  | 22.0%     | 7426 ms  | 2175 ms      |
| pbmc3k    | 8   | 78.7%  | 15.1%     | 881 ms   | 169 ms       |
| pbmc3k    | 32  | 53.0%  | 28.1%     | 640 ms   | 340 ms       |

**Why prefetch doesn't help enough**: With 256-column chunks, each chunk has
very little compute (3.5–11.7% of total). The prefetch overlaps one chunk
of I/O with one chunk of compute, but compute finishes ~3–12× faster than
I/O, so the system is always waiting for I/O.

**Solution: Larger chunks (H4) + adaptive sizing:**
To hide I/O, we need `compute_time(chunk) ≥ IO_time(next_chunk)`. For
synth_20k at k=32: compute = 2175ms / 78 fwd chunks ≈ 28ms/chunk, I/O =
7426ms / 78 chunks ≈ 95ms/chunk. We need chunks ~3.4× larger (≈870 cols)
to overlap compute with I/O at k=32. At k=8 we'd need ~12× larger chunks.

For GPU streaming, chunks should be sized to fill GPU VRAM:
- GPU VRAM budget → max columns per chunk → maximize compute per transfer
- CPU orchestrates: decompress chunk → H2D transfer → GPU compute → prefetch next
### Next Steps: Chunk Sizing & GPU-Driven Architecture

The profiling conclusively shows: **compute is fast, I/O is the bottleneck,
and the Gram-trick loss works perfectly.** The remaining work is:

1. **Adaptive chunk sizing (critical)**: SPZ default of 256 columns is too
   small. SPZ `sp_write()` should support larger `chunk_cols`. At read time,
   the loader should merge consecutive small chunks into larger logical
   chunks that match the compute budget.

2. **GPU-aware chunk sizing**: `compute_panel_cols()` already exists in
   `fit_gpu_streaming.cuh` — it queries `cudaMemGetInfo` and computes the
   optimal panel width for available VRAM. This logic needs to be used
   when the streaming path talks to the GPU, and the CPU side needs an
   analog that considers available RAM and rank.

3. **CPU as I/O orchestrator**: The double-buffered prefetch already serves
   this role. With larger chunks, the async prefetch will have enough time
   to decompress the next chunk while GPU computes on the current one.

4. **In-memory transpose (H3)**: Consider transposing forward chunks on CPU
   instead of reading a separate SPZ transpose store. This halves I/O and
   removes the requirement for `include_transpose=TRUE` in the file.
