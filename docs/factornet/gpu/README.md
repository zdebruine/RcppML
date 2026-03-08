# FactorNet GPU Architecture

## Overview

FactorNet provides CUDA GPU acceleration for all NMF operations. The GPU path implements the same algorithm as the CPU path — only the four primitive operations (Gram, RHS, NNLS, loss) are replaced with GPU kernels. GPU errors are caught and transparently retried on CPU.

## Architecture

```
R: nmf(A, k, resource="gpu")
  → RcppFunctions.cpp: build NMFConfig, call nmf::fit()
    → nmf/fit.hpp: detect GPU, select ResourcePlan::GPU
      ├── Standard NMF  →  fit_gpu.cuh  (sparse) / fit_gpu_dense.cuh (dense)
      ├── CV NMF         →  fit_cv_gpu.cuh
      └── On error       →  fallback to fit_cpu.hpp (CPU)
                               ↑
                          GPU primitives:
                          ├── gram.cuh      (cuBLAS SSYRK)
                          ├── rhs.cuh       (cuSPARSE SpMM / cuBLAS SGEMM)
                          ├── nnls_batch.cuh (custom CUDA CD kernel)
                          ├── loss.cuh      (tree-reduction kernel)
                          └── context.cuh   (streams, handles)
```

### Runtime Bridge (CPU-Only Builds)

When compiled without CUDA (`FACTORNET_HAS_GPU` not defined), GPU operations are dispatched at runtime via `dlsym` to a separately compiled shared library (`RcppML_gpu.so`). This allows the R package to ship a single binary that discovers GPU support at runtime.

```
CPU-only build:
  nmf::fit() → plan=GPU → gpu::bridge_nmf_sparse<Scalar>()
    → dlopen("RcppML_gpu.so") → dlsym("rcppml_gpu_nmf_unified_*")
    → execute in GPU library
    → on failure: fallback to CPU
```

## GPU Kernel Inventory

### Primitive Kernels (`primitives/gpu/`)

| Kernel | File | Operation | Implementation |
|--------|------|-----------|---------------|
| Gram | `gram.cuh` | G = H·H^T (k×k) | cuBLAS `cublasSsyrk` |
| RHS (sparse) | `rhs.cuh` | B = H·A (k×m) | cuSPARSE `cusparseSpMM` |
| RHS (dense) | `rhs.cuh` | B = H·A (k×m) | cuBLAS `cublasSgemm` |
| NNLS batch | `nnls_batch.cuh` | Per-column CD solve | Custom CUDA, 1 thread-block per column |
| NNLS IRLS | `nnls_batch_irls.cuh` | Weighted CD for non-MSE | Custom CUDA with per-element weights |
| NNLS ZI-IRLS | `nnls_batch_zi_irls.cuh` | Zero-inflation aware IRLS | Custom CUDA with ZI posterior weights |
| Cholesky+clip | `cholesky_clip.cuh` | Cholesky factorization + clip | cuSOLVER / custom |
| Loss | `loss.cuh` | Gram-trick loss reduction | Tree-reduction kernel |
| Context | `context.cuh` | Stream/handle management | cuBLAS + cuSPARSE handle pool |

### Algorithm Drivers (`nmf/`)

| Driver | File | Input | Features |
|--------|------|-------|----------|
| Standard sparse | `fit_gpu.cuh` | CSC sparse | Full feature set |
| Standard dense | `fit_gpu_dense.cuh` | Dense matrix | Full feature set |
| CV sparse | `fit_cv_gpu.cuh` | CSC sparse | Lazy mask, delta-G correction |

### CV-Specific GPU Kernels

| Kernel | Purpose |
|--------|---------|
| `cv_kernels.cuh` | Per-column delta-G Gram correction on GPU |
| `fused_cv.cuh` | Fused CV RHS + mask application |

### Clustering Kernels (`gpu/`)

| Kernel | File | Purpose |
|--------|------|---------|
| Bipartition | `gpu_bipartition.cuh` | GPU-accelerated rank-2 NMF bipartitioning |
| Dclust | `gpu_dclust.cuh` | GPU divisive clustering |
| K=2 NNLS | `gpu_k2.cuh` | Closed-form 2×2 NNLS (fast path for bipartition) |

### Mixed Precision

| Kernel | File | Purpose |
|--------|------|---------|
| FP16 shadows | `mixed_precision.cuh` | Maintain FP16 shadow copies of W/H for Tensor Core acceleration |

The mixed-precision path keeps FP32 master copies and FP16 shadow copies. Gram and RHS use FP16 via Tensor Cores (2-4× throughput on Volta+), while NNLS and loss use FP32 for numerical stability.

## Memory Management

### Device Memory Layout

All matrices reside in GPU global memory during the NMF iteration:

| Buffer | Size | Lifetime |
|--------|------|----------|
| A (sparse CSC) | `nnz * sizeof(float) + (nnz + n + 1) * sizeof(int)` | Allocated once at start |
| W | k × m × sizeof(float) | Persistent across iterations |
| H | k × n × sizeof(float) | Persistent across iterations |
| G (Gram) | k × k × sizeof(float) | Recomputed each half-iteration |
| B (RHS) | k × max(m, n) × sizeof(float) | Recomputed each half-iteration |

### Transfer Strategy

- **Input A**: Copied host → device once at NMF start
- **W, H**: Initialized on host, transferred to device; copied back at end
- **G, B**: Allocated on device, never transferred (intermediate)
- **Loss scalar**: Single float device → host per convergence check

### cuSPARSE Buffer

cuSPARSE SpMM requires a workspace buffer. Its size is queried via `cusparseSpMM_bufferSize()` and allocated once. The buffer is reused across all RHS calls.

## NNLS Kernel Design

The CD-based NNLS kernel is the most compute-intensive GPU operation. Each column of B corresponds to one NNLS problem of size k.

**Launch configuration:**
- 1 thread-block per column
- Block size = k (or next power of 2, capped at 1024)
- Shared memory: k×k Gram matrix + k-element work vector

**Algorithm per block:**
1. Load Gram row from global → shared memory
2. Load RHS column from global → registers
3. Coordinate descent iterations (all threads cooperate via `__syncthreads()`)
4. Write solution back to H column in global memory

For small k (≤ 32), the kernel is highly efficient — Gram fits in shared memory and CD iterations are register-bound. For larger k, the kernel falls back to global memory access patterns.

## Auto-Dispatch Rules

The GPU path is selected when ALL conditions hold:

1. `resource_override` is `"gpu"` or `"auto"` (default)
2. GPU hardware detected and healthy (`cudaGetDeviceCount() > 0`)
3. Sufficient VRAM for the problem
4. In `"auto"` mode: heuristic check (nnz ≥ 100K or n ≥ 5000)

Features that **prevent** GPU dispatch in auto mode:
- Graph Laplacian regularization (not yet ported)
- User-supplied mask (`config.mask != nullptr`)

Setting `resource_override = "gpu"` forces the GPU path regardless of heuristics (still falls back on error).

## Build Instructions

```bash
# On a GPU node with CUDA loaded:
module load cuda/12.8.1

cd src/
make -f Makefile.gpu           # Compile RcppML_gpu.so
make -f Makefile.gpu install   # Copy to inst/lib/
```

The GPU library is compiled separately from the R package and loaded at runtime. This avoids requiring CUDA on systems that only use CPU.

## Hardware Targets

| GPU | Architecture | Nodes | VRAM | Notes |
|-----|-------------|-------|------|-------|
| NVIDIA H100 NVL | sm_90 (Hopper) | g051-g052 | 94 GB HBM3e | NVLink P2P, Tensor Cores |
| NVIDIA V100S | sm_70 (Volta) | g001-g004 | 32 GB HBM2 | Tensor Cores |
| NVIDIA RTX 8000 | sm_75 (Turing) | g005-g006 | 48 GB GDDR6 | FP32 focus |

## R Interface

```r
# Check GPU availability
gpu_available()   # TRUE if CUDA GPUs detected
gpu_info()        # Device names, memory, compute capability

# Auto-dispatch (default)
result <- nmf(A, k = 16)  # Uses GPU when beneficial

# Force GPU
result <- nmf(A, k = 16, resource = "gpu")

# Force CPU
result <- nmf(A, k = 16, resource = "cpu")

# Global override
options(RcppML.gpu = TRUE)   # Force GPU for all calls
options(RcppML.gpu = FALSE)  # Force CPU for all calls
options(RcppML.gpu = "auto") # Restore auto-dispatch (default)
```
