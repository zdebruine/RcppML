# RcppML GPU Integration

## Architecture (Unified — 2026-02)

```
R    nmf(A, k)  →  .nmf_dispatch()  →  GPU path / CPU path
                                         │
                                    .try_gpu_dispatch()
                                         │
                                    .gpu_nmf_sparse()  (R/gpu_backend.R)
                                         │
                                    .C("rcppml_gpu_nmf_unified_*")
                                         │
                                    gpu_bridge.cu  (extern "C")
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
              Unified NMF          Unified CV NMF     Bipartition/Dclust
        fit_gpu_unified.cuh    fit_cv_gpu_unified.cuh  gpu_bipartition.cuh
                    │                    │              gpu_dclust.cuh
                    └────────┬───────────┘              gpu_k2.cuh
                             │
                 Shared GPU primitives:
                 gpu_gram, gpu_rhs, gpu_nnls,
                 gpu_loss, gpu_fused_cv,
                 gpu_mixed_precision, gpu_types
```

**Note:** The legacy dispatch layer (`nmf_dispatch_gpu.cuh`, `nmf_gpu.cuh`,
`nmf_streaming_gpu.cuh`, `gpu_memory.cuh`) was removed in
February 2026. All NMF now routes through the unified path.

## Build

The GPU library is compiled separately from the R package:

```bash
# On a GPU node with CUDA loaded
module load cuda/12.8.1

cd src/
make -f Makefile.gpu         # Build RcppML_gpu.so
make -f Makefile.gpu install # Copy to inst/lib/
```

## Usage

```r
library(RcppML)

# Check GPU availability
gpu_available()   # TRUE if CUDA GPUs detected
gpu_info()        # Device details (name, memory)

# Automatic dispatch (default)
result <- nmf(A, k = 16)  # Uses GPU if available and beneficial

# Force GPU
options(RcppML.gpu = TRUE)
result <- nmf(A, k = 16)

# Force CPU
options(RcppML.gpu = FALSE)
result <- nmf(A, k = 16)
```

## Auto-dispatch Rules

The GPU path is selected when ALL of:
1. `options(RcppML.gpu)` is `TRUE` or `"auto"` (default: `"auto"`)
2. Input is sparse (`dgCMatrix`) or dense (`matrix`)
3. GPU library is loadable and GPUs detected
4. In auto mode: nnz >= 100K or n >= 5000
5. No unsupported features: graph regularization, masks, IRLS loss

## Files

### R Layer
- `R/gpu_backend.R` — GPU detection, library loading, `.gpu_nmf_sparse()`
- `R/nmf_thin.R` — Auto CPU/GPU dispatch logic

### C Bridge
- `src/gpu_bridge.cu` — C-callable functions (via R's `.C()` FFI)
- `src/gpu_stubs.cpp` — Non-CUDA fallback stubs
- `src/Makefile.gpu` — nvcc compilation rules

### GPU Headers (`inst/include/RcppML/gpu/`)

#### Alive — Unified NMF
- `gpu_types.cuh` — GPUContext, memory abstractions
- `gpu_gram.cuh` — SYRK Gram computation
- `gpu_rhs.cuh` — SpMM right-hand side
- `gpu_nnls.cuh` — 3-tier NNLS solver
- `gpu_loss.cuh` — Gram-trick loss
- `gpu_mixed_precision.cuh` — fp16 shadow copies
- `gpu_fused_cv.cuh` — Fused CV kernels + shared utilities
- `gpu_cv_rng.cuh` — CV mask RNG
- `gpu_cv_delta.cuh` — CV Gram correction deltas
- `nmf_cv_gpu.cuh` — CV NMF driver

#### Alive — Bipartition/Dclust
- `gpu_bipartition.cuh` — GPU bipartitioning
- `gpu_dclust.cuh` — GPU divisive clustering
- `gpu_k2.cuh` — k=2 closed-form NNLS

#### Alive — Infrastructure
- `gpu_multi.cuh` — GPU detection (`detect_gpus()`, `GPUDeviceInfo`)

#### Removed (2026-02)
- ~~`nmf_dispatch_gpu.cuh`~~ — Old dispatch routing layer (411 lines)
- ~~`nmf_gpu.cuh`~~ — Old monolithic NMF driver (336 lines)
- ~~`nmf_streaming_gpu.cuh`~~ — Old streaming NMF driver (565 lines)
- ~~`gpu_memory.cuh`~~ — Panel planning for streaming (380 lines)
- Dead portions of `gpu_multi.cuh` removed

### Unified Algorithm Headers (`inst/include/RcppML/algorithms/nmf/`)
- `fit_gpu_unified.cuh` — Unified standard NMF GPU driver (sparse input)
- `fit_gpu_dense_unified.cuh` — Unified standard NMF GPU driver (dense input, cuBLAS GEMM)
- `fit_cv_gpu_unified.cuh` — Unified CV NMF GPU driver

## Benchmarking

```bash
# Submit GPU benchmark job
sbatch benchmarks/slurm/gpu_h100.sbatch

# Or run interactively
srun --partition=gpu --gres=gpu:nvidia_h100_nvl:2 --time=02:00:00 bash
Rscript benchmarks/R/gpu_cpu_benchmark.R
```

## Hardware Targets

- **NVIDIA H100 NVL** (sm_90): 2× per node, 94GB HBM3e each, NVLink P2P
- **NVIDIA V100S** (sm_70): g001-g004
- **NVIDIA RTX 8000** (sm_75): g005-g006
