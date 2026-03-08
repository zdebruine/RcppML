# FactorNet Architecture

## Overview

FactorNet is a header-only, template-driven C++ library. All code lives under `inst/include/FactorNet/`. The architecture follows a layered design:

```
┌─────────────────────────────────────────────────────────┐
│  R Layer  (nmf_thin.R → RcppFunctions.cpp)              │
├─────────────────────────────────────────────────────────┤
│  Gateway   (nmf/fit.hpp  ·  svd/gateway.hpp  ·  graph/) │
├─────────────────────────────────────────────────────────┤
│  Algorithm (fit_cpu.hpp  ·  fit_cv.hpp  ·  fit_gpu.cuh) │
├─────────────────────────────────────────────────────────┤
│  Features  (L1, L2, L21, angular, graph, guides, ZI)    │
├─────────────────────────────────────────────────────────┤
│  Primitives (gram · rhs · nnls_batch · trace_AtA)        │
│    ├── cpu/  (OpenMP + Eigen)                            │
│    └── gpu/  (cuBLAS + cuSPARSE + custom CUDA)           │
├─────────────────────────────────────────────────────────┤
│  Core      (types · config · result · traits · rng)      │
└─────────────────────────────────────────────────────────┘
```

## Dispatch Flow: R → C++ → Backend

### Step 1: R Entry Point

```
R: nmf(A, k=20, ...)
  → nmf_thin.R validates parameters
  → calls .Call("Rcpp_nmf_full", ...)
```

### Step 2: Rcpp Bridge (`src/RcppFunctions.cpp`)

The bridge function `Rcpp_nmf_full()`:
1. Extracts R S4 `dgCMatrix` slots → raw CSC pointers (`i`, `p`, `x`)
2. Builds `NMFConfig<float>` from R parameters via `build_config_from_params()`
3. Maps sparse data to `Eigen::Map<const SparseMatrix<float>>`
4. Casts `W_init` / `H_init` from double → float
5. Calls `FactorNet::nmf::fit(A, config, W_init, H_init)`
6. Packs `NMFResult<float>` back into an R list

### Step 3: Gateway (`nmf/fit.hpp`)

```cpp
template<typename Scalar, typename MatrixType>
NMFResult<Scalar> nmf::fit(const MatrixType& A, const NMFConfig<Scalar>& config, ...)
{
    // 1. Detect hardware
    auto resources = Resources::detect();  // CPU cores, GPU count, VRAM

    // 2. Select execution plan
    auto plan = select_resources(resources, config);
    // Respects config.resource_override ("auto"/"cpu"/"gpu")
    // Falls back to CPU if GPU unavailable

    // 3. Dispatch by mode
    if (config.is_cv())
        return dispatch_cv(A, config, plan, ...);
    else
        return dispatch_standard(A, config, plan, ...);
}
```

### Step 4: Backend Selection

```
dispatch_standard():
  plan == CPU  →  nmf::nmf_fit<primitives::CPU>(A, config, ...)
  plan == GPU  →  try GPU path → catch → fallback to CPU

dispatch_cv():
  plan == CPU  →  nmf::nmf_fit_cv<primitives::CPU>(A, config, ...)
  plan == GPU  →  try GPU CV path → catch → fallback to CPU
```

GPU errors (OOM, kernel failure) are caught and transparently retried on CPU. The `result.diagnostics` field records which backend was actually used.

### Step 5: Algorithm Loop (`fit_cpu.hpp`)

```
for iter = 0 to max_iter:
    // ── H update (fix W, solve for H) ──
    G_H = gram(W)              // k×k: W^T·W
    B_H = rhs(A^T, W)         // k×n: W^T·A (or fused path)
    apply_features(G_H, B_H)  // L2, angular, graph, L21, guides
    nnls_batch(G_H, B_H, H)   // per-column CD or Cholesky+clip
    apply_bounds(H)            // box constraints, upper_bound

    // ── W update (fix H, solve for W) ──
    G_W = gram(H)              // k×k: H·H^T
    B_W = rhs(A, H)           // k×m: H·A^T
    apply_features(G_W, B_W)
    nnls_batch(G_W, B_W, W)
    apply_bounds(W)

    // ── Normalize ──
    d = normalize_columns(W, H)

    // ── Convergence check ──
    loss = compute_loss(A, W, d, H)
    if |Δloss|/|loss| < tol for `patience` checks:
        break
```

## Resource Detection & Selection

### `Resources::detect()` (`core/resources.hpp`)

- **CPU**: `omp_get_max_threads()` for available cores
- **GPU**: `cudaGetDeviceCount()` + smoke test per device
- **VRAM**: `cudaMemGetInfo()` for available GPU memory

### `select_resources()` Logic

1. If `config.resource_override == "cpu"` → force CPU
2. If `config.resource_override == "gpu"` → force GPU (error if unavailable)
3. If `"auto"`:
   - GPU if detected, memory sufficient, and no unsupported features
   - CPU otherwise
4. Check `FACTORNET_RESOURCE` environment variable as override

## Template Hierarchy

### Scalar Type

All numeric computation is parameterized on `Scalar` (typically `float` or `double`):

```cpp
NMFConfig<float> config;        // single precision (default from R)
NMFResult<float> result;
DenseMatrix<float> W;           // Eigen::MatrixXf
SparseMatrix<float> A;          // Eigen::SparseMatrix<float, ColMajor>
```

The R bridge always uses `float` for computation (cast from R's `double` at entry, cast back at exit) for 2× memory savings and faster BLAS/GPU throughput.

### Resource Tag

Primitives are specialized per resource:

```cpp
template<typename Resource, typename Scalar>
void gram(const DenseMatrix<Scalar>& H, DenseMatrix<Scalar>& G);

// CPU: Eigen selfadjointView::rankUpdate
// GPU: cuBLAS SSYRK
```

`Resource` is either `primitives::CPU` or `primitives::GPU` — empty tag types that drive template specialization.

### Matrix Type

Algorithm templates accept both sparse and dense:

```cpp
template<typename Scalar, typename MatrixType>
NMFResult<Scalar> fit(const MatrixType& A, ...);
```

Type traits (`is_sparse_v<MatrixType>`) select the appropriate code paths at compile time.

## Memory Model

### Matrix Storage Conventions

| Matrix | Dimensions | Storage | Notes |
|--------|-----------|---------|-------|
| Input A | m × n | CSC (ColMajor) sparse or dense | Mapped, not copied |
| W (internal) | **k × m** | Dense, ColMajor | Transposed for cache efficiency |
| W (result) | m × k | Dense, ColMajor | Transposed back at finalization |
| H | k × n | Dense, ColMajor | Standard orientation |
| d | k | Vector | Diagonal scaling |
| G (Gram) | k × k | Dense, symmetric | Recomputed each iteration |
| B (RHS) | k × m or k × n | Dense | Recomputed each iteration |

### Why W is Transposed Internally

During the W-update, the NNLS solver processes one column of W at a time. Storing W as k×m means each column is a contiguous k-element vector in memory, matching the cache-line-friendly access pattern of the column-wise CD solver. The cost of the final transpose is O(mk), negligible compared to N iterations of O(nnz·k) work.

### Zero-Copy Sparse Input

R's `dgCMatrix` stores CSC data in three arrays: `i` (row indices), `p` (column pointers), `x` (values). The bridge maps these directly to `Eigen::Map<const SparseMatrix<float>>` — no copy, no allocation. Only the float cast creates temporary storage.

### Gram Trick for Loss

Instead of materializing the full m×n reconstruction, loss is computed as:

$$\|A - W \cdot \text{diag}(d) \cdot H\|_F^2 = \text{tr}(A^T A) - 2 \cdot \text{tr}(B^T H) + \text{tr}(G \cdot H H^T)$$

Cost: O(k²) instead of O(mn). The `tr(A^T A)` term is constant and precomputed once.

## Sparse vs Dense Paths

The matrix type determines the RHS primitive:

| Input Type | RHS Computation | GPU Kernel |
|-----------|----------------|------------|
| Sparse CSC | Column-wise iteration, OpenMP parallel | cuSPARSE SpMM |
| Dense | Eigen GEMM / BLAS DGEMM | cuBLAS SGEMM |

All other primitives (Gram, NNLS, loss) operate on the dense k×k / k×n matrices and are identical for both paths.

## Fused RHS+NNLS Path

For sparse MSE without masking or IRLS, the "fused" path combines RHS computation and NNLS solving into a single parallel loop over columns:

```
Standard: rhs(A, W, B)  →  nnls_batch(G, B, H)   // 2 passes over A
Fused:    fused_rhs_nnls_sparse(A, W, G, H)        // 1 pass over A
```

This eliminates one full O(nnz·k) pass and improves cache locality. The fused path is the default for standard sparse NMF.

## Cross-Validation Path

CV NMF (`fit_cv.hpp`) differs from standard NMF in:

1. **Lazy mask**: `SplitMix64::is_holdout(seed, i, j, frac)` determines holdout in O(1) — no mask matrix
2. **Gram correction**: Per-column delta-G subtracts held-out rows' contribution:
   ```
   G_col_j = G_full - Σ_{i ∈ holdout(j)} w_i · w_i^T
   ```
3. **Dual loss tracking**: Both train and test loss computed each check iteration
4. **Early stopping**: Best test loss tracked with patience counter

## Feature Application Order

Between Gram/RHS computation and NNLS solve, features modify the k×k Gram matrix G and k×n RHS matrix B:

```
1. L2       →  G.diagonal() += L2          (ridge)
2. Angular  →  G += angular · overlap       (orthogonality penalty)
3. Graph    →  G += λ · L^T L               (Laplacian smoothing)
4. L21      →  G(i,i) += L21 / ||row_i||   (group sparsity)
5. Guides   →  G, B modified per guide      (semi-supervised)
6. NNLS     →  solve per-column             (CD or Cholesky+clip)
7. L1       →  applied inside CD iterations  (soft thresholding)
8. Bounds   →  clip to [0, upper_bound]      (post-solve)
```

All feature costs are O(k²) or O(k²n), dwarfed by the O(nnz·k) primitives.

## IRLS Integration

Non-MSE losses use Iteratively Reweighted Least Squares. Each NNLS call becomes:

```
for irls_iter = 0 to irls_max_iter:
    weights = compute_irls_weights(residuals, loss_type)
    solve weighted_nnls(G, B, H, weights)
```

The weights are per-element scalars derived from the loss function's variance function. The weighted NNLS modifies the per-column Gram and RHS to incorporate weights, then uses the same CD solver.

## Directory Layout

```
inst/include/FactorNet/
├── core/           Config, result, types, traits, constants, resources, RNG
├── math/           Loss functions, BLAS utilities
├── nmf/            NMF algorithms (standard, CV, streaming, chunked, GPU)
├── svd/            SVD methods (5 algorithms + auto-select + streaming)
├── graph/          DAG graph: node types, compilation, execution, results
├── primitives/
│   ├── cpu/        Gram, RHS, NNLS (CD + Cholesky), fused, loss
│   └── gpu/        CUDA kernels (same API via template specialization)
├── features/       L1, L2, L21, angular, graph Laplacian, bounds
├── guides/         Classifier guide, external guide
├── clustering/     Bipartition, divisive clustering
├── io/             DataLoader interface, in-memory, SPZ streaming
├── rng/            SplitMix64 (seeded, thread-safe, GPU-compatible)
└── gpu/            GPU context, dlsym bridge for CPU-only builds
```
