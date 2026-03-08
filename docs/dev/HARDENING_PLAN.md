# RcppML / FactorNet — Production Hardening Plan

**Version**: Draft 2 · March 6, 2026  
**Scope**: Sequential framework for hardening all functionality — scoping supported features, crafting the API, validating correctness, optimizing performance, documenting everything, and publishing. Each section is a self-contained phase executed in order by AI agents.

---

## Table of Contents

1. [Guiding Principles](#1-guiding-principles)
2. [Phase 1 — Functional Coverage Matrix](#2-phase-1--functional-coverage-matrix)
3. [Phase 2 — API Design & Hardening](#3-phase-2--api-design--hardening)
4. [Phase 3 — C++ Library Testing](#4-phase-3--c-library-testing)
5. [Phase 4 — R Package Testing](#5-phase-4--r-package-testing)
6. [Phase 5 — Correctness Fixes & Guards](#6-phase-5--correctness-fixes--guards)
7. [Phase 6 — Performance Benchmarking](#7-phase-6--performance-benchmarking)
8. [Phase 7 — Performance Optimization](#8-phase-7--performance-optimization)
9. [Phase 8 — Algorithmic Methods Documentation](#9-phase-8--algorithmic-methods-documentation)
10. [Phase 9 — C++ FactorNet Library Documentation](#10-phase-9--c-factornet-library-documentation)
11. [Phase 10 — R Package Documentation](#11-phase-10--r-package-documentation)
12. [Phase 11 — Publication Roadmap](#12-phase-11--publication-roadmap)
13. [AI-Assisted Development Harness](#13-ai-assisted-development-harness)

---

## 1. Guiding Principles

These principles govern all hardening work and resolve tradeoffs:

1. **Correctness first.** Every algorithmic path must be mathematically validated before performance optimization.
2. **Performance over minor API convenience.** Never introduce abstraction layers that measurably degrade hot-loop throughput.
3. **Fail loud.** Every theoretically unsound parameter combination must throw a clear, documented error — never silently produce bad results.
4. **Feature parity across backends.** CPU, GPU, and streaming must support the same feature set unless a fundamental theoretical limitation prevents it (documented & guarded).
5. **One source of truth.** Each algorithm is implemented once in C++; R (and future Python) bindings are thin wrappers calling the documented C++ API.
6. **Measurable claims.** Every performance claim is backed by reproducible benchmarks with documented hardware, dataset, and configuration.
7. **Bias toward completeness.** We prefer supporting every combination of features on every backend. Gaps require explicit justification. Users should never be surprised by missing support.

---

## 2. Phase 1 — Functional Coverage Matrix

**Goal**: Produce a definitive, machine-readable map of every supported feature combination across all backends. This scopes the entire project: anything in the matrix is committed to; anything outside is either removed from the codebase or guarded with clear errors.

**Deliverables**:
- `docs/dev/COVERAGE_MATRIX.yaml` — machine-readable matrix
- `docs/dev/COVERAGE_MATRIX.md` — human-readable summary tables
- List of identified gaps with required actions (implement, guard, or remove)

### 2.1 Conceptual Model

FactorNet has two fundamental input types and two I/O layers:

| Concept | Role | Computes on? |
|---------|------|-------------|
| **sparse** (CSC/dgCMatrix) | Core compute format for sparse data | Yes — all algorithms |
| **dense** (column-major matrix) | Core compute format for dense data | Yes — all algorithms |
| **SPZ file** | I/O format — decompresses to sparse | No — decompresses into sparse, then computes |
| **gpu_sparse_matrix** | Zero-copy device wrapper around sparse | No — wraps sparse, avoids re-upload |

SPZ and gpu_sparse are not independent input types. They are I/O and memory-management layers over sparse. The coverage matrix tracks **compute backends × core input types** (sparse, dense), with streaming and GPU-direct as orthogonal I/O modes.

### 2.2 Feature Axes

The following axes fully describe the space of NMF parameter combinations:

**A. Backend** (where the hot loop runs):

| Backend | Description |
|---------|-------------|
| CPU | Standard in-memory CPU execution (OpenMP) |
| GPU | CUDA device execution, factors on device |
| Streaming-CPU | Out-of-core panel-wise CPU (SPZ I/O → chunked compute) |
| Streaming-GPU | Out-of-core panel-wise GPU (CPU I/O → GPU compute) |

MPI support has been **removed**. All distributed computing use cases should use GPU acceleration with streaming SPZ I/O instead.

**B. Input type**: sparse, dense

**C. Distribution / Loss** (the statistical model):

| Distribution | LossType enum | IRLS required | Dispersion parameter | Description |
|-------------|---------------|---------------|---------------------|-------------|
| Gaussian (MSE) | MSE (0) | No* | n/a | Default — Frobenius norm. *IRLS only if robust_delta > 0 |
| Generalized Poisson | GP (4) | Yes | θ (overdispersion) | Count data with Var > Mean. KL is the special case θ=0 (Poisson) |
| Negative Binomial | NB (5) | Yes | r (size parameter) | Count data with quadratic variance-mean: Var = μ + μ²/r |
| Gamma | GAMMA (6) | Yes | φ (dispersion) | Positive continuous, V(μ) = φμ² |
| Inverse Gaussian | INVGAUSS (7) | Yes | φ (dispersion) | Heavy right-skew, V(μ) = φμ³ |
| Tweedie | TWEEDIE (8) | Yes | p (power) | Generalized V(μ) = μᵖ, interpolates Poisson↔Gamma |

Note: MAE (1) and Huber (2) are **legacy/deprecated** loss types — they are fully subsumed by the `robust_delta` parameter which applies a Huber-like robustness modifier on Pearson residuals from **any** distribution. They remain in the enum for backward compatibility but should steer users toward `distribution + robust_delta`.

**D. Dispersion mode** (for GP/NB/Gamma/IG/Tweedie):

| Mode | Scope | Parameters estimated |
|------|-------|---------------------|
| none | Fixed (θ=0 or φ=1) | None — Poisson limit for GP, fixed for others |
| global | Entire matrix | Single θ, r, or φ |
| per_row | Per feature (row) | Vector of length m |
| per_col | Per sample (column) | Vector of length n |

**E. Zero-inflation mode** (ZIGP, ZINB):

| Mode | Parameters | Scope |
|------|-----------|-------|
| none | — | Standard (no dropout modeling) |
| row | π_i | Per-row (feature-level dropout) |
| col | π_j | Per-column (sample-level dropout) |
| twoway | π_ij = 1-(1-π_i)(1-π_j) | Multiplicative row×column (**currently broken on high-sparsity data**) |

**F. Robustness**:

| Setting | Effect |
|---------|--------|
| robust_delta = 0 | Standard (no robustness) |
| robust_delta > 0 | Huber modifier on Pearson residuals — downweights large residuals from any distribution |

**G. Solver** (NNLS subproblem):

| Solver | Description | Best for |
|--------|-------------|----------|
| CD (coordinate descent) | Iterative, active-set-like | Low k (k ≤ ~24) |
| Cholesky + clip | Direct factorization, project to NN | High k (k ≥ ~24), MSE only |

Note: IRLS distributions always use CD internally (Cholesky does not support per-column weight recomputation).

**H. Initialization**:

The initialization axis should be simplified. Rather than exposing Lanczos vs IRLBA vs random as user choices, the system should auto-select:
- **SVD init** (default): Use the fastest numerically stable SVD method for the given rank and backend. Typically Lanczos for low k, IRLBA for moderate k. For heavily masked/regularized problems, consider Krylov or deflation-based init.
- **Random init**: Uniform random, seed-controlled. Fallback when SVD is not appropriate.
- **User-supplied init**: Warm-start from external W, H matrices.

**I. Regularization** (per-factor, independent for W and H):

| Feature | Parameter | Tier | Effect |
|---------|-----------|------|--------|
| L1 (Lasso) | L1 ∈ [0,1] | 1 (Gram) | Element sparsity via soft-thresholding |
| L2 (Ridge) | L2 ≥ 0 | 1 (Gram) | Shrinkage via diagonal augmentation |
| L21 (Group Lasso) | L21 ≥ 0 | 2 (Factor) | Row-wise sparsity (entire rows → zero) |
| Angular (Orthogonality) | angular ≥ 0 | 1-2 | Penalizes factor inner products |
| Graph Laplacian | graph + graph_lambda | 2 (Factor) | Encourages smooth factors w.r.t. graph |
| Upper bound | upper_bound ≥ 0 | Post-NNLS | Element-wise ceiling |
| Non-negativity | nonneg (bool) | Built-in | Standard NMF constraint |

**J. Guides** (semi-supervised, per-factor):

| Guide type | Effect |
|-----------|--------|
| Classifier guide | Attracts factors toward class centroids (labelled supervision) |
| External guide | Attracts factors toward user-supplied reference matrix |

**K. NMF Variant**:

| Variant | Model | H-update |
|---------|-------|----------|
| Standard | A ≈ W·d·H | Independent NNLS |
| Projective | A ≈ W·d·W'A | H = d·W'·A (no NNLS, ~1.5× faster) |
| Symmetric | A ≈ W·d·W' | H = W' (no H-NNLS, ~0.5× cost) |

**L. Cross-validation mode**:

| Mode | Mask definition | Use case |
|------|----------------|----------|
| none | No holdout | Standard fitting |
| speckled (mask_zeros=TRUE) | Only mask existing nonzeros | Recommendation systems |
| full (mask_zeros=FALSE) | Mask all entries uniformly | Dense reconstruction |

**M. Normalization**: L1, L2, none (post-processing, applied to diagonal d)

### 2.3 Preventing Combinatorial Explosion

With ~12 axes, naive enumeration would yield millions of combinations. We reduce this through **factored testing**:

1. **Independent axes test independently.** Regularization (L1, L2, L21, angular, graph, bounds) adds to the Gram matrix or applies post-NNLS. These features are *independent of each other and of loss type*. We test each regularization feature once (on CPU, MSE) to verify its Gram/factor modification is correct. We do NOT test L1×L2×L21×angular×... cross-products.

2. **Backend parity tests are overlay tests.** For each core algorithm configuration (loss + variant + CV mode), we verify GPU matches CPU. We don't re-test every regularization combination on GPU — we test that the GPU feature-application path (download, apply, upload) works correctly once, then trust it for all features.

3. **Interaction tests only where interactions exist.** These are:
   - Distribution × CV (test loss computation depends on distribution)
   - Distribution × ZI (ZI modifies the EM loop within IRLS)
   - CV × solver (Cholesky vs CD affects CV per-column solve)
   - Variant × backend (projective/symmetric have special GPU paths)
   - Streaming × anything (streaming changes data access patterns)

4. **Reduction rule**: The coverage matrix enumerates **interaction groups**, not individual cells. Each group is a set of axes that interact, tested together. Non-interacting axes are tested in isolation.

### 2.4 Coverage Matrix Groups

**Group 1: Core Algorithm × Backend × Input** (~20 combinations)
```
NMF × {CPU, GPU, Streaming-CPU, Streaming-GPU} × {sparse, dense}
SVD × {CPU, GPU, Streaming-CPU} × {sparse, dense}
NNLS × {CPU, GPU} × standalone
```

**Group 2: Distribution × Backend** (~40 combinations)
```
{MSE, GP, NB, Gamma, InvGauss, Tweedie} × {CPU, GPU, Streaming-CPU, Streaming-GPU} × {sparse, dense}
+ robust_delta > 0 overlay for each
```

**Group 3: Distribution × ZI** (~15 combinations)
```
{GP, NB} × {ZI_NONE, ZI_ROW, ZI_COL, ZI_TWOWAY} × {CPU, GPU}
(Only GP and NB support ZI)
```

**Group 4: Distribution × CV** (~20 combinations)
```
{MSE, GP, NB, Gamma, InvGauss, Tweedie} × {no_CV, speckled, full} × {CPU, GPU}
```

**Group 5: Regularization (independent, tested in isolation)** (~14 tests)
```
{L1, L2, L21, angular, graph, upper_bound, nonneg=F} × {W, H}
Each tested once on CPU+sparse+MSE
```

**Group 6: GPU + Regularization path** (~7 tests)
```
Each regularization feature once on GPU to verify download-apply-upload path
```

**Group 7: Variants × Backend** (~8 combinations)
```
{standard, projective, symmetric} × {CPU, GPU} × {sparse}
+ CV overlay for each variant
```

**Group 8: Guides** (~4 combinations)
```
{classifier, external} × {CPU, GPU}
```

**Group 9: Streaming** (~10 combinations)
```
Streaming-CPU × {MSE, GP, NB, ...} (verify which work)
Streaming-GPU × {MSE} + rejection tests for unsupported
```

**Group 10: SVD Methods** (~15 combinations)
```
{deflation, krylov, lanczos, irlba, randomized} × {CPU, GPU}
+ constrained SVD (L1, nonneg, etc.) on deflation/krylov only
+ CV auto-rank
```

**Group 11: FactorNet Graph** (~5 combinations)
```
Single-layer, deep (2-layer), multi-modal (shared H), branching × {CPU, GPU}
```

**Group 12: Clustering & Utilities** (~5 tests)
```
dclust, bipartition, bipartiteMatch, consensus_nmf, predict/project
```

**Total**: ~160 interaction-aware test points (not millions).

### 2.5 Coverage Matrix YAML Schema

```yaml
# docs/dev/COVERAGE_MATRIX.yaml
groups:
  - id: "core_algorithm_backend"
    description: "Core algorithm × backend × input type"
    entries:
      - id: "nmf.cpu.sparse"
        status: validated
        test: test_nmf_correctness.R::nmf_cpu_sparse_*
        notes: ""
      - id: "nmf.gpu.sparse"
        status: validated
        test: test_gpu_nmf.R::nmf_gpu_sparse_*
        notes: ""
      - id: "nmf.streaming_cpu.sparse"
        status: implemented
        test: test_streaming_cpu.R::*
        notes: "MSE only; non-MSE needs chunked dense materialization"
      # ... etc

  - id: "distribution_backend"
    description: "Distribution × backend × input"
    entries:
      - id: "gp.cpu.sparse"
        status: validated
        test: test_loss_gp.R::gp_cpu_sparse_*
        notes: ""
      - id: "gp.gpu.sparse"
        status: implemented
        test: null
        notes: "GPU IRLS native on-device. Needs parity test."
      # ... etc

  - id: "distribution_zi"
    description: "Distribution × zero-inflation mode"
    entries:
      - id: "gp.zi_row.cpu"
        status: validated
        test: test_loss_zi.R::*
        notes: "Ship-ready per ZIGP assessment"
      - id: "gp.zi_twoway.cpu"
        status: broken
        test: null
        notes: "Runaway π on high-sparsity data. Fix or disable."
        action: "Fix with damping/cap OR disable with clear error"
      # ... etc
```

**Status values**:
- `validated` — Implemented, tested against ground truth, passing
- `implemented` — Code exists, untested or test incomplete
- `broken` — Known incorrect behavior
- `unsupported` — Theoretically invalid — must have error guard
- `planned` — Not yet implemented, will be
- `removed` — Code removed, no longer supported

### 2.6 Known Gaps (from audit — to verify during matrix generation)

| Gap | Current Status | Required Action |
|-----|---------------|-----------------|
| ZI-twoway runaway on sparse data | broken | Fix with damping/safe guards, or disable + error |
| SVD: no testthat tests | implemented | Comprehensive test_svd_correctness.R |
| SVD: no `predict()` S4 method | missing | Implement |
| SVD: deflation ghost eigenvalues | broken (numerical) | Add Gram-Schmidt reorthogonalization |
| GPU CV NMF: untested | implemented | Test GPU CVvs CPU CV |
| GPU IRLS: fully on-device (NOT fallback, per audit) | implemented | Needs parity tests |
| Projective/symmetric NMF: untested | implemented | Write tests |
| Streaming + non-MSE losses | unsupported on GPU streaming; supported on CPU chunked | Verify CPU chunked correctness; guard GPU streaming |
| Dense GPU CV | unsupported | Guard with clear error |
| Adaptive solver selection | hard-coded | Benchmark crossover, implement auto |
| GPU SVD: which methods have native GPU? | partially implemented | Audit and document per-method |

### 2.7 Agent Task: Generate Coverage Matrix

The agent for this phase must:

1. **Enumerate all groups** per §2.4, reading the actual C++ config structs and dispatch logic
2. **For each entry**: determine status by checking (a) does the code path exist? (b) does a test exist? (c) does a guard exist if unsupported?
3. **Write `COVERAGE_MATRIX.yaml`** with every entry classified
4. **Write `COVERAGE_MATRIX.md`** with human-readable summary tables organized by group
5. **Produce a gap report**: every entry that is not `validated` or `unsupported+guarded`
6. **For every `unsupported` entry**: verify that a guard exists in R validation AND C++ gateway; if not, flag as a required fix

---

## 3. Phase 2 — API Design & Hardening

**Goal**: Using the coverage matrix as the specification, lock down the public API surface for both C++ (FactorNet) and R (RcppML). Every supported combination must be reachable through a clean, consistent API. Every unsupported combination must produce a clear error.

**Deliverables**:
- Finalized R API: `nmf()`, `svd()`/`pca()`, `nnls()`, `factor_net()`, `dclust()`, etc.
- Finalized C++ public API: `FactorNet::nmf::fit()`, `FactorNet::svd::fit()`, etc.
- Complete R validation layer covering every coverage matrix entry
- C++ gateway guards for every `unsupported` entry (defense in depth)

### 3.1 R API Principles

- **Naming**: `nmf()`, `svd()`, `pca()`, `nnls()`, `factor_net()`, `consensus_nmf()`, `dclust()`, `bipartition()`
- **Parameter consistency**: Same names across functions (e.g., `tol`, `maxit`, `seed`, `verbose`, `resource`, `threads`)
- **Sensible defaults**: Every function works with just `f(data, k)`
- **Vectorized rank**: `nmf(A, k=c(5,10,15))` does multi-rank CV automatically
- **S4 return types**: Consistent, documented slot structure
- **Backward compatibility**: Old `model$w` syntax preserved

### 3.2 Validation Completeness Checklist

For every axis in §2.2:
- [ ] Type check at R level
- [ ] Range check at R level
- [ ] Mutual exclusivity check (e.g., `projective + symmetric`)
- [ ] Backend availability check (GPU not available → message + CPU fallback)
- [ ] Unsupported combination check against coverage matrix
- [ ] C++ gateway guard (defense in depth)

### 3.3 C++ Public API Surface

Minimal, stable, documented:

```cpp
namespace FactorNet {
  // NMF
  NMFResult<Scalar> nmf::fit(const Matrix& A, const NMFConfig<Scalar>& config);
  
  // SVD
  SVDResult<Scalar> svd::fit(const Matrix& A, const SVDConfig<Scalar>& config);
  
  // NNLS
  DenseMatrix<Scalar> nnls::solve(const DenseMatrix<Scalar>& G, const DenseMatrix<Scalar>& B, ...);
  
  // Graph
  GraphResult<Scalar> graph::fit(const Graph& g);
  
  // Clustering
  DClustResult clustering::dclust(const SparseMatrix<Scalar>& A, const DClustConfig& config);
}
```

---

## 4. Phase 3 — C++ Library Testing

**Goal**: Write tests for the C++ FactorNet library that verify algorithmic correctness and numerical stability. These tests validate the C++ implementation **independent of R**. They are the ground truth: if C++ is correct, then R and Python wrappers only need to verify correct parameter passing.

**Deliverables**:
- C++ test harness (R-callable but testing C++ directly via thin Rcpp wrappers)
- Tests organized by coverage matrix groups
- Ground truth validation for every `validated` entry

### 4.1 C++ Test Strategy

Since FactorNet is a header-only library distributed within an R package, C++ tests are written as R test files that call thin Rcpp-exported C++ functions with known inputs and check outputs. The R test layer is minimal — it provides the test data and asserts on the result, but all computation happens in C++.

### 4.2 Ground Truth Methods

| Algorithm | Ground Truth |
|-----------|-------------|
| NMF (MSE) | Synthetic A = W·d·H + noise. Verify cosine(W_est, W_true) > threshold per factor. |
| NMF (GP/NB/Gamma) | Simulated count data from known distribution. Verify (a) loss decreases monotonically, (b) dispersion parameter estimates converge to true values within tolerance. |
| SVD | Compare against `base::svd()` on small matrices. Verify relative reconstruction error. |
| NNLS | Analytically solved small systems. Verify residual per column. |
| CV | Verify mask determinism, train/test loss separation, early stopping behavior. |
| Streaming | Identical config on in-memory vs streaming → results match within tolerance. |
| GPU parity | CPU result vs GPU result with same config+seed → match within fp32 tolerance. |

### 4.3 Test Organization (Coverage Matrix Groups → Test Files)

```
tests/testthat/
├── # ── C++ correctness (ground truth) ──
├── test_cpp_nmf_mse.R            # Group 1: NMF MSE on CPU × {sparse, dense}
├── test_cpp_nmf_distributions.R  # Group 2: GP, NB, Gamma, InvGauss, Tweedie on CPU
├── test_cpp_nmf_zi.R             # Group 3: ZI modes × {GP, NB} on CPU
├── test_cpp_nmf_cv.R             # Group 4: CV × distributions on CPU
├── test_cpp_nmf_regularization.R # Group 5: Each reg feature independently
├── test_cpp_nmf_variants.R       # Group 7: Projective, symmetric
├── test_cpp_nmf_guides.R         # Group 8: Classifier + external guides
├── test_cpp_svd.R                # Group 10: All 5 SVD methods vs base::svd
├── test_cpp_nnls.R               # NNLS solver vs known solutions
├── test_cpp_factornet_graph.R    # Group 11: Graph composition
├── test_cpp_clustering.R         # Group 12: dclust, bipartition

├── # ── GPU parity ──
├── test_gpu_nmf_parity.R         # Group 1: GPU NMF ≈ CPU NMF
├── test_gpu_distributions.R      # Group 2: GPU IRLS ≈ CPU IRLS
├── test_gpu_cv_parity.R          # Group 4: GPU CV ≈ CPU CV
├── test_gpu_regularization.R     # Group 6: GPU feature path
├── test_gpu_variants.R           # Group 7: Projective/symmetric on GPU
├── test_gpu_svd_parity.R         # Group 10: GPU SVD methods

├── # ── Streaming parity ──
├── test_streaming_cpu.R          # Group 9: SPZ streaming ≈ in-memory
├── test_streaming_gpu.R          # Group 9: GPU streaming ≈ in-memory
├── test_streaming_rejection.R    # Unsupported combos throw errors
```

---

## 5. Phase 4 — R Package Testing

**Goal**: Write tests for the R wrapper layer. These verify that R-level parameter validation, dispatch, S4 methods, plotting, and convenience functions work correctly. They do NOT re-test C++ algorithmic correctness — that's Phase 3's job.

**Deliverables**:
- R wrapper tests covering parameter validation, dispatch, return types
- S4 method tests (summary, print, subset, align, predict, plot)
- Error message tests for every unsupported combination
- Backward compatibility tests

### 5.1 R Test Files

```
tests/testthat/
├── # ── R wrapper & validation ──
├── test_r_nmf_validation.R       # Parameter checks, error messages, defaults
├── test_r_svd_validation.R       # SVD parameter checks
├── test_r_dispatch.R             # Correct backend selection, GPU fallback
├── test_r_s4_methods.R           # S4: summary, print, dim, align, subset, $
├── test_r_predict.R              # predict() and project()
├── test_r_plotting.R             # All plot functions (no visual test, just no-error)
├── test_r_consensus.R            # consensus_nmf
├── test_r_factor_net.R           # factor_net R-level API
├── test_r_spz_io.R              # SPZ read/write round-trips
├── test_r_backward_compat.R      # Old API patterns still work
├── test_r_edge_cases.R           # k=1, empty cols, single-element, dimension names
├── test_r_reproducibility.R      # Seed determinism from R level
```

---

## 6. Phase 5 — Correctness Fixes & Guards

**Goal**: Address every gap identified in the coverage matrix. Fix broken entries, add guards for unsupported entries, implement planned entries.

**Deliverables**:
- Every `broken` entry fixed or disabled with error
- Every `unsupported` entry guarded at both R and C++ levels
- Every `planned` entry implemented or re-classified
- Updated COVERAGE_MATRIX with all entries at `validated` or `unsupported+guarded`

### 6.1 Known Fixes Required

| Issue | Action |
|-------|--------|
| ZI-twoway runaway on sparse data | Add damping + π cap, or disable and throw clear error |
| SVD deflation ghost eigenvalues | Add Gram-Schmidt reorthogonalization across extracted factors |
| SVD `predict()` S4 method missing | Implement S4 method using V projection |
| Dense GPU CV unsupported | Add R + C++ guard with informative error |
| Streaming + ZI unsupported | Add guard |
| Cholesky + non-MSE | Verify guard (Cholesky doesn't support IRLS weights) |

### 6.2 Guard Implementation Pattern

For every `unsupported` entry:

```r
# R validation (nmf_validation.R):
if (streaming && loss != "mse") {
  stop("Streaming NMF currently supports MSE loss only. ",
       "Non-MSE losses (GP, NB, etc.) require in-memory fitting. ",
       "Set streaming=FALSE or use loss='mse'.", call. = FALSE)
}
```

```cpp
// C++ gateway guard (fit.hpp) — defense in depth:
if (config.spz_path.size() > 0 && config.loss.type != LossType::MSE) {
  throw std::invalid_argument(
    "Streaming NMF only supports MSE loss. "
    "Use in-memory fitting for non-MSE losses.");
}
```

```r
# Test (test_streaming_rejection.R):
test_that("streaming_with_gp_loss_throws_error", {
  expect_error(nmf("data.spz", k=5, loss="gp"),
               "Streaming NMF currently supports MSE")
})
```

---

## 7. Phase 6 — Performance Benchmarking

**Goal**: Establish reproducible, automated benchmarks for every major code path. Measure baseline performance. Identify bottlenecks. All benchmarks must complete within 10 minutes total to enable routine regression checking.

**Deliverables**:
- `benchmarks/harness/` structured benchmark suite
- Baseline results frozen at current code state
- Regression detection script (flag Δ > 5%)
- Performance profile identifying top bottlenecks per code path

### 7.1 Benchmark Design Constraints

- **10-minute wall-clock budget** for the full suite (enables routine checking)
- **Measure total runtime only** (no in-code phase breakdown instrumentation). If a regression is detected, then and only then do a detailed manual profile.
- **5 replicates** per configuration to capture variance
- **Fixed iterations** (`tol=1e-10`) to force `maxit` iterations for fair timing
- **Reproducible** via pinned seed, dataset, hardware, and git commit

### 7.2 Benchmark Suite Structure

```
benchmarks/harness/
├── config.yaml              # Suite definitions
├── run_all.R                # Master harness (< 10 min total)
├── datasets/
│   └── generate.R           # Synthetic data: 5K×2K sparse (90% zeros), 1K×500 dense
├── suites/
│   ├── nmf_cpu_baseline.R   # MSE NMF: k ∈ {8,16,32,64}, sparse+dense, 20 iters, 3 rep
│   ├── nmf_gpu_baseline.R   # Same configs on GPU
│   ├── nmf_distributions.R  # GP, NB, Gamma on CPU+GPU, k=16, 10 iters
│   ├── nmf_cv.R             # CV NMF: CPU vs GPU, k=16, 20 iters
│   ├── nmf_streaming.R      # Streaming vs in-memory, k=16
│   ├── svd_methods.R        # 5 SVD methods, k ∈ {5,10,20}
│   ├── nnls_crossover.R     # CD vs Cholesky, k ∈ {4,8,16,24,32,48,64}
│   └── feature_overhead.R   # L1, L2, L21, angular, graph overhead vs baseline
├── results/
│   ├── baseline/            # Frozen baselines (timestamped, git-tagged)
│   └── current/             # Latest run
├── analysis/
│   ├── regression_check.R   # Flag Δ > 5%
│   └── generate_report.R    # Markdown tables + plots
└── README.md
```

### 7.3 Standard Result Schema

```yaml
metadata:
  git_commit: abc1234
  timestamp: "2026-03-06T12:00:00Z"
  node: c003
  cpu: "AMD EPYC 7763"
  gpu: "NVIDIA H100 80GB"  # or null
  omp_threads: 4

results:
  - name: "nmf_cpu_sparse_mse_k16"
    backend: cpu
    input: sparse
    distribution: mse
    rank: 16
    iterations: 20
    replicates: 3
    mean_sec: 1.23
    sd_sec: 0.05
    final_loss: 0.0412
```

---

## 8. Phase 7 — Performance Optimization

**Goal**: Using benchmark results, optimize bottlenecks where gains are significant (>10% improvement). Stop when marginal gains are unlikely.

**Deliverables**:
- Adaptive solver selection (`solver="auto"`) with empirical crossover data
- GPU warm-start NNLS (if benchmark shows significant gain)
- Streaming double-buffering (CPU I/O overlapped with GPU compute)
- C++ deduplication (SpMVContext, centering kernels)
- Updated benchmarks showing improvement

### 8.1 Priority Optimizations

| Optimization | Expected Gain | Benchmark to validate |
|-------------|--------------|----------------------|
| Adaptive CD↔Cholesky selection | 2-5× for high k on GPU | nnls_crossover.R |
| GPU warm-start NNLS | 2-3× NNLS phase | nmf_gpu_baseline.R before/after |
| Streaming double-buffer | 30-50% total streaming | nmf_streaming.R |
| SpMVContext dedup (SVD) | Maintainability, not speed | N/A |

### 8.2 Optimization Discipline

- **Only optimize measured bottlenecks.** No speculative optimization.
- **Benchmark before and after.** Every optimization must show measured improvement.
- **Don't break correctness.** Re-run Phase 3 tests after every optimization.
- **Document crossover points.** For adaptive decisions (e.g., CD vs Cholesky), record the empirical data.

---

## 9. Phase 8 — Algorithmic Methods Documentation

**Goal**: Write the definitive algorithmic reference for every non-trivial method in FactorNet. This is built **first** among the four documentation tiers because it is the intellectual foundation that all other docs reference. These documents also serve as manuscript drafts.

**Deliverables**:
- `docs/factornet/algorithms/` — One document per major algorithm
- Mathematical definitions, update rules, convergence proofs or guarantees
- Complexity analysis
- Relationship to literature

### 9.1 Documents Required

| Document | Content |
|----------|---------|
| `algorithms/nmf.md` | ALS update rules, W/d/H normalization, convergence, projective/symmetric variants |
| `algorithms/irls.md` | IRLS framework: weight functions for each distribution (GP, NB, Gamma, IG, Tweedie), convergence properties, robust_delta modifier, interaction with CD solver |
| `algorithms/zero_inflation.md` | ZIGP/ZINB EM: E-step (posterior π), M-step (update π_row/π_col), soft imputation, interaction with IRLS loop, twoway instability analysis |
| `algorithms/cross_validation.md` | Speckled mask (lazy PRNG hash), per-column Gram correction derivation, train/test loss separation, early stopping, mask_zeros semantics |
| `algorithms/svd.md` | All 5 SVD methods (deflation, Krylov, Lanczos, IRLBA, randomized): algorithms, when to use each, constrained SVD via ALS, auto-rank via CV |
| `algorithms/nnls.md` | CD solver (active set, convergence), Cholesky+clip solver, crossover analysis, warm-start strategy |
| `algorithms/distributions.md` | Mathematical specification of each distribution: PMF/PDF, variance function V(μ), IRLS weight derivation, dispersion estimation (MoM, Newton-Raphson) |
| `algorithms/sparsepress.md` | SPZ v2 format specification, Golomb-Rice + rANS coding, streaming decompression, panel-wise I/O |
| `algorithms/factornet_graph.md` | Graph composition model, node types, compilation, multi-modal shared factors, deep factorization |

---

## 10. Phase 9 — C++ FactorNet Library Documentation

**Goal**: Document the C++ library for non-R consumers (Python bindings, standalone C++ users). This references the algorithmic methods docs and adds implementation-specific details.

**Deliverables**:
- `docs/factornet/README.md` — Library overview and quick start
- `docs/factornet/ARCHITECTURE.md` — Dispatch flow, backend selection, memory model
- `docs/factornet/API_REFERENCE.md` — Public API with all entry points, config structs, return types
- Doxygen-style comment blocks on all public C++ headers
- `docs/factornet/gpu/` — GPU architecture, kernel inventory, performance data, fallback rules
- `docs/factornet/io/` — SPZ streaming API, panel boundaries

---

## 11. Phase 10 — R Package Documentation

**Goal**: Complete the R-facing documentation tier. This is built last because it references the API (Phase 2) and algorithmic methods (Phase 8).

**Deliverables**:
- All roxygen `@param`, `@return`, `@examples`, `@seealso` complete for every exported function
- Vignettes (pkgdown articles)
- pkgdown site built and deployable
- All `unsupported` combinations documented in `@details`

### 11.1 Vignette Plan

Final vignette selection should be decided by the agent based on what users most need, but the likely set:

| Vignette | Priority | Content |
|----------|----------|---------|
| Getting started | P0 | Quick tutorial: install, load, run NMF, inspect results |
| NMF deep dive | P0 | Regularization, constraints, loss functions, CV, variants |
| SVD and PCA | P0 | Methods, constrained SVD, auto-rank, PCA workflow |
| Cross-validation | P1 | Rank selection, mask semantics, interpretation |
| GPU acceleration | P1 | Setup, when to use, performance expectations |
| Statistical distributions | P1 | When to use GP/NB/Gamma, zero-inflation, robust estimation |
| SparsePress and streaming | P1 | SPZ format, out-of-core workflow, memory estimation |
| FactorNet graphs | P2 | Composable layers, deep NMF, multi-modal |
| Application examples | P2 | scRNA-seq, recommendation, image decomposition |
| Performance guide | P2 | CPU vs GPU decision tree, solver selection, thread tuning |

---

## 12. Phase 11 — Publication Roadmap

**Goal**: Identify publication units, prepare reproducible manuscript scaffolds, connect each to validated code modules.

### 12.1 Publication Units

| # | Working Title | Key Novelty | Target Venue |
|---|---------------|-------------|--------------|
| **P1** | *RcppML: Scalable NMF, SVD, and Divisive Clustering in R* | Unified R interface with GPU/streaming/IRLS/CV | JSS / R Journal |
| **P2** | *GPU-Accelerated NMF Cross-Validation via Fused Per-Column Gram Correction* | 100×+ CV speedup; per-column Gram correction theory | Bioinformatics / JCGS |
| **P3** | *Distribution-Aware NMF: IRLS Framework for Count, Continuous, and Zero-Inflated Data* | Unified IRLS + ZI-EM for NMF with 7 distributions | Biostatistics / Stat Methods |
| **P4** | *SparsePress: Entropy-Coded Compressed Sparse Format for Out-of-Core Factorization* | Streaming decompression eliminates memory bottleneck | SoftwareX / JOSS |
| **P5** | *FactorNet: Composable Constrained Matrix Factorization Graphs* | Graph abstraction over NMF/SVD layers | JMLR / NeurIPS |
| **P6** | *Constrained SVD via ALS with Regularization, Non-negativity, and Cross-validation* | Regularized SVD framework with auto-rank | Computational Statistics |

### 12.2 Publication-Code Coupling

Each manuscript requires:
1. Validated code modules (from Phase 3-5)
2. Benchmark data (from Phase 6)
3. Reproducible scripts in `manuscript/<paper>/`
4. Comparison baselines against existing methods

---

## 13. AI-Assisted Development Harness

### 13.1 Sequential Execution Model

Phases 1-11 are executed **sequentially** by agents. Each phase's output is the input to the next. An agent completes one phase in entirety before the next begins.

```
Phase 1: Coverage Matrix
    ↓ (scopes everything)
Phase 2: API Design
    ↓ (defines the interface)
Phase 3: C++ Tests
    ↓ (validates correctness)
Phase 4: R Tests
    ↓ (validates wrapper)
Phase 5: Fixes & Guards
    ↓ (closes gaps)
Phase 6: Benchmarks
    ↓ (measures performance)
Phase 7: Optimization
    ↓ (improves performance)
Phase 8: Algorithm Docs
    ↓ (documents theory)
Phase 9: C++ Library Docs
    ↓ (documents implementation)
Phase 10: R Package Docs
    ↓ (documents usage)
Phase 11: Publications
```

### 13.2 Key Files for Agent Context

| File | Purpose |
|------|---------|
| `HARDENING_PLAN.md` | Master plan — read at start of every phase |
| `docs/dev/COVERAGE_MATRIX.yaml` | Feature scope — generated in Phase 1 |
| `docs/dev/COVERAGE_MATRIX.md` | Human-readable coverage — generated in Phase 1 |
| `.github/copilot-instructions.md` | Build/test commands, HPC rules, pitfalls |
| `tests/testthat/helper-test-utils.R` | Test infrastructure |

### 13.3 Agent Task Format

Each phase is assigned to an agent with this structure:

```
Phase: [N]
Goal: [One-sentence goal from this document]
Deliverables: [Exact files to create/modify]
Context files to read first:
  - HARDENING_PLAN.md (this file)
  - .github/copilot-instructions.md
  - [Phase-specific files]
Constraints:
  - Follow HPC rules (hostname check, SSH to compute nodes)
  - Use R/Rscript commands (never bash scripts) for auto-approve
  - Update COVERAGE_MATRIX after changes
Success criteria:
  - [Measurable: all tests pass, all entries classified, etc.]
```

### 13.4 MPI Decision

MPI support has been **removed** from the codebase. The `distribute/` directory, `R/nmf_mpi.R`, and all associated code were deleted. For large-scale analysis, use GPU acceleration with streaming SPZ I/O.

---

## Appendix A: Decision Log Template

```markdown
### Decision: [Title]
**Date**: YYYY-MM-DD
**Context**: What prompted this decision?
**Options**: 
1. Option A — pros/cons
2. Option B — pros/cons
**Chosen**: Option X because...
**Status**: Implemented / Pending
```

Maintain at `docs/dev/DECISION_LOG.md`.

## Appendix B: Quick Reference — Where to Edit

| "I want to..." | Edit this file |
|-----------------|---------------|
| Add a new R parameter to `nmf()` | `R/nmf_thin.R` + `R/nmf_validation.R` |
| Add corresponding C++ config field | `inst/include/FactorNet/core/config.hpp` |
| Wire R param to C++ | `src/RcppFunctions_nmf.cpp` |
| Add a GPU kernel | `inst/include/FactorNet/primitives/gpu/` or `inst/include/FactorNet/gpu/` |
| Add a new distribution/loss | `inst/include/FactorNet/math/loss.hpp` + `primitives/cpu/nnls_batch_irls.hpp` |
| Add a C++ test | `tests/testthat/test_cpp_*.R` |
| Add an R wrapper test | `tests/testthat/test_r_*.R` |
| Add a benchmark | `benchmarks/harness/suites/*.R` |
| Document algorithm | `docs/factornet/algorithms/*.md` |
| Document for R users | Roxygen in `R/*.R` + `vignettes/*.Rmd` |

## Appendix C: Root File Cleanup

These should be archived to `docs/dev/proposals/` or `docs/dev/sessions/`:

```
DISTRIBUTION_API_PROPOSAL.md → docs/dev/proposals/
GP_NMF_DEVELOPMENT_PLAN.md → docs/dev/proposals/
GP_NMF_IMPLEMENTATION_PLAN.md → docs/dev/proposals/
GPU_OPTIMIZATION_AUDIT.md → docs/dev/
GPU_ZIGP_ZINB_PROPOSAL.md → docs/dev/proposals/
GRAPH_WIRING_PROPOSAL.md → docs/dev/proposals/
GUIDED_NMF_PROPOSAL.md → docs/dev/proposals/
RESTRUCTURE_PLAN.md → docs/dev/proposals/
SESSION7_PROGRESS.md → docs/dev/sessions/
WEIGHTED_MASKED_SVD_PROPOSAL.md → docs/dev/proposals/
ZIGP_PRODUCTION_ASSESSMENT.md → docs/dev/
```
