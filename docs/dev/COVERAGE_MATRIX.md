# Coverage Matrix — FactorNet / RcppML

> **Auto-derived from `COVERAGE_MATRIX.yaml` Rev 4 (2026-03)**
>
> This document provides human-readable summary tables for the functional
> coverage matrix. The YAML file is the authoritative source; regenerate
> this document after any YAML edits.
>
> **Rev 4 changes**: Fixed SPZ/dense framing — dense streaming uses SPZ v3
> dense format (not a separate DataLoader). Added full cross-product entries:
> CV × GPU for all variants, all init modes on GPU, graph streaming for
> deep/multi-modal/branching, dense entries for all features. Added "fallback"
> status for GPU projective/symmetric CV (dispatches to CPU). ~320 total entries.

---

## Status Distribution

| Status | Count | Description |
|--------|------:|-------------|
| **validated** | ~4 | Ground-truth or mathematical property verification |
| **tested** | ~65 | R-level tests (smoke/property/regression) |
| **implemented** | ~95 | Code exists, no test coverage |
| **planned** | ~130 | Target feature — not yet implemented |
| **broken** | 6 | Known incorrect behavior (ZI twoway) |
| **unsupported** | 4 | Theoretically invalid — needs guards |
| **deprecated** | 2 | MAE, Huber (subsumed by robust_delta) |
| **fallback** | 2 | GPU → CPU fallback (projective/symmetric CV) |
| **TOTAL** | **~320** | |

## Validation Level Distribution

| Level | Count | Meaning |
|-------|------:|---------|
| ground_truth | ~4 | Compared against known analytical solution |
| property | ~25 | Verified a mathematical invariant |
| parity | ~8 | Two backends produce matching results |
| regression | ~3 | Matches a known-good output value |
| smoke | ~35 | Runs without error; correct type/shape |
| none | ~245 | No tests (includes all planned + implemented-untested) |

---

## Group-by-Group Summary

### G1 — Core Algorithm × Backend × Input Type

Four algorithm families (NMF, SVD, NNLS, LS) × four backends (CPU, GPU, Streaming-CPU, Streaming-GPU) × two input types (sparse, dense).

#### NMF

| Backend × Input | Status | Notes |
|----------------|--------|-------|
| CPU sparse | tested/property | Primary path, all features |
| CPU dense | tested/property | Same fit_cpu.hpp code |
| GPU sparse | tested/parity | cuSPARSE + cuBLAS |
| GPU dense | tested/smoke | fit_gpu_dense.cuh |
| Streaming CPU sparse | tested/parity | SPZ chunked |
| **Streaming CPU dense** | **planned** | SPZ v3 dense format needed |
| Streaming GPU sparse | tested/smoke | Auto VRAM chunking |
| **Streaming GPU dense** | **planned** | After SPZ v3 + GPU streaming |

#### SVD

| Backend × Input | Status | Notes |
|----------------|--------|-------|
| CPU sparse | tested/property | All 5 methods |
| CPU dense | tested/smoke | All methods |
| GPU sparse | implemented | All 5 GPU variants, zero tests |
| GPU dense | implemented | 4 methods (krylov dense not impl) |
| Streaming CPU sparse | tested/smoke | Out-of-core SVD via SPZ |
| **Streaming CPU dense** | **planned** | SPZ v3 dense format needed |
| **Streaming GPU sparse** | **planned** | GPU streaming matvec |
| **Streaming GPU dense** | **planned** | After SPZ v3 |

#### NNLS (standalone)

| Backend × Input | Status | Notes |
|----------------|--------|-------|
| CPU sparse | validated/ground_truth | Primary ground-truth |
| CPU dense | validated/ground_truth | Dense parity with sparse |
| GPU sparse | implemented | Not exposed as standalone R function |
| **GPU dense** | **planned** | cuBLAS GEMM |
| **Streaming CPU sparse** | **planned** | Stream columns via SPZ |
| **Streaming CPU dense** | **planned** | SPZ v3 |
| **Streaming GPU sparse** | **planned** | After CPU streaming NNLS |
| **Streaming GPU dense** | **planned** | After SPZ v3 |

#### LS (unconstrained / Semi-NMF solve)

| Backend × Input | Status | Notes |
|----------------|--------|-------|
| **CPU sparse** | **planned** | nonneg=FALSE Cholesky solve |
| **CPU dense** | **planned** | Same solver |
| **GPU sparse** | **planned** | GPU LS solver |
| **GPU dense** | **planned** | cuSOLVER / cuBLAS |
| **Streaming CPU sparse** | **planned** | After standalone LS |
| **Streaming CPU dense** | **planned** | After SPZ v3 |
| **Streaming GPU sparse** | **planned** | After CPU streaming LS |
| **Streaming GPU dense** | **planned** | After SPZ v3 |

---

### G2 — Distribution × Backend × Input Type

Eight IRLS distributions + robust modifier. MSE is special (bare CD, no IRLS).

| Distribution | CPU Sparse | CPU Dense | GPU Sparse | GPU Dense | Stream CPU | Stream GPU |
|-------------|-----------|----------|-----------|----------|-----------|-----------|
| **MSE** | tested | tested | tested | tested | tested | tested |
| **GP** | tested | tested | impl | planned | impl | impl |
| **NB** | tested | tested | impl | planned | impl | impl |
| **Gamma** | tested | impl | impl | planned | impl | impl |
| **InvGauss** | tested | impl | impl | planned | impl | impl |
| **Tweedie** | tested | impl | impl | planned | impl | impl |
| **Robust MSE** | impl | planned | impl | planned | planned | planned |
| **Robust GP** | impl | planned | impl | planned | planned | planned |
| **MAE** | deprecated | — | — | — | — | — |
| **Huber** | deprecated | — | — | — | — | — |

> Dense streaming for all distributions depends on **SPZ v3 dense format** (planned).
> GPU dense for non-MSE distributions depends on **GPU dense IRLS kernel** (planned).

---

### G3 — Distribution × Zero-Inflation

Only GP and NB support zero-inflation. ZI twoway is **BROKEN** on all backends.

| Distribution × ZI Mode | CPU | GPU | Dense CPU | Dense GPU |
|------------------------|-----|-----|-----------|-----------|
| GP zi_none | tested | — | — | — |
| GP zi_row | tested | tested | planned | planned |
| GP zi_col | impl | impl | — | — |
| GP **zi_twoway** | **BROKEN** | **BROKEN** | — | — |
| NB zi_none | tested | — | — | — |
| NB zi_row | impl | impl | planned | planned |
| NB zi_col | impl | impl | — | — |
| NB **zi_twoway** | **BROKEN** | **BROKEN** | — | — |

Unsupported ZI combos (need guards): MSE, Gamma, InvGauss, Tweedie × any ZI mode.

---

### G4 — Cross-Validation × Distribution × Backend × Input Type

CV is an orthogonal axis: speckled mask or full mask. GPU CV falls back to CPU for projective/symmetric.

#### MSE × CV

| CV Mode × Backend | Sparse | Dense |
|-------------------|--------|-------|
| Speckled CPU | tested | impl |
| Full CPU | tested | impl |
| Speckled GPU | impl | impl (via sparseView) |
| Full GPU | impl | impl (via sparseView) |

#### Non-MSE × CV (GP, NB, Gamma, InvGauss, Tweedie)

| Distribution | CPU Speckled | CPU Full | CPU Dense | GPU Speckled | GPU Full | GPU Dense |
|-------------|-------------|---------|----------|-------------|---------|----------|
| GP | tested | impl | impl | impl | planned | planned |
| NB | impl | impl | impl | impl | planned | planned |
| Gamma | impl | impl | impl | planned | planned | planned |
| InvGauss | impl | impl | impl | planned | planned | planned |
| Tweedie | impl | impl | impl | planned | planned | planned |

#### Streaming × CV

| Backend | Sparse | Dense |
|---------|--------|-------|
| Streaming CPU | impl | planned |
| **Streaming GPU** | **planned** | **planned** |

#### NMF Variant × CV

| Variant × Backend | CPU Sparse | CPU Dense | GPU Sparse | GPU Dense |
|-------------------|-----------|----------|-----------|----------|
| Projective | impl | impl | **fallback** (→CPU) | planned |
| Symmetric | impl | impl | **fallback** (→CPU) | planned |

---

### G5 — NNLS Solver Configuration

| Solver | Distribution | CPU Sparse | CPU Dense | GPU Sparse | GPU Dense |
|--------|-------------|-----------|----------|-----------|----------|
| CD | MSE | tested | tested | tested | impl |
| CD | IRLS (any) | tested | tested | impl | **planned** |
| Cholesky | MSE | tested | impl | planned | — |
| Cholesky | IRLS | unsupported | — | — | — |

> **GPU dense IRLS** is a critical dependency: blocks ~15 GPU dense non-MSE entries.

---

### G6 — Factor Scaling / Normalization

| Scaling | CPU | GPU | Streaming CPU | Streaming GPU |
|---------|-----|-----|--------------|--------------|
| L1 | tested | tested | impl | impl |
| L2 | tested | impl | impl | impl |
| None | tested | impl | impl | impl |

---

### G7 — NMF Variants × Backend × Input Type

| Variant | CPU Sparse | CPU Dense | GPU Sparse | GPU Dense | Stream CPU | Stream GPU |
|---------|-----------|----------|-----------|----------|-----------|-----------|
| Projective | tested | impl | impl | planned | planned | planned |
| Symmetric | tested | impl | impl | planned | planned | planned |
| **Semi-NMF** | **planned** | **planned** | **planned** | **planned** | **planned** | **planned** |

> Semi-NMF (W unconstrained) is nearly free given existing solver — add `nonneg_W=FALSE` option.

---

### G8 — Guided / Informed NMF

| Feature | CPU Sparse | CPU Dense | GPU | Streaming CPU | Streaming GPU |
|---------|-----------|----------|-----|--------------|--------------|
| Seed W | impl | planned | planned | planned | planned |
| Seed H | impl | planned | planned | planned | planned |
| Seed both | impl | planned | planned | planned | planned |

---

### G9 — Streaming Architecture (SPZ DataLoader)

| Component | Sparse | Dense |
|-----------|--------|-------|
| SpzLoader read | tested | **planned** (SPZ v3) |
| SpzLoader write | tested | **planned** (SPZ v3) |
| Auto-chunking CPU | impl | — |
| Auto-chunking GPU | impl | — |

> Dense streaming is **NOT** a separate DataLoader class. The existing `SpzLoader` is extended
> with a format flag for SPZ v3 dense, decompressing panels into `Eigen::MatrixXf`.

---

### G10 — Regularization

| Reg Type | CPU | GPU | Streaming CPU | Streaming GPU | Dense |
|----------|-----|-----|--------------|--------------|-------|
| L1 on W | tested | impl | impl | impl | impl |
| L1 on H | tested | impl | impl | impl | impl |
| L2 on W | tested | impl | impl | impl | impl |
| L2 on H | tested | impl | impl | impl | impl |

---

### G11 — Graph / Hierarchical NMF

| Structure | CPU Sparse | CPU Dense | GPU Sparse | GPU Dense | Stream CPU | Stream GPU |
|-----------|-----------|----------|-----------|----------|-----------|-----------|
| Single-layer | tested | impl | impl | impl | impl | impl |
| Deep | tested | impl | impl | planned | **planned** | **planned** |
| Multi-modal | impl | planned | planned | planned | **planned** | **planned** |
| Branching | impl | planned | planned | planned | **planned** | **planned** |

> Single-layer streaming works (delegates to streaming NMF). Deep/multi-modal/branching
> streaming requires intermediate factor materialization — fundamentally harder.

---

### G12 — Clustering

| Component | CPU | GPU | Streaming CPU | Streaming GPU |
|-----------|-----|-----|--------------|--------------|
| dclust | tested | impl | planned | planned |
| bipartiteMatch | tested | planned | — | — |
| consensus NMF | tested | impl | planned | planned |

---

### G13 — SVD Methods × Auto-Selection

| Method | CPU Sparse | CPU Dense | GPU Sparse | GPU Dense |
|--------|-----------|----------|-----------|----------|
| Deflation | tested | impl | impl | impl |
| Krylov | tested | impl | impl | **planned** |
| Lanczos | tested | impl | impl | impl |
| IRLBA | tested | impl | impl | impl |
| Randomized | tested | impl | impl | impl |
| Auto-select | tested | — | **planned** | — |

---

### G14 — Masking

| Mask Mode | CPU Sparse | CPU Dense | GPU Sparse | GPU Dense |
|-----------|-----------|----------|-----------|----------|
| NA mask | tested | impl | impl | planned |
| mask_zeros | tested | impl | impl | planned |

---

### G15 — Loss Computation

| Mode | CPU | GPU | Streaming CPU | Streaming GPU |
|------|-----|-----|--------------|--------------|
| Dense eval | tested | impl | — | — |
| Sparse eval | tested | impl | — | — |
| Per-iteration | tested | impl | impl | impl |

---

### G16 — Initialization × Backend

All init modes work on GPU (init on host CPU → upload to device).

| Init Mode | CPU Sparse | CPU Dense | GPU Sparse | GPU Dense |
|-----------|-----------|----------|-----------|----------|
| SVD/Lanczos | tested | impl | impl | impl |
| SVD/IRLBA | tested | impl | impl | impl |
| Random | tested | impl | impl | impl |
| User-supplied | tested | impl | impl | impl |
| Streaming (any) | — | — | impl | impl |

---

### G17 — SPZ File Format

| Feature | Status |
|---------|--------|
| v2 read (sparse) | tested |
| v2 write (sparse) | tested |
| **v3 read (dense)** | **planned** |
| **v3 write (dense)** | **planned** |
| Metadata | tested |
| R bindings | tested |

> SPZ v3 dense format enables **all** dense streaming entries (~25 across the matrix).

---

### G18 — R-Level API & Dispatch

| Component | Status |
|-----------|--------|
| nmf() dispatch | tested |
| Parameter validation | tested |
| S4 class & methods | tested |
| Factor alignment | tested |
| CV dispatch | tested |
| GPU auto-dispatch | impl |
| Streaming auto-dispatch | impl |
| distribution= param | tested |
| robust_delta= param | impl |
| zi_mode= param | tested |

---

### G19 — Edge Cases & Guards

| Edge Case | Status |
|-----------|--------|
| k=1 | tested |
| k=min(m,n) | impl |
| Empty columns | tested |
| All-zero rows | impl |
| Single nonzero | impl |
| tol=0 | tested |
| maxit=1 | impl |
| dgCMatrix only | tested |
| Negative entries | tested |
| NaN/Inf input | impl |
| GPU OOM | impl |
| Thread safety | impl |

---

## Key Dependency Chains

| Dependency | Unblocks |
|-----------|----------|
| **SPZ v3 dense format** | ~25 dense streaming entries |
| **GPU dense IRLS kernel** | ~15 GPU dense non-MSE entries |
| **Native GPU dense CV** | ~10 GPU dense CV entries |
| **CPU streaming projective/symmetric** | ~8 streaming variant entries |

## Critical Fixes Needed

1. **ZI twoway** — 6 broken entries (GP+NB × CPU+GPU). Fix with damping/cap or remove.
2. **GPU SVD tests** — 10 implemented GPU SVD methods with zero test coverage.
3. **GPU CV parity tests** — Multiple implemented GPU CV paths with no tests.
4. **Unsupported ZI guards** — 4 unsupported combos (MSE/Gamma/InvGauss/Tweedie × ZI) need R+C++ guards.

---

## Infrastructure

| Component | Status |
|-----------|--------|
| Rcpp exports | auto-generated (never edit) |
| Roxygen docs | auto-generated (never edit) |
| Rcpp info bug fix | tools/fix_rcpp_info_bug.sh |
| GPU CI | planned |
| C++ unit tests | planned (no framework yet) |
| GPU test harness | planned (skip_if_no_gpu) |
