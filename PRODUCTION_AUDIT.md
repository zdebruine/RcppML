# RcppML / FactorNet — Production Audit

**Updated**: March 2026  
**Status**: Active reference document. Supersedes `HARDENING_PLAN.md` (archived to
`docs/dev/`) and `WORKSTREAMS.md` (archived to `docs/dev/`).  
**Purpose**: Authoritative record of what is complete, what is outstanding, and
what needs architectural rework before CRAN/publication release.

---

## Table of Contents

1. [What Is Complete](#1-what-is-complete)
2. [Outstanding Items](#2-outstanding-items)
3. [Streaming Out-of-Core: Architecture Critique](#3-streaming-out-of-core-architecture-critique)
4. [CRAN Submission Checklist](#4-cran-submission-checklist)
5. [Publication Readiness](#5-publication-readiness)
6. [StreamPress Architecture Revision Plan](#6-streampress-architecture-revision-plan)

---

## 1. What Is Complete

### 1.1 Package Hygiene
- ✅ `.Rbuildignore` — excludes benchmarks/, manuscript/, tools/, experiments/,
  GPU source (src/*.cu), .github/, .vscode/, planning docs
- ✅ `DESCRIPTION` — correct author metadata, no VignetteBuilder (vignettes excluded)
- ✅ `devtools::document()` + `tools/fix_rcpp_info_bug.sh` workflow established
- ✅ `R CMD build` produces clean tarball

### 1.2 Guards & Correctness
- ✅ `zi="twoway"` disabled — throws clear error at R level (`nmf_thin.R:543`)
  and C++ gateway level (`config.hpp:365`); covered in `test_unsupported_combos.R`
- ✅ ZI × invalid distribution guards — 4 guards in `nmf_thin.R:556–568`
  (`mse`, `gamma`, `inverse_gaussian`, `tweedie`); 11 tests in
  `test_unsupported_combos.R`
- ✅ Cholesky + non-MSE guard — `nmf_thin.R:589–594`; tests pass
- ✅ Streaming + NB theta=0 bug — fixed; `theta=config.nb_size_init` when
  `nb_size_init > 0` prevents silent division-by-zero in IRLS

### 1.3 Test Suite
- ✅ **CPU tests** (c001): `[ FAIL 0 | WARN 15 | SKIP 148 | PASS 1987 ]`
- ✅ **GPU tests** (g051): `[ FAIL 0 | WARN 17 | SKIP 20 | PASS 2484 ]`
- ✅ SVD ground-truth tests — `test_svd.R` uses `gt_m=100; gt_n=80` for stable
  Krylov/Lanczos on Intel Xeon AVX-512
- ✅ Streaming SPZ test — `sp_write(..., include_transpose=TRUE)` in
  `test_file_input.R`
- ✅ GPU skip pattern — all GPU-sensitive tests use
  `skip_if(!identical(getOption("RcppML.gpu", FALSE), FALSE), ...)`,
  which correctly catches TRUE, "auto", and any non-FALSE value
- ✅ `diag$plan` check — `grepl("^(CPU|GPU)", diag$plan)` in
  `test_unified_backend.R:234`; handles "GPU (dlsym bridge)", etc.
- ✅ factor_net comparison tests — explicit `resource="cpu"` in both
  `factor_config()` and `nmf()` calls; immune to GPU auto-dispatch
  contamination from different test ordering

### 1.4 R CMD check
- ✅ `R CMD check --as-cran` — Status: **1 WARNING** only
- The warning is `checkbashisms` script not installed on c001; this is a
  system environment artifact (the script is not part of R CMD check itself).
  The configure script is POSIX-compliant. Will not appear on CRAN's
  Linux infrastructure.

### 1.5 Feature Implementation
- ✅ Unified NMF gateway (`nmf/gateway/nmf_full.hpp` → `fit_unified.hpp`)
- ✅ Cross-validation NMF with lazy mask + per-column Gram correction
- ✅ GPU IRLS for GP, NB, Gamma, InvGauss, Tweedie
- ✅ GPU ZI (row/col modes) for GP and NB
- ✅ Streaming CPU NMF from SPZ v2 (sparse) and v3 (dense) via `fit_chunked.hpp`
- ✅ Streaming GPU NMF via `fit_chunked_gpu.cuh`
- ✅ FactorNet graph API (multi-modal, deep, branching)
- ✅ All SVD methods: deflation, Krylov, Lanczos, IRLBA, randomized
- ✅ Guided NMF (classifier + external guide)
- ✅ Full regularization suite: L1, L2, L21, angular, graph Laplacian, upper bounds

---

## 2. Outstanding Items

Items are listed in priority order within each category.

### 2.1 Critical (Must Fix Before CRAN)

#### 2.1.1 "Planned" Entry Audit (Workstream B.4)

The coverage matrix (`docs/dev/COVERAGE_MATRIX.yaml`) has 123 entries with
`status: planned`. These have NOT been audited for user-reachability. Every
user-reachable "planned" path must either be implemented or guarded with a
clear error before CRAN submission.

**Key paths to audit:**
- Dense streaming (SPZ v3): does `nmf("file.spz", ...)` with a dense `.spz`
  silently misfire or hit an unimplemented branch?
- Semi-NMF (`nonneg_W = FALSE`): is this a user-facing parameter? If so,
  test that it reaches implemented code and doesn't silently produce W factors
  with NaN values.
- GPU dense + non-MSE: does GPU dispatch with dense matrix + loss="gp" work,
  fall back gracefully, or crash?
- Standalone GPU NNLS: is `nnls()` R function exposed for GPU acceleration? If
  not, is there a misleading parameter suggesting it?

**Action**: Trace every `planned` entry from R API → validation → C++ dispatch.
For each user-reachable one: (a) implement, or (b) add guard. Update
`COVERAGE_MATRIX.yaml`.

#### 2.1.2 Vignettes Excluded from Package

Two vignettes are excluded in `.Rbuildignore`:
- `vignettes/sparsepress.Rmd`
- `vignettes/nmf-deep-dive.Rmd`

These need to be either fixed and re-included, or permanently excluded with an
explanation in the vignette header (`\dontrun{}` pattern or precomputed
approach). CRAN expects all included vignettes to build cleanly; excluded
vignettes are fine if intentional, but important user documentation should be
accessible via pkgdown even if not built by CRAN.

**Action**: Attempt `rmarkdown::render()` on each. Fix build failures. If
examples are too slow, use `eval=FALSE` or precomputed results.

#### 2.1.3 Example Runtimes

All `man/*.Rd` examples must run in under 5 seconds on CRAN hardware.

**Likely offenders** (need manual check):
- `nmf()` examples with large `k` on `movielens` data
- `cv_nmf()` examples (multiple NMF fits)
- SPZ streaming examples
- GPU examples

**Action**: Run `devtools::check_examples()` and wrap slow examples in
`\donttest{}`.

#### 2.1.4 Package Size

CRAN tarball limit is 5 MB. Check `file.size("RcppML_1.0.1.tar.gz") / 1e6`.

If over 5 MB, candidates for trimming:
- `data/digits_full.rda` (784-dim full MNIST — large)
- `data/olivetti.rda` (4096-dim face images — large)
- `inst/extdata/pbmc3k.spz`

**Action**: Check tarball size. If > 5 MB, trim data objects or move to
separate data package.

### 2.2 Significant (Fix Before Release, OK to Defer for CRAN)

#### 2.2.1 NB Dispersion Not Updated in Streaming Path ✅

> **Status**: Fixed (R-level warning). `R/nmf_thin.R` warns when `loss="nb"` in streaming mode without pre-estimated `nb_size`. Full C++ dispersion re-estimation deferred.

**File**: `inst/include/FactorNet/nmf/fit_chunked.hpp`  
**Line**: ~205–215 (nb_size_vec initialization)

`nb_size_vec` is initialized from `config.nb_size_init` and **never updated
during NMF iteration** in the streaming path. The in-memory NB path
(`fit_unified.hpp`) updates the dispersion vector each iteration via
moment-matching or Newton-Raphson. This means streaming NB NMF silently uses
a fixed dispersion — incorrect if dispersion estimation is needed.

**Impact**: Medium. The user must pass a pre-estimated `nb_size` parameter.
If they don't, results will be wrong (not incorrect bounds — wrong estimates).

**Fix**: After each complete pass through the transpose chunks (end of W-update
in `fit_chunked.hpp`), re-estimate `nb_size_vec` from residuals. This requires
a third pass through the data per iteration (or a clever approximation from the
existing two passes' accumulated statistics).

**Workaround**: Document that streaming NB requires `nb_size` to be pre-specified.
Add a warning when `loss="nb"` and `nb_size` is not provided in streaming mode.

#### 2.2.2 `mask_zeros=FALSE` CV Is O(m·n) in Streaming Path ✅

> **Status**: Fixed. Added `holdout_zero_rows()` and `holdout_zero_cols()` to `speckled_cv.hpp` using SplitMix64 hash sampling. Added Gram-trick zero-entry branch in `fit_chunked.hpp`. No longer O(m) per column.

**File**: `inst/include/FactorNet/nmf/fit_chunked.hpp`  
**Lines**: ~332–346, ~384–396 (per-column loop, zero-entry loop)

When `mask_zeros=FALSE` (full-matrix CV), the streaming path scans ALL `m`
rows for every column j to find zero entries that might be holdout:

```cpp
for (uint32_t i = 0; i < m; ++i) {
    if (ni < nz_rows.size() && nz_rows[ni] == (int)i) { ++ni; continue; }
    if (cv_mask->is_holdout(i, gj))
        excl_rows.push_back((int)i);
}
```

This is O(m) per column — O(m·n) total per H-update pass, which is identical
to dense matrix complexity. It defeats the point of sparse streaming for dense
CV. Similarly in the W-update transpose pass.

**Impact**: High for large sparse matrices with `mask_zeros=FALSE`. In practice
`mask_zeros=TRUE` (sparse CV) is the natural choice for SPZ data
(recommendation-style), so most users won't hit this. Still should be fixed.

**Fix**: Pre-generate the zero-entry holdout set per panel by sampling
uniformly with the same hash function, without iterating dense. The
`LazySpeckledMask` already uses a hash — extend it to enumerate holdout zero
entries within a column range without scanning all rows.

#### 2.2.3 SVD Init Decompresses Full Matrix During Streaming ✅

> **Status**: Fixed. Added RAM check (70% threshold) in `fit_streaming_spz.hpp`. Falls back to random init with warning. `platform.hpp` provides `get_available_ram_bytes()` utility.

**File**: `inst/include/FactorNet/nmf/fit_streaming_spz.hpp`  
**Lines**: ~115–168

When `init_mode=1` (Lanczos) or `init_mode=2` (IRLBA), the streaming entry
point decompresses the entire matrix into a dense `SpMat A_full` in memory:

```cpp
// Decompress all forward chunks into a full sparse matrix
SpMat A_full(m, n);
// ... fills A_full from chunks ...
detail::initialize_lanczos(W_T_svd, H_svd, d_svd, A_full, ...);
```

This completely negates the memory benefit of streaming for SVD initialization.
A matrix that requires streaming precisely because it doesn't fit in memory
will OOM here during SVD init.

**Workarounds that exist**:
- Random init (`init_mode=0`) avoids this — no decompression needed.
  Already the default for streaming.
- The code logs: `"Building temporary matrix for Lanczos init..."` so the
  user can observe this happening.

**Impact**: High if SVD init is requested with a matrix that doesn't fit in RAM.
Low in common usage since random init is the default for streaming.

**Fix**: Implement streaming Lanczos power iteration that processes data
column-by-column through the DataLoader, maintaining only the k Lanczos vectors
in memory. This is similar to the streaming SVD in `svd/streaming.hpp` —
check if that code can be reused. If streaming SVD already works out-of-core,
wire it into the streaming NMF init path.

**Short-term**: Add a check: if `init_mode != 0` and the file is large
(heuristic based on `loader.nnz()` and available RAM), warn and fall back to
random init.

#### 2.2.4 `std::async` Per Chunk Creates Thread Per I/O Operation ✅

> **Status**: Fixed. Replaced with `PingPongPrefetcher` (persistent background thread + 2-slot ping-pong buffer) in `io/ping_pong_prefetch.hpp`. All 3 async sites in `fit_chunked.hpp` replaced. Race condition in shared `write_idx_` found and fixed (separate `read_idx_` and `write_target`).

**File**: `inst/include/FactorNet/nmf/fit_chunked.hpp`  
**Lines**: ~289, ~462, ~493 (all `std::async(std::launch::async, ...)`)

The double-buffered prefetch creates a new thread for every single chunk:

```cpp
auto prefetch = std::async(std::launch::async, [&loader, &chunk_next]() {
    return loader.next_forward(*chunk_next);
});
```

Spawning a thread per chunk has non-trivial overhead on most platforms, and the
future's `.get()` is a full synchronization point. For a file with hundreds of
chunks, this creates hundreds of short-lived threads.

**Better pattern**: Use a persistent background I/O thread that dequeues
requests from a work queue, or a pre-allocated thread from a pool. The current
pattern technically works and overlaps compute with I/O since `std::launch::async`
guarantees a new thread, but is inefficient for many small chunks.

**Impact**: Marginal for large chunk sizes (e.g., 10K columns per chunk) where
the thread startup overhead (~10–50 μs) is negligible vs the I/O time. Matters
for small chunk sizes or fast storage (NVMe where I/O is μs-scale).

**Fix**: Replace double-buffered `std::async` with a persistent background
thread using `std::thread` + a 2-slot ping-pong buffer with `std::condition_variable`
synchronization. See: `profiling/stream_timer.hpp` for an example threading pattern.

#### 2.2.5 `goto` Statement in Forward Pass ✅

> **Status**: Fixed. Replaced `goto forward_done` with `if (has_first)` conditional block wrapping the forward pass loop.

**File**: `inst/include/FactorNet/nmf/fit_chunked.hpp`  
**Line**: ~248 (`goto forward_done;`)  
**Line**: ~462 (`forward_done:` label)

The symmetric NMF path skips the entire forward H-update loop with `goto`:
```cpp
if (is_symmetric) {
    G_H.noalias() = W_T * W_T.transpose();
} else {
    loader.reset_forward();
    // ... 200 lines of forward chunk processing ...
    goto forward_done;  // <-- symmetric jumps here
}
...
forward_done:
```

This is a code maintainability issue. The `goto` jumps backward over a code
block that was never entered, which is confusing to read and may trigger
static analysis warnings.

**Fix**: Wrap the `else` branch in a function or use `do { } while(false)` with
`break`, or restructure with a lambda. E.g.:
```cpp
auto run_forward_pass = [&]() { /* ... */ };
if (!is_symmetric) run_forward_pass();
```

#### 2.2.6 Loss History Reports MSE for IRLS Losses ✅

> **Status**: Fixed. Added `compute_loss()` from `math/loss.hpp` for nonzero entries and IRLS-derived zero-entry loss branch in streaming per-column loop. Loss history now reports correct IRLS objective.

**File**: `inst/include/FactorNet/nmf/fit_chunked.hpp`  
**Lines**: ~440–463 (Gram-trick loss) and ~430–437 (per-column loss)

The Gram-trick batch path computes `train_loss_accum = trAtA - 2*cross_term + recon_norm`,
which is **squared Frobenius loss**. The per-column path accumulates `diff * diff`
at nonzero entries. Both are MSE approximations.

For `use_irls=true` (GP, NB, Gamma, etc.), the solver is IRLS but the
convergence check uses MSE. This means:
1. `result.loss_history` contains MSE values, not GP/NB negative log-likelihood
2. Convergence tolerance is checked against MSE change, not the actual IRLS objective

**Impact**: Medium. The MSE proxy usually tracks the IRLS objective adequately
for convergence, but the reported loss values are misleading (labeled as
"loss" but actually MSE). The `test_streaming.R:TEST-STREAM-LOSS-TRACK` test
verifies that `decreasing_loss=TRUE`, which would catch a loss that's not
decreasing in MSE terms.

**Fix**: When `use_irls=true` in streaming mode, compute the per-column IRLS
loss (GP log-likelihood, NB log-likelihood, etc.) alongside the MSE at each
nonzero entry. This is already done per-column in `fit_unified.hpp`; port that
logic into the streaming path's per-column loop.

### 2.3 Documentation Gaps (Required for Publication)

#### 2.3.1 README.md Not Updated

Current README reflects an older version of the API. Missing:
- FactorNet graph DSL (`factor_net()`, `link_nmf()`, etc.)
- Statistical distributions (GP, NB, Gamma, InvGauss, Tweedie)
- GPU acceleration section
- Streaming / SPZ section
- New parameters: `zi`, `distribution`, `guided_nmf`
- Installation from GitHub

**Action**: Full rewrite of README.md. See `HARDENING_PLAN.md §11.1` for
vignette plan (reusable content structure).

#### 2.3.2 Roxygen Coverage Gaps

Recently added functions that likely lack complete documentation:
- `factor_net()`, `link_nmf()`, `link_svd()`, graph DSL functions
- `auto_nmf_distribution()`, `diagnose_dispersion()`, `diagnose_zero_inflation()`
- `gpu_available()`, `gpu_info()`, `sp_read_gpu()`, `sp_free_gpu()`
- `classify_embedding()`, `classify_logistic()`, `classify_rf()`
- `training_logger()`, `export_log()`

**Action**: Run `devtools::check_man()`. Add `@param`, `@return`, `@examples`,
`@seealso` to each gap. Wrap slow examples in `\donttest{}`.

#### 2.3.3 ~~Developer Documentation Stubs~~ — ALREADY DONE

**AUDIT FINDING**: `docs/factornet/gpu/README.md` (182 lines) and
`docs/factornet/io/README.md` (192 lines) both exist with real content.
`docs/factornet/ARCHITECTURE.md` (276 lines) and
`docs/factornet/API_REFERENCE.md` (689 lines) also exist.

**No action needed.** These were written in a prior session.

#### 2.3.4 ~~`docs/factornet/algorithms/` Content~~ — ALREADY DONE

**AUDIT FINDING**: All 9 algorithm documents already exist under
`docs/factornet/algorithms/` with substantial content (210–293 lines each):
- `nmf.md` (257 lines), `irls.md` (238), `zero_inflation.md` (210)
- `cross_validation.md` (221), `svd.md` (293), `nnls.md` (238)
- `distributions.md` (281), `sparsepress.md` (227), `factornet_graph.md` (244)

**No action needed.** These were written in a prior session.

### 2.4 Performance Gaps (Defer to Phase 7)

Covered in `HARDENING_PLAN.md §7–8`. In priority order:

1. ~~**Adaptive CD↔Cholesky crossover**~~ ✅ — `solver="auto"` implemented in `R/nmf_thin.R`: Cholesky for k<32 + MSE + no L1, else CD.
2. ~~**Streaming double-buffer**~~ ✅ — Replaced with PingPongPrefetcher (see §2.2.4).
3. **Benchmark harness** — `benchmarks/harness/` structure in
   `HARDENING_PLAN.md §7.2` not yet built

---

## 3. Streaming Out-of-Core: Architecture Critique

This section critically evaluates the streaming implementation in
`inst/include/FactorNet/nmf/fit_streaming_spz.hpp` and
`inst/include/FactorNet/nmf/fit_chunked.hpp` against best practices for
out-of-core matrix factorization.

### 3.1 Strengths

**Correct fundamental structure**: The forward pass (H-update) and transpose
pass (W-update) are clearly separated with double-buffered async chunk reading.
This is the right algorithm for out-of-core ALS.

**DataLoader abstraction**: `fit_chunked.hpp` takes a generic `DataLoader<Scalar>&`
rather than being tied to SPZ files. This means the same chunked NMF logic works
for in-memory loaders (testing, FactorNet graph layers) and streaming loaders
(SPZ files). Clean separation of I/O from compute.

**Memory claims correct**: The comment `Memory: O(m*k + n*k + max_chunk_nnz)`
is accurate. The factors W and H are fully in-memory (required for ALS — you
need H when computing W-update), and only one chunk of the data at a time is
materialized.

**Double-buffered chunk I/O**: The `std::async` prefetch overlaps I/O with
compute — the next chunk is being decompressed while the current chunk is
being processed. This is the correct approach to hide I/O latency.

**Speckled CV in streaming**: The `LazySpeckledMask` hash-based holdout works
correctly for `mask_zeros=TRUE` (sparse) CV — no precomputed mask needed,
just a hash of `(row, col, seed)`. This is efficient for streaming.

### 3.2 Critical Flaws

#### 3.2.1 SVD initialization defeats streaming purpose

*Severity: High*

SVD init (the default for in-memory NMF) decompresses the entire matrix:

```cpp
// fit_streaming_spz.hpp:~115
SpMat A_full(m, n);
// ... fills from all forward chunks ...
detail::initialize_lanczos(W_T_svd, H_svd, d_svd, A_full, ...);
```

If your matrix requires streaming because it doesn't fit in RAM, OOM occurs
here. The implementation notes that random init (`init_mode=0`) avoids this,
and random init is the actual default when `spz_path` is set. But the code path
exists and can be triggered by `init="lanczos"` or `init="irlba"`.

**Best practice**: Out-of-core SVD uses a single-pass randomized power iteration
(Halko, Martinsson, Tropp 2011) or streaming Lanczos. The chunk-compatible
streaming Lanczos (for SpMV rather than random access) is documented in
`svd/streaming.hpp` and `svd/streaming_matvec.hpp` — these should be wired
into the streaming NMF init path to make SVD init streaming-safe.

#### 3.2.2 NB dispersion parameter is fixed in streaming

*Severity: High for correctness; Medium for practical impact*

The `nb_size_vec` (NB inverse dispersion `r`) is set once at the start:

```cpp
nb_size_vec = DenseVector<Scalar>::Constant(m, config.nb_size_init);
```

and never updated during the streaming NMF iteration. The in-memory NB path
re-estimates dispersion from residuals each iteration (or each few iterations).
Streaming NB therefore uses fixed dispersion throughout, which is equivalent to
specifying a known dispersion upfront — a reasonable assumption if the user
passes `nb_size`, but silently wrong if default `nb_size_init=1` is used for
data with different overdispersion.

**Best practice**: Either (a) implement a streaming dispersion rescanning pass
at end of each iteration (adds a third pass, but dispersion estimation converges
quickly), or (b) document clearly that streaming NB requires user-supplied
`nb_size` and warn loudly (`warning("Streaming NB uses fixed dispersion..."))`
when the default is used.

#### 3.2.3 mask_zeros=FALSE CV is O(m·n) per iteration

*Severity: High for large matrices*

In the per-column H-update and W-update loops, finding holdout zero entries
requires scanning all `m` rows (or all `n` columns in W-update):

```cpp
for (uint32_t i = 0; i < m; ++i) {            // O(m) per column
    if (nz_rows[ni] == (int)i) { ++ni; continue; }
    if (cv_mask->is_holdout(i, gj)) excl_rows.push_back((int)i);
}
```

For a 100K×200K matrix, this is 100K iterations × 200K columns = 2×10¹⁰
operations per H-update pass — indistinguishable from dense complexity.

`mask_zeros=TRUE` (the default for sparse data) avoids this by only examining
existing nonzeros. But `mask_zeros=FALSE` is the only option for dense CV
and is explicitly supported.

**Best practice**: Reformulate the `LazySpeckledMask` to support range-based
enumeration: given a column j and row range [0, m), return only the holdout
rows using the hash function directly without iterating all m rows. Since the
mask uses a uniform hash (hold fraction f), the expected number of holdout
entries per column is f·m — these can be generated by hashing candidate rows
rather than testing all of them (e.g., using inverse hash sampling or
pre-sampled Poisson draws per column).

### 3.3 Moderate Issues

#### 3.3.1 Thread-per-chunk prefetch overhead

The `std::async(std::launch::async, ...)` call in the inner chunk loop spawns
and joins a thread for each chunk:

```cpp
auto prefetch = std::async(std::launch::async, [&loader, chunk_next]() {
    return loader.next_forward(*chunk_next);
});
// ... compute ...
bool has_next = prefetch.get();  // blocks until I/O done
```

For fast NVMe storage where SPZ decompression is the bottleneck (not disk I/O),
this works well because compute and decompression run concurrently. But for
network file systems (NFS — common on HPC) where I/O latency is high,
spawning threads per-chunk adds overhead.

**Better**: A persistent producer thread maintains a ring buffer of 2–4
pre-decompressed chunks. The consumer (main thread) simply picks up the next
ready chunk. This eliminates thread spawn/join overhead and can hold more
pre-fetched chunks.

#### 3.3.2 goto in symmetric NMF path

A `goto forward_done;` is used to skip the forward pass for symmetric NMF.
This is technically valid C++ but unusual and confusing. Should be a function
or `if (!is_symmetric) { ... }` block around ~250 lines. Does not affect
correctness, but hurts readability and maintenance.

#### 3.3.3 W_T variable name shadowing

`MatS W_T(k, m)` is declared at the top of the iteration loop (~line 182)
for use in the H-update, and re-declared inside the batch W-update path
(~line 650) shadowing the outer one. The inner `W_T` holds the NNLS-solved
result of `W^T`, not the pre-transposed W. Both are legitimate uses but the
same name is confusing.

#### 3.3.4 Loss history is MSE even for IRLS losses

When `use_irls=true`, `result.loss_history` stores per-iteration MSE
approximations, not the IRLS log-likelihood objective. The reported "loss"
is misleading when `distribution="gp"` or `distribution="nb"` is used.
Convergence checking against MSE is a reasonable proxy but should be
documented, not silently the case.

### 3.4 Best Practice Comparison

| Criterion | Current Implementation | Best Practice | Gap |
|-----------|----------------------|---------------|-----|
| Memory | O(m·k + n·k + chunk) ✅ | Same | None |
| I/O pattern | Sequential with double-buffer ✅ | Same | Minor (thread per chunk) |
| SVD init | Decompresses full matrix ❌ | Streaming Lanczos | Critical |
| NB dispersion | Fixed throughout ❌ | Re-estimated per iteration | High |
| mask_zeros=FALSE | O(m·n) per column ❌ | O(f·m) via hash sampling | High |
| Loss reporting | Always MSE ⚠️ | Distribution-appropriate | Medium |
| Convergence | MSE-based ✅ | Acceptable proxy | Minor |
| Error recovery | None ⚠️ | Partial-write detection | Medium |
| Thread model | Per-chunk async ⚠️ | Persistent producer thread | Minor |
| Code quality | `goto` ⚠️ | Lambda or block | Cosmetic |

### 3.5 Recommended Remediation Priority

1. **[Critical — correctness for large matrices]** Connect `svd/streaming.hpp`
   to streaming NMF SVD init so Lanczos/IRLBA work without full decompression.
2. **[High — correctness for NB streaming]** Add warning + documentation for
   fixed NB dispersion in streaming mode; implement streaming dispersion update
   or require explicit `nb_size`.
3. **[High — correctness for mask_zeros=FALSE CV]** Fix O(m·n) scan in per-column
   loop by extending `LazySpeckledMask` with range enumeration.
4. **[Medium — usability]** Report IRLS loss in `loss_history` when `use_irls=true`.
5. **[Minor — code quality]** Replace `goto` with function/block. Rename
   inner `W_T` variable. Replace per-chunk `std::async` with persistent thread.

---

## 4. CRAN Submission Checklist

```
Pre-submission:
  [ ] R CMD check --as-cran: 0 ERRORs, 0 WARNINGs (currently 1 WARNING — env artifact)
  [ ] Run devtools::check_examples() — all examples < 5 seconds
  [ ] Run devtools::check_man() — all exported functions documented
  [ ] Package tarball < 5 MB
  [ ] NEWS.md finalized for v1.0.1
  [ ] Vignettes: either build cleanly or confirmed-excluded in .Rbuildignore with reason
  [ ] pkgdown site builds cleanly (pkgdown::build_site())
  [ ] Git state: all tracked files committed, no untracked source files
  [ ] "planned" entries audit complete (§2.1.1)
  [ ] streaming NB dispersion documented (§2.2.1)

Tests:
  [✅] CPU tests: [ FAIL 0 | WARN 15 | SKIP 148 | PASS 1987 ] (c001)
  [✅] GPU tests: [ FAIL 0 | WARN 17 | SKIP 20 | PASS 2484 ] (g051)
  [ ] Streaming tests with NOT_CRAN=true (validate 10 tests in test_streaming.R)
  [ ] All test files tracked in git
```

---

## 5. Publication Readiness

From `HARDENING_PLAN.md §12`, planned publications remain future work. Immediate
prerequisite is completing §2.3 (documentation gaps) and §2.4 (benchmarks).

| Paper | Status | Blocker |
|-------|--------|---------|
| P1 — RcppML overview (JSS/R Journal) | Not started | Needs README, vignettes, benchmarks |
| P2 — GPU CV via per-column Gram (Bio/JCGS) | Not started | Needs benchmark harness |
| P3 — IRLS framework (Biostatistics) | Not started | Needs algorithm docs |
| P4 — StreamPress streaming (SoftwareX) | Not started | Needs §6 StreamPress revision + GEO k=64 benchmark |
| P5 — FactorNet graph (JMLR/NeurIPS) | Not started | Needs FactorNet graph docs |
| P6 — Constrained SVD (Comp Stats) | Not started | Needs SVD algorithm docs |

---

*This document replaces `HARDENING_PLAN.md` and `WORKSTREAMS.md`, both of which
have been archived to `docs/dev/`. The coverage matrix remains live at
`docs/dev/COVERAGE_MATRIX.yaml`.*

---

## 6. StreamPress Architecture Revision Plan

**Status**: ✅ All phases (0–7) implemented. See individual phase sections below.  
**Priority**: Implement after CRAN blockers in §2.1 are resolved.  
**Compatibility guarantee**: All existing sparse v2 `.spz` files (GEO reprocessed
corpus, ~1 TB) remain fully readable with zero format changes. Only the dense v3
format and loader infrastructure are being extended.

### 6.0 Context and Motivation

The `.spz` file format and the R API around it are currently named **SparsePress**
(`sp_*` functions, `sparsepress` C++ namespace). The format has grown to support
both sparse (v2) and dense (v3) matrices. The next phase of development elevates
it to a first-class streaming I/O library — **StreamPress** — with real
compression for dense panels, true out-of-core streaming (not just reading the
entire file into RAM), adaptive chunking for modern hardware, and
memory-aware dispatch that automatically selects the right execution mode.

The flagship use case for the StreamPress revision is: **`nmf(path_to_geo_spz,
k=64, loss="nb")`** on the full GEO reprocessed single-cell corpus, running on a
compute node with hundreds of GB RAM, demonstrated without any manual tuning
from the user. This is the primary benchmark target for Publication P4.

#### Key findings from architecture review

- **Critical loader bug**: Both `SpzLoader` and `DenseSpzLoader` read the
  *entire file into RAM* in their constructors (`file_data_.resize(file_size_);
  fread(...)`) — this completely defeats out-of-core streaming. A 50 GB `.spz`
  file requires 50 GB RAM just to open it. This is a correctness bug, not
  a performance issue.
- **Dense v3 has no compression**: Raw float32/float64 bytes only. The 48
  reserved bytes in `FileHeader_v3` are unused and can carry codec metadata.
- **Chunk size 256 is severely sub-optimal**: `DEFAULT_CHUNK_COLS=256` causes
  hundreds to thousands of PCIe transfers and kernel launches per epoch on GPU.
  The optimal range for modern hardware is 2048–4096+ columns per chunk.
- **Streaming I/O dominates at low k**: From `benchmarks/stream_opt/STREAMING_ANALYSIS.md`,
  I/O accounts for 71–73% of total time at k=8. Chunk size directly controls
  this overhead.
- **No `/proc/meminfo` on Windows**: Memory detection logic must be
  platform-conditional. Laptop users (Windows, macOS, 8–16 GB RAM) are an
  important target — auto-dispatch must handle low-memory environments safely.

### 6.1 Phase 0: Locate and Verify GEO Reprocessed Corpus

**Goal**: Confirm the exact paths and format of the GEO reprocessed `.spz`
files before any implementation work begins. The old `cellcensus_500k.spz`
and `cellcensus_900k.spz` benchmarking files have been deleted. The new
corpus comes from reprocessed GEO downloads stored somewhere under
`/mnt/projects/debruinz_project/`.

**Steps**:

1. Identify active SLURM jobs writing `.spz` output:
   ```bash
   squeue -u debruinz -t R -o "%j %Z %N" | grep -i spz
   ```
2. Enumerate finished `.spz` files:
   ```bash
   find /mnt/projects/debruinz_project/ -name "*.spz" 2>/dev/null | head -30
   du -sh /mnt/projects/debruinz_project/*.spz 2>/dev/null
   ```
3. Run header validation on a sample of 100 files:
   - Check magic bytes (`SPRZ`), version field, dimensions m × n
   - Confirm all files are v2 sparse (no v3 dense files expected in GEO corpus)
   - Log any files with unexpected version numbers or truncated headers
4. Record the authoritative directory path in this document once confirmed.

**✅ CONFIRMED (2026-03-08)**:
- **Corpus directory**: `/mnt/projects/debruinz_project/cellarium/pipeline/quant/`
- **File count**: 9,119 `counts.spz` files (9,121 total `.spz` including 3 test exports)
- **Total size**: 717 GB
- **Format**: All v2 sparse (`version=2`, magic `SPRZ`)
- **Active jobs**: `scgeo-v5b` (bigmem), `scgeo-v5c` (cpu), `scgeo-v5h` (gpu-h100) — still writing

**Compatibility guarantee**: No Phase 0 action modifies any v2 format bytes.
All subsequent phases are tested against this corpus before merge.

**Test script**: `tools/verify_spz_corpus.R` — to be written in Phase 7.

### 6.2 Phase 1: Fix SpzLoader — True Seek-Based Streaming [CRITICAL]

**Files**: `inst/include/FactorNet/io/spz_loader.hpp`,
`inst/include/FactorNet/io/dense_spz_loader.hpp`

This is the highest-priority fix — it is a **correctness bug**. The current
behaviour makes it impossible to stream a file larger than available RAM.

#### Approach

Replace the `file_data_` bulk-read approach with a seek-based random-access
reader that keeps only the header and chunk index table in RAM:

- **RAM resident** (small, O(num_chunks)): complete `FileHeader_v2`, all
  `ChunkDescriptor_v2` entries (24 bytes × num_chunks ≈ up to ~100 KB for a
  3M-cell matrix with chunk_cols=256, or ~12 KB at chunk_cols=2048)
- **Streamed on demand**: actual chunk bytes — decompressed, used, discarded

#### Platform-conditional I/O

True random-access file reads require different APIs per platform:

| Platform | API | Notes |
|----------|-----|-------|
| Linux/macOS (POSIX) | `pread(fd, buf, size, offset)` | Thread-safe, no cursor race, O_DIRECT possible |
| Windows | `ReadFile()` with `OVERLAPPED` + `LARGE_INTEGER` offset | Must be used instead of `pread` |
| NFS (all platforms) | `pread()` via standard POSIX | Works correctly on NFS |

The new `FileReader` abstraction in `inst/include/streampress/io/file_reader.hpp`
wraps this platform difference behind a single `pread(offset, buf, size)` call.
On Windows, it wraps `ReadFile()` with `OVERLAPPED`. This is the correct
foundation for Windows/laptop support throughout the codebase.

#### Optional in-core threshold

For small files (default threshold: 2 GB), auto-loading into RAM is still
optimal. The loader checks file size at open and decides:
```
if file_size < ram_threshold → legacy bulk-read (fast, RAM permitting)
else             → seek-based streaming (any file size)
```
The threshold is configurable; on a laptop with 8 GB RAM the threshold should
be much smaller (e.g., 512 MB). This connects directly to §6.7 auto-dispatch.

### 6.3 Phase 2: Dense v3 Compression

**Files**: `inst/include/sparsepress/format/header_v3.hpp`,
`inst/include/sparsepress/sparsepress_v3.hpp`

The dense v3 format has 48 reserved bytes in `FileHeader_v3.reserved[48]`. Use
the first two bytes as codec metadata:

```cpp
// reserved[0]: compression codec
enum DenseCodec : uint8_t {
    RAW_FP32   = 0,  // current format — no change, fully backwards compatible
    FP16       = 1,  // lossless fp16 truncation (50% size)
    QUANT8     = 2,  // 8-bit quantization (75% size, ~0.4% error)
    FP16_RANS  = 3,  // fp16 + XOR-delta columns + rANS entropy (best ratio)
    FP32_RANS  = 4,  // fp32 + XOR-delta columns + rANS entropy
};
// reserved[1]: delta encoding flag (0=none, 1=XOR-delta between adjacent columns)
```

Because existing v3 files were written with `reserved[0]=0` (zero-filled by
default), they read as `RAW_FP32` — automatically backwards compatible. No
new version number is needed.

#### Codec pipeline (write path)
```
float32 panel → fp16_convert (value_map.hpp) → XOR-delta transform
              → rANS encode (rans.hpp) → compressed bytes → DenseChunkDescriptor
```
The `DenseChunkDescriptor` gains an `uncompressed_size` field (currently
`byte_size == m × nc × val_bytes` exactly; now `byte_size` = compressed
bytes and `uncompressed_size` = original bytes for decompression buffer sizing).

Reuse existing codecs without modification:
- `sparsepress/codec/rans.hpp` — rANS entropy codec
- `sparsepress/transform/value_map.hpp` — fp16/quant8 conversion

No existing v3 files are in production → no backwards compat needed for the v3
chunk descriptor extension.

### 6.4 Phase 3: User-Configurable Chunk Size (Default 2048)

**Default change**: `DEFAULT_CHUNK_COLS` raised from **256 → 2048**.

This single change is expected to reduce streaming I/O overhead by ~8× at low k
(direct scaling with chunk size for the same per-chunk cost) and significantly
improve GPU utilisation.

#### User-facing API

Expose `chunk_cols` as a named parameter on the write function:
```r
st_write(x, path, chunk_cols = 2048, compression = "fp16_rans", ...)
```

- `chunk_cols = 2048`: default; good balance for HPC and laptop
- `chunk_cols = "auto"`: system-aware selection (§6.7)
- A numeric value: explicit override; power users can set 256 for debugging
  or 8192 for a beefy server

**No validation hazard**: chunk_cols affects only write-time layout. The reader
always consults `ChunkDescriptor.num_cols` from the file header, so reading
a file written with any chunk size always works correctly.

#### Auto-sizing heuristic (`choose_chunk_cols()`)

```cpp
uint32_t choose_chunk_cols(uint64_t m, uint32_t k, uint64_t ram_avail_bytes) {
    // Target: one chunk decompressed ≈ 1% of available memory budget
    // (ensures ~100 chunks can be kept in memory simultaneously during transpose)
    uint64_t bytes_per_col = m * sizeof(float);
    uint64_t target_chunk_bytes = ram_avail_bytes / 100;
    uint32_t auto_cols = (uint32_t)(target_chunk_bytes / bytes_per_col);
    return std::clamp(auto_cols, (uint32_t)256, (uint32_t)32768);
}
```

On a laptop with 8 GB RAM and m=30K genes, this gives ~2,700 columns/chunk.
On a 256 GB HPC node, ~85,000 columns/chunk — the whole matrix in one chunk,
falling back to in-core mode automatically.

### 6.5 Phase 4: Full Rename SparsePress → StreamPress

All user-visible names change from `sparsepress`/`sp_*` to `streampress`/`st_*`.
Old names are retained as `@Deprecated` aliases for one release cycle.

| Before | After | Scope |
|--------|-------|-------|
| `namespace sparsepress` | `namespace streampress` | All C++ headers |
| `#include <sparsepress/...>` | `#include <streampress/...>` | All consumers |
| `inst/include/sparsepress/` | `inst/include/streampress/` | Directory |
| `sp_write()` | `st_write()` | R API |
| `sp_read()` | `st_read()` | R API |
| `sp_info()` | `st_info()` | R API |
| `sp_convert()` | `st_convert()` | R API |
| `R/sparsepress.R` | `R/streampress.R` | R source |

The v1/v2/v3 on-disk magic bytes (`SPRZ`, `SPEN`) are **not changed** — they
are format-level constants and changing them would invalidate the entire GEO
corpus. The rename is purely at the software API level.

Deprecation wrappers in `R/sparsepress_compat.R`:
```r
#' @export
#' @rdname streampress-deprecated
sp_write <- function(...) {
  .Deprecated("st_write", package = "RcppML")
  st_write(...)
}
```

### 6.6 Phase 5: Distributed Streaming Transpose

Two complementary modes:

**During write** (`include_transpose=TRUE`):
```r
st_write(x, path, include_transpose = TRUE)
```
While streaming `x` chunk by chunk (col-major), accumulate a sorted-row
buffer for the transpose section. Working memory is O(m × chunk_cols × val_bytes)
— the same as one forward chunk. No second pass required. The transpose section
is appended to the same file; `FileHeader` offsets are patched at close.

**Post-hoc** (`st_add_transpose()`):
```r
st_add_transpose(path, tmpdir = tempdir(), verbose = TRUE)
```
For existing `.spz` files that lack a transpose section. Uses the same
streaming sort algorithm with `tmpdir` for intermediate chunk buffers.
Memory requirement: O(chunk_cols × m × val_bytes) at any time.

Both modes produce an identical on-disk result. The forward and transpose
sections are independently addressable via `FileHeader` offsets, so
`SpzLoader::next_forward()` and `SpzLoader::next_transpose()` continue to
work unchanged.

### 6.7 Phase 6: Memory-Aware Auto-Dispatch

Whenever a `.spz` file path is passed to `nmf()` or `cv_nmf()`, the dispatch
mode is determined **automatically** without user intervention. The user never
needs to set `resource=` or `streaming=` manually for `.spz` inputs.

#### Dispatch modes

```
IN_CORE_GPU    matrix fits in GPU VRAM → load all to device, run in-core GPU NMF
CPU_TO_GPU     fits in CPU RAM but not VRAM → pin to CPU RAM, stream chunks to GPU
STREAMING_GPU  matrix too large for CPU RAM → I/O→CPU→GPU pipeline (1 chunk at a time)
IN_CORE_CPU    matrix fits in CPU RAM, no GPU → load all, run in-core CPU NMF
STREAMING_CPU  matrix too large for CPU RAM, no GPU → streaming CPU NMF
```

The threshold decision:
```
compressed_file_size × decompression_ratio < available × safety_margin
```
where `safety_margin=0.70` and `decompression_ratio` is read from the
file header (fp16≈2.0, raw≈1.0, rANS≈0.5—2.0).

#### Memory detection — platform-conditional

| Platform | CPU RAM query | GPU VRAM query |
|----------|--------------|----------------|
| Linux | `/proc/meminfo` `MemAvailable` | `cudaMemGetInfo()` |
| macOS | `sysctl hw.memsize` + `vm_stat` | Metal API or skip |
| Windows | `GlobalMemoryStatusEx()` | `cudaMemGetInfo()` or skip |

This is essential for laptop support. A Windows laptop with 8 GB RAM and
no GPU must auto-select `STREAMING_CPU` for any file exceeding ~5 GB.
A 256 GB HPC node with an 80 GB H100 would auto-select `IN_CORE_GPU` for
most single-cell datasets.

#### Advanced manual override

A power-user escape hatch is exposed as a named parameter:

```r
nmf("data.spz", k = 32,
    dispatch = "STREAMING_CPU")  # manual override — dangerous!
```

When `dispatch` is set explicitly, auto-detection is **skipped entirely**. A
bold `WARNING` is emitted at the R level:

```
Warning: `dispatch` is set manually. Auto-dispatch ensures sufficient RAM
  is available before loading. Manual dispatch may cause out-of-memory errors
  or crashes. Current setting: STREAMING_CPU.
  Remove `dispatch=` to restore safe automatic mode.
```

This warning is **non-suppressable** (not going through `suppressWarnings`)
via a direct `message()` call, ensuring users who copy-paste production code
always see the caveat.

#### Integration point

Auto-dispatch is wired into `nmf_thin.R` before any C++ call is made:
```r
if (is.character(A) && grepl("\\.spz$", A)) {
  mode <- .auto_dispatch(A, k, resource = resource)
  resource <- mode$resource    # "cpu" or "gpu"
  streaming <- mode$streaming  # TRUE or FALSE
  # ... proceed ...
}
```

### 6.8 Phase 7: GEO Corpus Compatibility and k=64 Benchmark

#### Compatibility verification

The test target is the **GEO reprocessed download corpus** under
`/mnt/projects/debruinz_project/` (exact path confirmed in Phase 0). The old
`cellcensus_500k.spz` and `cellcensus_900k.spz` files have been deleted and
are **not** the test target.

Compatibility test script: `tools/verify_spz_corpus.R`
```r
# Sample 100 files from GEO corpus, check header + first chunk
for (f in sample(geo_files, 100)) {
  info <- st_info(f)  # must succeed without error
  stopifnot(info$version == 2L)
  stopifnot(info$m > 0, info$n > 0, info$nnz > 0)
}
```

This does **not** decompress full content — just header + first chunk. A full
dataaset round-trip test is reserved for a dedicated benchmark job.

#### Unit tests

`tests/testthat/test_streampress_compat.R`:
- Round-trip: write sparse matrix → `st_write()` → `st_read()` → identical
- Round-trip: write dense matrix → `st_write(codec="fp16_rans")` → `st_read()` → within fp16 tolerance
- Read v2 file (pbmc3k.spz): dimensions and nnz match expected values
- Memory test: `SpzLoader` on a 5 GB `.spz` uses < 50 MB non-file RAM after fix
- Auto-dispatch test: mock a 200 GB file, verify `IN_CORE_GPU` selected when
  mocked VRAM > file size, `STREAMING_GPU` when VRAM < file size

#### k=64 benchmark on GEO corpus

This is the **flagship publication benchmark** target for Paper P4:

```r
# Real-world benchmark: GEO single-cell dataset, k=64, NB loss
result <- nmf(
  path_to_geo_spz_file,  # confirmed from Phase 0
  k   = 64,
  loss = "nb",
  maxit = 100,
  tol   = 1e-4
)
```

The benchmark should record:
- Total wall time (and per-iteration breakdown)
- Peak RAM usage (not just available RAM)
- Chunk I/O time vs. NNLS compute time vs. other (using the
  per-section timing from `benchmarks/stream_opt/STREAMING_ANALYSIS.md` as
  baseline comparison)
- Whether auto-dispatch selected the correct mode
- Convergence curve (MSE or NB loss per iteration)

This demonstrates that a researcher on an HPC node can run
`nmf("mydata.spz", k=64)` on a multi-GB single-cell dataset with **no
configuration required** — the auto-dispatch, chunk sizing, and memory
management are all handled transparently.

### 6.9 Cross-Platform and Laptop Strategy

The StreamPress revision must work correctly on Windows laptops, not just
Linux HPC nodes. This affects three subsystems:

#### Random-access file I/O (Phase 1)

POSIX `pread()` does not exist on Windows. The `FileReader` abstraction
must compile and work on all platforms:

```cpp
#ifdef _WIN32
  // Windows: use ReadFile() with OVERLAPPED for positional reads
  HANDLE hFile_;
  size_t pread(size_t offset, void* buf, size_t size);
#else
  // POSIX (Linux, macOS, NFS): pread()
  int fd_;
  size_t pread(size_t offset, void* buf, size_t size);
#endif
```

#### Memory detection (Phase 6)

See §6.7 table above. On Windows, `GlobalMemoryStatusEx()` is used instead
of `/proc/meminfo`. On macOS, `sysctl hw.memsize`. All paths must be compiled
and tested — missing a platform branch would cause auto-dispatch to assume
"unlimited RAM" and crash on a laptop.

#### Chunk size defaults (Phase 3)

The default `chunk_cols=2048` is chosen to be safe for laptops. A 30K-gene matrix
with chunk_cols=2048 requires 30,000 × 2,048 × 4 bytes ≈ **246 MB per chunk**.
That is reasonable for an 8 GB laptop (one chunk < 3% of RAM).

For very small laptops (4 GB RAM), even this may be tight. The `auto` setting
from `choose_chunk_cols()` will automatically select a smaller chunk size based
on `MemAvailable`.

#### Windows CI

Once Phase 1 (FileReader) and Phase 4 (rename) are complete, add a Windows
`R CMD check` target to the GitHub Actions workflow (`.github/workflows/`).
This is a CRAN requirement for packages claiming Windows compatibility.

### 6.10 Implementation Order and Dependencies

```
Phase 0  → Phase 1  (must know file locations before testing loader fix)
Phase 1  → Phase 7  (loader fix is prerequisite for any streaming test)
Phase 2  ← independent (dense compression, doesn't touch sparse loader)
Phase 3  → Phase 6  (chunk sizing is part of auto-dispatch decision)
Phase 4  ← independent (rename, can run in parallel with other phases)
Phase 5  ← after Phase 1 (transpose uses same FileReader abstraction)
Phase 6  ← after Phase 3 and Phase 1 (dispatch needs accurate memory + loader)
Phase 7  ← after all other phases (final integration + benchmarks)
```

**Recommended order**: Phase 0 → Phase 1 → Phase 4 (rename interleaved) →
Phase 2 → Phase 3 → Phase 5 → Phase 6 → Phase 7

### 6.11 Relevant Files Summary

| File | Role | Change |
|------|------|--------|
| `inst/include/FactorNet/io/spz_loader.hpp` | SpzLoader — sparse v2 | Fix loader bug (Phase 1) |
| `inst/include/FactorNet/io/dense_spz_loader.hpp` | DenseSpzLoader — dense v3 | Fix loader bug (Phase 1) |
| `inst/include/FactorNet/io/loader.hpp` | DataLoader interface | No change |
| `inst/include/sparsepress/format/header_v2.hpp` | v2 format spec | **DO NOT MODIFY** |
| `inst/include/sparsepress/format/header_v3.hpp` | v3 format spec | Add codec in reserved[0:1] (Phase 2) |
| `inst/include/sparsepress/sparsepress_v3.hpp` | v3 write/read | Add compression pipeline (Phase 2) |
| `inst/include/sparsepress/codec/rans.hpp` | rANS codec | Reuse as-is — no changes |
| `inst/include/sparsepress/transform/value_map.hpp` | fp16/quant8 convert | Reuse as-is — no changes |
| `inst/include/FactorNet/nmf/fit_streaming_spz.hpp` | Streaming NMF entry | Update to streampress ns (Phase 4) |
| `inst/include/FactorNet/nmf/fit_chunked.hpp` | Chunked CPU NMF | Update chunk_cols default (Phase 3) |
| `inst/include/FactorNet/nmf/fit_chunked_gpu.cuh` | Chunked GPU NMF | Update chunk_cols default (Phase 3) |
| `R/sparsepress.R` → `R/streampress.R` | R API | Full rename + deprecated wrappers (Phase 4) |
| `R/nmf_thin.R` | Dispatch logic | Wire auto-dispatch for .spz input (Phase 6) |
| `tools/verify_spz_corpus.R` | Corpus compat check | New (Phase 7) |
| `tests/testthat/test_streampress_compat.R` | Unit tests | New (Phase 7) |
