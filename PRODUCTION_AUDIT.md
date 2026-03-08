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

#### 2.2.1 NB Dispersion Not Updated in Streaming Path

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

#### 2.2.2 `mask_zeros=FALSE` CV Is O(m·n) in Streaming Path

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

#### 2.2.3 SVD Init Decompresses Full Matrix During Streaming

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

#### 2.2.4 `std::async` Per Chunk Creates Thread Per I/O Operation

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

#### 2.2.5 `goto` Statement in Forward Pass

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

#### 2.2.6 Loss History Reports MSE for IRLS Losses

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

1. **Adaptive CD↔Cholesky crossover** — `solver="auto"` not yet implemented
   based on empirical crossover; instead falls back to fixed rules
2. **Streaming double-buffer** — current `std::async` approach works but is
   suboptimal (see §2.2.4 above)
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
| P4 — SparsePress streaming (SoftwareX) | Not started | Needs streaming critique fixes |
| P5 — FactorNet graph (JMLR/NeurIPS) | Not started | Needs FactorNet graph docs |
| P6 — Constrained SVD (Comp Stats) | Not started | Needs SVD algorithm docs |

---

*This document replaces `HARDENING_PLAN.md` and `WORKSTREAMS.md`, both of which
have been archived to `docs/dev/`. The coverage matrix remains live at
`docs/dev/COVERAGE_MATRIX.yaml`.*
