# RcppML v1.0.1 — Production Workstreams

**Created**: March 7, 2026 — Post-cleanup audit
**Status**: Active roadmap for shipping RcppML v1.0.1

This document replaces the 11-phase sequential HARDENING_PLAN with 5 focused,
parallelizable workstreams derived from a comprehensive codebase audit.

---

## Current State (Post-Audit Summary)

### What's Solid

| Area | Status |
|------|--------|
| **R source layer** (32 files) | Production-quality. All functions documented. |
| **C++ FactorNet headers** (62+ files) | Clean architecture, well-organized. |
| **SparsePress I/O** | Fully functional, tested. |
| **Algorithm docs** (9 documents) | Complete in `docs/factornet/algorithms/`. |
| **Coverage matrix** | Rev 4 with ~320 entries in `docs/dev/`. |
| **testthat suite** | 76 test files. 842 tests: 387 pass, 452 skipped (GPU/CRAN), 2 fail, 4 warnings. |
| **S4 methods** | Comprehensive (show, summary, dim, subset, predict, plot, etc.). |
| **IRLS framework** | All 6 distributions implemented and working on CPU. |
| **Factor network DSL** | Composable graph-based factorization functional. |
| **Benchmark harness** | Structured framework in `benchmarks/harness/` with baselines. |
| **Vignettes** | 12 Rmd articles covering major features. |
| **JSS manuscript** | In `manuscript/jss/`, under active revision. |

### What Needs Work

| Gap | Impact | Blocking? |
|-----|--------|-----------|
| ZI-twoway broken | Users hit silent bad results | **Yes** — must guard |
| ~123 "planned" entries (future scope) | Not yet coded; would need guards if user-reachable | Audit needed |
| 5 "unsupported" entries already guarded | ZI×MSE/Gamma/InvGauss/Tweedie, Cholesky+IRLS — all have R+C++ guards | **No** — already done |
| GPU SVD: test files exist but skipped (no GPU in CI) | GPU parity untested in practice | Moderate |
| GPU CV: test_gpu_cv.R exists with 10 tests (skipped) | Tests written but need GPU to validate | Moderate |
| GPU IRLS: test_gpu_distributions.R exists (skipped) | Tests written but need GPU to validate | Moderate |
| 2 SVD ground-truth failures | krylov+lanczos fail on dense-as-sparse matrix (test_svd.R:628,637) | Moderate |
| 1 streaming SPZ test failure | test_file_input.R:48 — test writes SPZ without transpose | Low (test bug) |
| R CMD check status unknown | CRAN submission blocked | **Yes** for CRAN |
| 2 vignettes excluded (build issues) | Incomplete user docs | Low |
| GPU/IO doc stubs in factornet/ | Incomplete dev docs | Low |
| NEWS.md not tracked in git | Missing from repo | Low |
| 19 R source files untracked | New files need `git add` | **Yes** for repo |
| 12 src/ files untracked | All C++/CUDA bridge files need `git add` | **Yes** for repo |
| 68 test files untracked | New testthat files need `git add` | **Yes** for repo |
| 111 man/ pages untracked | Generated .Rd files need `git add` | **Yes** for repo |
| All data/*.rda untracked | Package datasets need `git add` | **Yes** for repo |
| devtools::document() fails | Rcpp compile error during document regeneration | **Yes** — must fix |

---

## Workstream A: Repository & Build Hygiene

**Goal**: Clean git state, all source files tracked, `R CMD build` produces a
minimal correct tarball.

**Priority**: Do first — unblocks everything else.

### A.1 Track all source files

The following R source files exist on disk but are NOT in git (created during
recent development iterations). They need to be committed:

```
R/auto_distribution.R    R/classifier_metrics.R   R/consensus.R
R/cross_validate_graph.R R/data.R                 R/deprecated.R
R/factor_methods.R       R/factor_net.R           R/gpu_backend.R
R/guides.R               R/nmf_plots.R            R/nmf_thin.R
R/nmf_validation.R       R/plot_nmf.R             R/predict_nmf.R
R/random.R               R/simulateNMF.R          R/simulateSwimmer.R
R/solve.R                R/sp_gpu.R               R/sparsepress.R
R/svd.R                  R/svd_methods.R          R/training_log.R
R/utils_globals.R
```

Also need to be tracked:
- `NEWS.md`
- `CONTRIBUTING.md`
- `HARDENING_PLAN.md` (or remove if no longer needed)
- `configure`, `cleanup` (build scripts)
- `_pkgdown.yml`
- All `src/*.cpp`, `src/*.cu`, `src/*.cuh`, `src/Makefile.gpu`, `src/Makevars*`
- All `inst/include/` headers (if not already)
- All `man/*.Rd` files
- All `data/*.rda` files and `data/datalist`
- All `vignettes/*.Rmd` files
- `tests/testthat/*.R` (many new test files)
- `tests/cpp/` (C++ test infrastructure)
- `docs/dev/COVERAGE_MATRIX.yaml` and `.md`
- `docs/factornet/` (developer documentation)

**Task**: Run `git add` for all the above. Verify with `git status` that
nothing essential is untracked.

### A.2 Verify .Rbuildignore

Ensure these are excluded from the R package tarball:
- `benchmarks/`, `manuscript/`, `tools/`, `experiments/` (if any remain)
- `.github/`, `.vscode/`
- `HARDENING_PLAN.md`, `WORKSTREAMS.md`, `CONTRIBUTING.md`
- `docs/dev/`, `docs/factornet/`, `docs/site/`
- `tests/cpp/` (standalone C++ tests, not testthat)
- All `.sbatch` files
- All `src/*.cu`, `src/*.cuh`, `src/Makefile.gpu` (GPU code compiled separately)

### A.3 Verify R CMD build

```bash
R CMD build .
# Check tarball contents — no GPU code, no benchmarks, no manuscripts
tar tzf RcppML_1.0.1.tar.gz | head -50
```

### A.4 ~~Fix devtools::document() Build Failure~~ — RESOLVED

**AUDIT FINDING**: The failure was caused by stale compiled objects. After
a clean rebuild, `devtools::document()` and `devtools::test()` both work.
`R CMD INSTALL .` always worked.

**No action needed** — just ensure clean objects before running devtools.

### A.5 Checklist

- [ ] All R source files tracked in git
- [ ] All new testthat files tracked
- [ ] All C++ headers tracked
- [ ] NEWS.md tracked
- [ ] .Rbuildignore excludes non-package content
- [ ] `R CMD build` produces clean tarball < 10 MB
- [ ] No compiled objects (`.o`, `.so`) in the tarball

---

## Workstream B: Guard & Disable Unfinished Features

**Goal**: Every parameter combination that a user can reach either (a) works
correctly, or (b) throws a clear, helpful error message. No silent failures.

**Priority**: Critical for correctness. Do before testing.

### B.1 ZI-Twoway — Disable with Guard

The two-way zero-inflation model (`zi_mode = "twoway"`) produces runaway π
estimates on high-sparsity data. It is marked BROKEN in the coverage matrix
for both GP and NB on all backends.

**Action**: Add error guard at R validation layer:

```r
# In nmf_validation.R or nmf_thin.R:
if (!is.null(zi_mode) && zi_mode == "twoway") {
  stop("zi_mode='twoway' is currently disabled due to numerical instability ",
       "on high-sparsity data. Use zi_mode='row' or zi_mode='col' instead. ",
       "See GitHub issue #XX for status.", call. = FALSE)
}
```

Also add C++ gateway guard (defense in depth):
```cpp
// In nmf/fit.hpp or gateway:
if (config.zi_mode == ZIMode::TWOWAY) {
  throw std::invalid_argument(
    "zi_mode=TWOWAY is disabled (numerical instability). Use ROW or COL.");
}
```

### B.2 ~~Unsupported ZI × Distribution Guards~~ — ALREADY DONE

**AUDIT FINDING**: All ZI × invalid distribution guards already exist in `nmf_thin.R` lines 556–568:
- MSE + ZI → `stop("not supported with distribution='mse'")`
- Gamma + ZI → `stop("not supported with distribution='gamma'")`
- InvGauss + ZI → `stop("not supported with distribution='inverse_gaussian'")`
- Tweedie + ZI → `stop("not supported with distribution='tweedie'")`
- Catch-all: `zi != 'none' requires loss='gp' or loss='nb'`

C++ also guards in `config.validate()`: throws if `zi_mode != ZI_NONE && loss != GP && loss != NB`.

**Tests**: Already covered in `test_unsupported_combos.R` (11 guard tests, all pass).

**No action needed.**

### B.3 ~~Cholesky + Non-MSE Guard~~ — ALREADY DONE

**AUDIT FINDING**: Guard exists in `nmf_thin.R` lines 589–594:
```r
stop("solver='cholesky' is not supported with non-MSE distributions (got '", loss, "'). ...")
stop("solver='cholesky' is not supported with robust IRLS (robust_delta > 0). ...")
```

C++ also guards in `config.validate()`: throws if `solver_mode == 1 && requires_irls()`.

**Tests**: Covered by `GUARD-CHOLESKY-IRLS` and `GUARD-CHOLESKY-ROBUST` in `test_unsupported_combos.R`.

**No action needed.**

### B.4 Audit "Planned" Entries for User-Reachable Paths

Go through each `planned` entry in `docs/dev/COVERAGE_MATRIX.yaml`. For each one:

1. **Can a user reach this code path?** Trace from R API → validation → C++ dispatch.
2. **If yes**: Add a guard that intercepts before reaching unimplemented code.
3. **If no**: The entry is fine — it's unreachable by design.

Key paths to check:
- Dense streaming (SPZ v3 not implemented) — does `nmf("file.spz", ...)` with
  a dense SPZ file silently fail?
- Semi-NMF (`nonneg_W = FALSE`) — is this a user-facing parameter?
- GPU dense + non-MSE — does GPU dispatch with dense + GP silently fail?
- Standalone GPU NNLS — is `nnls()` exposed for GPU?

### B.5 Checklist

- [ ] ZI-twoway disabled with clear error (R + C++)
- [ ] ZI × invalid distribution guarded
- [ ] Cholesky + non-MSE guarded
- [ ] Every user-reachable "planned" path either works or throws an error
- [ ] Every "unsupported" entry has R + C++ guard
- [ ] Test: attempt every guarded combination → get clear error message

---

## Workstream C: Test Hardening

**Goal**: Bring test coverage from ~65 "tested" entries to cover the most
important untested paths, especially GPU and guards.

**Priority**: After Workstream B (tests verify guards work).

### C.1 Guard Rejection Tests

For every guard added in Workstream B, write a testthat test:

```r
test_that("zi_twoway_throws_error", {
  A <- simulateNMF(50, 30, k = 3)$A
  expect_error(nmf(A, k = 3, loss = "gp", zi_mode = "twoway"),
               "disabled")
})

test_that("zi_with_mse_throws_error", {
  A <- simulateNMF(50, 30, k = 3)$A
  expect_error(nmf(A, k = 3, loss = "mse", zi_mode = "row"),
               "only supported for")
})

test_that("cholesky_with_gp_throws_error", {
  A <- simulateNMF(50, 30, k = 3)$A
  expect_error(nmf(A, k = 3, loss = "gp", solver = "cholesky"),
               "only supports")
})
```

**Target**: One test per guard from B.1–B.4. Collect in
`tests/testthat/test_unsupported_combos.R` (may already exist — extend it).

### C.2 ~~GPU Parity Tests~~ — Tests Already Written (Need GPU Validation)

**AUDIT FINDING**: GPU tests already exist and are comprehensive:
- `test_gpu_accuracy.R` — 7 tests: CPU/GPU parity, fp32/fp64, multi-GPU, tight parity
- `test_gpu_cv.R` — 10 tests: MSE/GP/NB CV, multi-rank, speckled/full-mask
- `test_gpu_svd.R` — GPU SVD tests
- `test_gpu_distributions.R` — GPU IRLS for distributions
- `test_gpu_dense.R`, `test_gpu_streaming.R`, `test_gpu_features.R`, etc.

All 452 skipped tests are skipped with `skip("On CRAN")` or `skip_if(!gpu_available())`.

**Action**: Run the full test suite once on a GPU node to validate:
```bash
ssh g051 'cd /mnt/home/debruinz/RcppML-2 && module load r/4.5.2 cuda/12.8.1 && \
  Rscript -e "Sys.setenv(NOT_CRAN=\"true\"); testthat::test_dir(\"tests/testthat\", reporter=\"summary\", package=\"RcppML\", stop_on_failure=FALSE)"'
```

### C.3 Fix SVD Ground-Truth Test Failures

Two SVD ground-truth tests fail in `test_svd.R`:

**Failures** (lines 628–638):
- `TEST-SVD-KRYLOV-GROUNDTRUTH`: krylov gives `d = [50, 30, 15, 4.28, 0.39]`
  instead of `[50, 30, 15, 8, 4]`. Reconstruction error 12.9% (expected < 0.1%).
- `TEST-SVD-LANCZOS-GROUNDTRUTH`: Same values, same error.

**Root cause analysis**: The test creates a rank-5 dense matrix, converts to
`dgCMatrix` (which is dense — no actual zeros), then asks krylov/lanczos to
recover all 5 singular values. These iterative methods may struggle when:
1. The matrix has no spectral gap beyond rank 3 in the sparse representation
2. The restart dimension is too small for k=5
3. The convergence tolerance or max iterations aren't sufficient

**Investigation steps**:
1. Check if deflation and irlba pass the same test (they do — only krylov/lanczos fail)
2. Try increasing maxit or decreasing tol
3. Try with a truly sparse matrix (should work better)
4. If it's a fundamental limitation of these methods on this edge case,
   adjust test tolerances or skip for this matrix type

### C.4 Fix Streaming SPZ Test

`test_file_input.R:48` fails because `sp_write()` writes without
`include_transpose = TRUE`, but streaming NMF requires a pre-stored transpose.

**Fix**: Update the test to pass `include_transpose = TRUE`:
```r
sp_write(movielens, tmp, include_transpose = TRUE)
```

### C.5 ~~Streaming Parity Tests~~ — Already Written (Skipped on CRAN)

**AUDIT FINDING**: Comprehensive streaming tests already exist:
- `test_streaming.R` — 10 tests: MSE, GP, NB, regularization, scaling, loss tracking, init, CV, auto-chunking
- `test_streaming_loss_rejection.R` — Tests for unsupported streaming combos
- `test_streaming_svd_cv.R` — Streaming SVD with CV

All are skipped with `skip("On CRAN")`. Need to run with `NOT_CRAN=true`.

**No new tests needed.** Just validate on compute node.

### C.6 Checklist

- [ ] Guard rejection tests for ZI-twoway (Workstream B.1)
- [ ] Run GPU tests on g051/g052 — validate all 452 skipped tests
- [ ] Fix SVD krylov/lanczos ground-truth failures (C.3)
- [ ] Fix streaming SPZ test (C.4 — `include_transpose = TRUE`)
- [ ] Run streaming tests with `NOT_CRAN=true` on compute node
- [ ] Full `devtools::test()` passes (387 pass + 0 fail after fixes)

---

## Workstream D: CRAN Submission Prep

**Goal**: Clean `R CMD check --as-cran` with 0 ERRORs, 0 WARNINGs, ≤2 NOTEs.

**Priority**: Final gate before release. Depends on B and C.

### D.1 R CMD check --as-cran

Run and fix ALL issues:

```bash
ssh <node> 'cd /mnt/home/debruinz/RcppML-2 && module load r/4.5.2 && \
  R CMD build . && R CMD check --as-cran RcppML_1.0.1.tar.gz 2>&1'
```

**Common issues to expect and fix**:

| Issue | Likely Fix |
|-------|-----------|
| `checking installed package size ... NOTE` | Compress data, trim vignettes |
| `checking examples ... ERROR` (> 5s) | Wrap slow examples in `\donttest{}` |
| `checking for missing documentation entries` | Run `devtools::document()` |
| `Undocumented S4 methods` | Add `@rdname` or `@aliases` tags |
| `checking Rd cross-references` | Fix broken `\link{}` references |
| `no visible binding for global variable` | Add to `utils_globals.R` |
| `Non-standard files in src/` | Verify `.Rbuildignore` excludes GPU code |
| `checking dependencies in R code` | Ensure all used packages in Imports/Suggests |

### D.2 Vignettes

Two vignettes are currently excluded in `.Rbuildignore`:
- `sparsepress.Rmd`
- `nmf-deep-dive.Rmd`

For each:
1. Try building: `rmarkdown::render("vignettes/sparsepress.Rmd")`
2. If it fails: fix the build error (missing data, broken code, etc.)
3. If examples are too slow: use `eval=FALSE` or precomputed results
4. If it can't be fixed quickly: leave excluded, document as known gap

### D.3 Example Runtime

Scan all `man/*.Rd` files for `\examples{}` blocks. Any example that takes > 5s
needs `\donttest{}` wrapping. Common offenders:
- NMF with large k on bundled datasets
- CV NMF (multiple fits)
- Streaming examples (file I/O)
- GPU examples (device init overhead)

### D.4 Package Size

CRAN limit is 5 MB for tarball. Check:
```r
file.size("RcppML_1.0.1.tar.gz") / 1e6  # MB
```

If too large, candidates for trimming:
- `data/digits_full.rda` (784-dim full MNIST digits — large)
- `data/olivetti.rda` (4096-dim face images — large)
- `inst/extdata/pbmc3k.spz` (compressed but still adds size)

### D.5 Checklist

- [ ] `R CMD check --as-cran` passes with 0 ERRORs, 0 WARNINGs
- [ ] All included vignettes render without error
- [ ] No examples take > 5 seconds
- [ ] Package tarball < 5 MB
- [ ] `devtools::document()` + `tools/fix_rcpp_info_bug.sh` run cleanly
- [ ] DESCRIPTION version, date, and author correct
- [ ] NEWS.md documents all changes since last CRAN release

---

## Workstream E: Documentation Polish

**Goal**: All user-facing and developer-facing documentation is complete and
consistent.

**Priority**: Parallel with C and D. Feed into CRAN submission.

### E.1 README.md

Update the package README to reflect the current feature set:
- Installation (CRAN + GitHub dev)
- Quick start (5-line NMF example)
- Feature highlights table (distributions, GPU, streaming, CV, etc.)
- Links to vignettes
- Citation info

### E.2 Roxygen Completeness

Scan for any exported function missing complete documentation:
```r
# Check for missing docs
devtools::check_man()
```

Pay special attention to recently added functions:
- `factor_net()` and related DSL functions
- `auto_nmf_distribution()`, `diagnose_dispersion()`, `diagnose_zero_inflation()`
- GPU functions: `gpu_available()`, `gpu_info()`, `sp_read_gpu()`, `sp_free_gpu()`
- Classifier metrics: `classify_embedding()`, `classify_logistic()`, `classify_rf()`
- Training logger: `training_logger()`, `export_log()`

### E.3 Developer Documentation Stubs

Fill the stub READMEs in:
- `docs/factornet/gpu/README.md` — GPU architecture, kernel inventory, VRAM management
- `docs/factornet/io/README.md` — SPZ format details, streaming API, chunk boundaries

### E.4 pkgdown Site

Verify the pkgdown site builds cleanly:
```r
pkgdown::build_site()
```

Check:
- All articles render
- All reference pages present
- Navigation works
- `_pkgdown.yml` sections are complete

### E.5 Checklist

- [ ] README.md updated with current features
- [ ] All exported functions have complete roxygen docs
- [ ] GPU/IO developer doc stubs filled
- [ ] pkgdown site builds without errors
- [ ] NEWS.md finalized for v1.0.1

---

## Sequencing & Dependencies

```
A (Repo Hygiene)
  │
  ├──→ B (Guards)
  │      │
  │      └──→ C (Tests)
  │             │
  │             └──→ D (CRAN Prep)
  │                    ↑
  └──→ E (Docs) ───────┘
```

- **A** must come first: clean git state, all files tracked
- **B** must come before C: tests verify guards work
- **C** and **E** are parallel
- **D** depends on B, C, and E all being done

### Estimated Effort

| Workstream | Scope | Agent Sessions |
|-----------|-------|---------------|
| A — Repo Hygiene | git add, .Rbuildignore, R CMD build | 1 |
| B — Guards | ~10-15 guards to add/verify | 1–2 |
| C — Tests | ~15-20 new tests + 1 bug fix | 2–3 |
| D — CRAN Prep | R CMD check iteration | 1–2 |
| E — Docs | README, stubs, pkgdown | 1–2 |

---

## Appendix: Coverage Matrix Status (Actual from Audit)

### Current Status (367 entries total)

| Status | Count | Description |
|--------|------:|-------------|
| tested | 83 | Has R-level tests with expected behavior verification |
| implemented | 144 | Code exists, no tests yet |
| planned | 123 | Not yet coded; in scope for future development |
| validated | 2 | Tested with ground-truth / mathematical verification |
| broken | 4 | ZI twoway (GP & NB × CPU + GPU) |
| unsupported | 5 | Guarded: ZI×MSE/Gamma/InvGauss/Tweedie, Cholesky+IRLS |
| deprecated | 2 | MAE and Huber legacy loss functions |
| fallback | 4 | GPU projective/symmetric CV → CPU fallback with logging |

### After Completing Workstreams

| Status | Before | Target | Change |
|--------|-------:|-------:|--------|
| tested | 83 | ~90 | +7 (ZI guard test, SPZ fix, SVD fixes) |
| implemented | 144 | 144 | — (testing backlog remains for future) |
| planned | 123 | 123 | — (future scope) |
| broken | 4 | 0 | −4 (ZI-twoway disabled) |
| unsupported+guarded | 5 | 9 | +4 (ZI-twoway moved here) |""

The key outcome: **zero "broken" entries**, and every user-reachable path
either works or produces a clear error. The 144 \"implemented\" entries without
tests represent a future testing backlog but are code-complete and functional.
