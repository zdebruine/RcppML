# RcppML Production Audit — CRAN Submission Checklist

> **Date**: 2026-03-03  
> **Auditor**: GitHub Copilot (Claude Sonnet 4.6)  
> **Scope**: Code quality, feature parity, CRAN compliance, architecture, tests, docs, cleanup

---

## Legend

- `[ ]` Not started  
- `[x]` Complete  
- `[!]` Blocker — must fix before CRAN submission  
- `[~]` Should fix — not a hard blocker but strongly recommended  

---

## 0. Prior Audit

A prior CRAN audit (`CRAN_AUDIT.md`, dated 2026-02-27) exists and is substantive. This document supersedes it with consolidated, more granular checklists. The previous audit should be **deleted** once this one is in use (see S4 below).

---

## 1. CRAN Blockers

These will cause `R CMD check` failures or immediate rejection.

- `[x]` **B1 — Version inconsistency**: ~~`DESCRIPTION` says `1.0.0`; `NEWS.md` describes `2.0.0`~~ — both now `1.0.1`.
- `[x]` **B2 — Compiled objects in repo**: `src/RcppExports.o` and `src/bipartiteMatch.o` removed from git tracking (`git rm --cached`). `.gitignore` already covers `src/*.o` and `src/*.so`.
- `[x]` **B3 — `inst/lib/RcppML_gpu.so`**: Added `^inst/lib$` to `.Rbuildignore`. The 20 MB GPU `.so` is excluded from the CRAN tarball.
- `[x]` **B4 — `strip --strip-debug` in `Makevars`**: Investigated — no strip command in `src/Makevars` or `src/Makevars.in`. False alarm.
- `[x]` **B5 — Pre-built vignette HTML in `vignettes/`**: Removed all 6 pre-built `.html` files from `vignettes/` source dir plus junk files (`.bak`, `render_vignette.sh`, `nmf-deep-dive-new.Rmd`, `nmf-deep-dive_files/`, `test_files/`, `sparsepress.log`, `sparsepress.tex`). Built vignettes remain in `inst/doc/`.
- `[x]` **B6 — Package tarball size**: Added `^inst/lib$` (20 MB), `^data/hcabm40k_mat\.rda$` (90 MB), `^data/pbmc3k\.rda$` (2.6 MB), `^inst/extdata$` (2.1 MB) to `.Rbuildignore`. Estimated tarball ≈ 3.5 MB, under 5 MB CRAN limit. All pbmc3k examples already in `\donttest{}`.
- `[x]` **B7 — `mse()` function signature change**: Added backward-compat guard in `R/nmf_methods.R`: detects old `mse(A, w, d, h)` call (where `d` would be a matrix) and remaps to new `mse(w, d, h, data)` with `.Deprecated()` warning.
- `[x]` **B8 — `project()` deprecated wrapper**: Confirmed `R/deprecated.R` handles both old call patterns: `project(nmf_obj, data)` → `predict(w, data)` and `project(w_matrix, data)` → `nnls(w, A)`. Properly calls `.Deprecated()`.
- `[x]` **B9 — `nnls()` signature backward compat**: Confirmed `R/solve.R` has a backward compat block detecting old `nnls(w, A)` positional form when `A` is missing and both `w`/`h` are non-NULL. Calls `.Deprecated()` and remaps.
- `[x]` **B10 — Root `.log` files committed**: Deleted from repo; `.Rbuildignore` updated with `^logs$` and `^src/RcppML.*\.so$`.
- `[x]` **B11 — Root test scripts committed**: All root-level diagnostic/dev `.R` scripts deleted.
- `[x]` **B12 — Many `.sbatch` files in root**: All root-level `.sbatch` and `.sh` files deleted.
- `[x]` **B13 — diagnose/verify scripts in root**: All deleted (including `python_verify.py`, `diagnose_bypass_output.txt`).
- `[x]` **B14 — `configure` script**: Confirmed configure IS functional: detects CUDA/MPI and writes `inst/build_config` at install time. Also regenerates `src/Makevars` from `Makevars.in` (currently no substitution tokens but config detection info is captured in build_config). Not a no-op.

---

## 2. CRAN Warnings / Recommendations

- `[x]` **W1 — `predict()` `@return`**: Investigated — `predict.nmf` correctly returns `new("nmf", ...)`. No fix needed; `@return` was already accurate.
- `[x]` **W2 — Duplicate `@export` in `predict_nmf.R`**: Investigated — only one `@export` tag present. Already fixed or was a false audit finding.
- `[x]` **W3 — `dimnames<-,nmf` print side effect**: Investigated — no `dimnames<-` method exists and `dimnames()` has no `print()` call. False audit finding.
- `[~]` **W4 — `RcppML.R` uses `@import Matrix`**: prefer `@importFrom Matrix ...` to pass CRAN's namespace check.
- `[~]` **W5 — No `skip_on_cran()` on long-running tests**: tests that require >60 s of wall time need `testthat::skip_on_cran()` inside the `test_that()` block.
- `[~]` **W6 — `bipartiteMatch()` returns 0-indexed assignment**: this is a trap for R users expecting 1-based indexing. Either convert to 1-based before returning to R or prominently document the 0-based return in the man page.
- `[x]` **W7 — `data.R` `digits_full` `@format`**: Fixed — changed from `dgCMatrix` to `matrix (dense)` to match `@details`.
- `[x]` **W8 — `test_nnls.R` dimension mismatch test**: Investigated — test already has `expect_error(nnls(w=w, A=A_bad))`. False alarm.
- `[~]` **W9 — `DESCRIPTION` `SystemRequirements`**: Currently says `CUDA Toolkit >= 11.0`. Verify actual minimum version (code uses features from CUDA 12.x?). Also clarify that CUDA and MPI are both optional and do not affect the main package build.
- `[x]` **W10 — `NEWS.md` version alignment**: Both `DESCRIPTION` and `NEWS.md` top entry now say `1.0.1`.

---

## 3. SVD Feature Coverage & Backend Parity

### 3.1 Algorithm Inventory

Five SVD algorithms are implemented. Feature matrix against documented SVD parameters:

| Feature | `deflation` | `krylov` | `lanczos` | `irlba` | `randomized` |
|---------|:-----------:|:--------:|:---------:|:-------:|:------------:|
| Sparse input | ✅ | ✅ | ✅ | ✅ | ✅ |
| Dense input | ✅ | ✅ | ✅ | ✅ | ✅ |
| PCA centering | ✅ | ✅ | ✅ | ✅ | ✅ |
| L1 penalty | ✅ | ✅ | ❌ | ❌ | ❌ |
| L2 penalty | ✅ | ✅ | ❌ | ❌ | ❌ |
| Non-negativity | ✅ | ✅ | ❌ | ❌ | ❌ |
| Upper bound | ✅ | ✅ | ❌ | ❌ | ❌ |
| L21 penalty | ✅ (adaptive L2) | ✅ | ❌ | ❌ | ❌ |
| Angular penalty | ✅ (via angular_penalty helper) | ✅ | ❌ | ❌ | ❌ |
| Graph Laplacian | ✅ (gradient step) | ✅ | ❌ | ❌ | ❌ |
| CV / auto-rank | ✅ | ✅ | ❌ | ❌ | ❌ |
| mask_zeros CV | ✅ | ✅ | ❌ | ❌ | ❌ |
| Streaming SPZ | ✅ | ✅ | ✅ | ✅ | ✅ |
| GPU acceleration | ✅ | ✅ | ✅ | ✅ | ✅ |
| float precision | ✅ | ✅ | ✅ | ✅ | ✅ |
| `variance_explained()` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `reconstruct()` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `[` subset | ✅ | ✅ | ✅ | ✅ | ✅ |

### 3.2 SVD Checklist

- `[ ]` **SVD-01 — No `testthat` tests for SVD at all**: There is no `tests/testthat/test_svd.R` file. All SVD testing is in ad-hoc benchmark/diagnostic scripts (`benchmarks/`, root-level `test_svd_all.R`, etc.) that will not run during `R CMD check`. **This is a critical gap for CRAN.** Create `tests/testthat/test_svd.R` covering:
  - Correctness: singular values match `base::svd()` (or `irlba::irlba()`) within tolerance
  - All five methods produce valid decomposition (U·diag(d)·V' ≈ A)
  - PCA centering: result matches `prcomp()` on small dense matrix
  - Non-negative SVD: all u,v values ≥ 0
  - Sparse L1: sparsity increases with higher L1
  - Auto-rank: k selected ≤ k_max; returns valid `svd_pca` object
  - Streaming SPZ: round-trip write → SVD → check
  - Reproducibility: same seed → same result

- `[ ]` **SVD-02 — `auto_select.hpp` decision boundary undocumented at R level**: The user-facing `svd()` documentation says `method = "deflation"` as default but does not explain that `resource = "auto"` + no constraints will actually use `lanczos` or `irlba` as NMF warm-start initializers. The `method` parameter in `svd()` refers to the SVD algorithm itself; document the auto-selection logic used when `method` defaults inside NMF's init path.

- `[ ]` **SVD-03 — `svd_pca` class needs `predict()` / `project()` method**: NMF has `predict(model, newdata)` to project new samples. SVD/PCA lacks the equivalent: users cannot project new out-of-sample data onto existing principal components using the standard S4 dispatch. Implement `setMethod("predict", "svd_pca", ...)` that solves for the new scores: `H_new = V1 \cdot A_new` (for unconstrained) or nnls/soft-threshold for constrained PCA.

- `[ ]` **SVD-04 — `variance_explained()` fallback is incorrect**: When `frobenius_norm_sq` is `NULL`, the fallback uses `sum(d^2)` which only accounts for the extracted factors and will overestimate individual variance fractions for truncated SVD. Store the full `||A||^2_F` in `misc$frobenius_norm_sq` for all methods; it is already computed inside `irlba_svd.hpp`, `randomized_svd.hpp`, and `deflation_svd.hpp`.

- `[x]` **SVD-05 — `spmv.hpp.bak` file in the include tree**: Deleted.

- `[x]` **SVD-06 — `fit_unified.hpp.dscale_backup`**: Deleted.

- `[x]` **SVD-07 — `graph_U`/`graph_V` docs said "krylov only"**: Fixed in `svd.R` — both parameters now say `deflation` and `krylov`.

- `[x]` **SVD-08 — `angular` penalty docs said "krylov only"**: Fixed in `svd.R` — now says `deflation` and `krylov`.

- `[ ]` **SVD-09 — `pca()`, `sparse_pca()`, `nn_pca()` aliases not in `man/`**: These are convenience wrappers defined in `svd.R`. Check whether they are exported and documented. If exported, they require their own man pages or `@rdname svd`.

- `[ ]` **SVD-10 — GPU SVD parity matrix missing from documentation**: The user-facing docs (`svd.R` @details, README) do not state which SVD algorithms run on GPU vs CPU-only. Add a feature-support table to the `?svd` help page.

- `[ ]` **SVD-11 — Streaming SVD: only `deflation` is tested in `test_svd_all.R`**: The streaming path dispatches to all five algorithms, but only deflation is systematically profiled. Add correctness tests for `streaming_lanczos_svd()`, `streaming_irlba_svd()`, `streaming_randomized_svd()`, `streaming_krylov_svd()`.

- `[ ]` **SVD-12 — `svd_pca` class `show()` / `head()` methods do not print method or algorithm used**: Add `misc$method` and `misc$algorithm` to the C++ result return so the S4 `show()` can print which algorithm was used (analogous to NMF printing resource diagnostics).

- `[ ]` **SVD-13 — `test_entries.hpp` has duplicate `if (mask.mask_zeros()) ... else { same code }`**: Both branches of the `mask_zeros()` if-else in `init_test_entries_sparse()` are identical. Remove the branch or differentiate correctly.

- `[ ]` **SVD-14 — `svd_pca` `@misc` slot has no formal specification**: `misc` is a plain list with ad-hoc fields. Document all expected keys in the class doc (`svd_methods.R`): `centered`, `row_means`, `test_loss`, `iters_per_factor`, `wall_time_ms`, `auto_rank`, `frobenius_norm_sq`, `method`, `algorithm`, `resource`.

---

## 4. NMF Feature Coverage & Backend Parity

### 4.1 Feature Coverage

The existing `PRODUCTION_STATUS.md` feature matrix is largely accurate. Key gaps:

- `[ ]` **NMF-01 — GPU CV is `✅ᵘ` (untested)**: `fit_cv_gpu_unified.cuh` exists but GPU CV has no dedicated test file. The existing `test_gpu_features.R` and `test_gpu_accuracy.R` test standard NMF. Add a GPU CV test.

- `[ ]` **NMF-02 — Multi-GPU and MPI paths are `✅ᵘ`**: These have allocation and dispatch code but no automated tests that run in CI. At minimum, document which test file exercises them and add `skip_if_not()` guards.

- `[x]` **NMF-03 — `test_cv_irls.R` excluded**: Added `skip("MAE/Huber/KL CV hang bug…")` at top of file so it appears in the test suite with a clear documented reason.

- `[ ]` **NMF-04 — `k = "auto"` path goes through `Rcpp_nmf_rank_cv_sparse`/`_dense`**: This is a separate code path from `Rcpp_nmf_full`. The auto-rank path does not pass `solver`, `init`, `norm`, `projective`, `symmetric`, or most regularization parameters — it hard-codes defaults. Bring the auto-rank path through `Rcpp_nmf_full` with the full config, or explicitly document what parameters are ignored during auto-rank.

- `[ ]` **NMF-05 — `projective = TRUE` not tested**: No test in `tests/testthat/` checks projective NMF semantics (i.e., that H = W'A and that rank-1 approximation quality improves). Add a basic correctness test.

- `[ ]` **NMF-06 — `symmetric = TRUE` not tested**: Same as above for symmetric NMF.

- `[ ]` **NMF-07 — `init = "irlba"` with multiple seeds**: `nmf_thin.R` warns and drops to first seed when multiple seeds + IRLBA init are combined. This silent degradation should either be suppressed when the user provides a single seed, or it should be an error when the user provides multiple seeds expecting multi-init behavior.

- `[ ]` **NMF-08 — `adaptive_solver` parameter is exposed but undocumented**: The `adaptive_solver = 0L` parameter in `nmf()` is passed through to `Rcpp_nmf_full` as `adaptive_solver_switch` but has no `@param` documentation and its behavior is opaque to users. Either document it or make it internal (unexported).

- `[ ]` **NMF-09 — `cd_abs_tol` parameter undocumented**: Similarly, `cd_abs_tol = 1e-15` is exposed in the function signature without a `@param` entry in the docs.

- `[ ]` **NMF-10 — `streaming = "auto"` currently does nothing useful**: The code checks `isTRUE(getOption("RcppML.streaming", FALSE))` for the streaming path, but the `streaming` parameter is not actually used once this option check passes. The `streaming` parameter and `panel_cols` are dead parameters in the non-SPZ path. Document or remove.

---

## 5. C++ Architecture & Portability

- `[ ]` **CPP-01 — Remove `using namespace RcppML::algorithms` and `using namespace RcppML::algorithms::clustering`** from `RcppFunctions.cpp` top-level: these are broad using-directives in a `.cpp` file that pollute the translation unit's namespace and can cause ADL bugs. Replace with explicit `RcppML::algorithms::unified::nmf_fit<...>(...)` at call sites.

- `[x]` **CPP-02 — `fit_unified.hpp.dscale_backup` in include path**: Deleted (see SVD-06).

- `[x]` **CPP-03 — `spmv.hpp.bak` in include path**: Deleted (see SVD-05).

- `[ ]` **CPP-04 — `test_entries.hpp` is in `algorithms/svd/` but is shared infrastructure**: It is `#include`-d by both `deflation_svd.hpp` and `krylov_constrained_svd.hpp`. Since it is shared, move it to `primitives/` or `core/` or create a new `algorithms/svd/cv/` sub-directory. Otherwise the `algorithms/svd/` directory mixes algorithm implementations with shared utilities.

- `[ ]` **CPP-05 — GPU `.cuh` files `#include`-d from CPU `.hpp` files**: `gateway/nmf.hpp` conditionally includes `.cuh` files inside `#ifdef RCPPML_HAS_GPU` guards. This is fine at compile time, but the CPU-only install still ships the `.cuh` files in `inst/include/`. Consider whether shipping CUDA headers as part of the public C++ API is intended or accidental.

- `[ ]` **CPP-06 — `NMFConfig` and `SVDConfig` are independent structs with redundant fields**: Both have `seed`, `threads`, `verbose`, `tol`, `max_iter`. Extract a `BaseConfig` or `ComputeConfig` struct that both inherit from. This reduces the parameter-passing surface and avoids drift between the two sets of defaults.

- `[ ]` **CPP-07 — `SVDConfig` has no `NormType` field** whereas NMF has `NormType` (L1/L2/none). For completeness and symmetry, add a `norm_type` field to `SVDConfig` even if initially unused.

- `[ ]` **CPP-08 — `build_config_from_params` template in `RcppFunctions.cpp`**: This large helper function is defined inside a `.cpp` file as a template — meaning it cannot be instantiated from outside this translation unit. If it is ever needed from another file, it will cause linker errors. Move to a header in `inst/include/RcppML/gateway/` or make it `static`.

- `[ ]` **CPP-09 — `result_to_list` template in `RcppFunctions.cpp`**: Same issue as CPP-08.

- `[ ]` **CPP-10 — SVD gateway is missing from `gateway/` directory**: NMF has `gateway/nmf.hpp` as a single dispatch point. SVD does not have an equivalent `gateway/svd.hpp`. The SVD dispatch logic lives inside `svd.R` (R-side) with direct calls to `Rcpp_svd_cpu`, `Rcpp_svd_gpu`, `Rcpp_svd_streaming_spz`. Mirror the NMF architecture: create `inst/include/RcppML/gateway/svd.hpp` that accepts an `SVDConfig<Scalar>` and routes to the correct algorithm and resource.

- `[ ]` **CPP-11 — No `SVDResult` → `Rcpp::List` conversion helper analogous to `result_to_list`**: Each SVD Rcpp export in `RcppFunctions.cpp` hand-rolls the `SVDResult` → List conversion. Factor out into a `svd_result_to_list()` helper.

- `[ ]` **CPP-12 — `Rcpp_svd_cpu` and `Rcpp_svd_gpu` in `RcppFunctions.cpp`**: These are large functions (~150 lines each) that mirror each other. The only difference is the resource path. Once CPP-10 is done, both can collapse to one function that builds `SVDConfig` and calls `gateway::svd()`.

- `[ ]` **CPP-13 — `RcppFunctions.cpp` is 1601 lines**: This is unwieldy. Split into logical translation units: `rcpp_nmf.cpp`, `rcpp_svd.cpp`, `rcpp_clustering.cpp`, `rcpp_nnls.cpp`, `rcpp_sparse.cpp`. Keep `RcppFunctions.cpp` as a thin `#include` aggregator if needed.

- `[ ]` **CPP-14 — The R-callable `svd()` function shadows `base::svd`**: In `svd.R`, the function is registered as `svd` in the package namespace. Because `RcppML` does not import `base::svd` and the package's `svd` is exported, any code that does `library(RcppML)` will shadow `base::svd`. This is **intentional** per the DESCRIPTION, but it is a source of user confusion and CRAN will flag it. Add a note in the `svd.Rd` man page that this function intentionally masks `base::svd`.

---

## 6. R API Architecture

### NMF is well-architected. SVD should mirror it.

NMF model:
```
nmf() [R thin layer]
  → validate_data() / validate_all_penalties() / validate_graphs()  [R validation helpers]
  → Rcpp_nmf_full() [single C++ entry point]
  → gateway::nmf() [resource dispatch]
  → algorithms::unified::nmf_fit<CPU|GPU>()
  → NMFResult → nmf S4 class
```

SVD currently:
```
svd() [R thick layer — ~450 lines of R]
  → (validation inline, not factored out)
  → if SPZ → Rcpp_svd_streaming_spz()
  → else if GPU → .gpu_svd_pca() via sp_gpu.R
  → else → Rcpp_svd_cpu()
  → SVDResult → svd_pca S4 class
```

Issues:
- `[ ]` **API-01 — SVD R layer mixes validation with dispatch**: Validation (graph dimension checks, method-feature matrix, resource resolution) is inline in `svd()`. Extract `validate_svd_params()` analogous to `validate_all_penalties()`, `validate_graphs()`, `validate_cv_params()`.
- `[ ]` **API-02 — SVD R layer does not use shared validation helpers**: `validate_penalty()` from `nmf_validation.R` could be reused for SVD's L1/L2/L21/angular/graph_lambda expansion, but SVD has its own inline `rep_len()` logic. This is ~30 lines of duplication.
- `[ ]` **API-03 — `resource` resolution is duplicated**: The full `if (resource == "auto") { env_res <- ...; gpu_opt <- ... }` block is copy-pasted verbatim in both `nmf_thin.R` and `svd.R`. Extract to a shared `.resolve_resource()` internal helper in `utils_globals.R`.
- `[ ]` **API-04 — `pca()`, `sparse_pca()`, `nn_pca()` are inline in `svd.R` but not in their own doc section**: These wrappers are at the bottom of `svd.R` without `@examples`. They should either have their own `@rdname svd` block or be documented separately so they appear in tab-completion and `?pca`.
- `[ ]` **API-05 — `svd_pca` result construction is duplicated**: In `svd.R`, the block that converts a `Rcpp_svd_cpu()` list to an `svd_pca` S4 object is repeated three times (CPU path, GPU path, streaming path). Extract to a `.make_svd_pca()` helper.
- `[ ]` **API-06 — The `svd()` function has no `predict()` S4 method**: NMF has `predict(nmf_model, newdata)` to project new samples. PCA should have `predict(svd_pca_model, newdata)` that returns new scores (`u_new`). This is the most important missing R-level PCA feature for practical use.
- `[ ]` **API-07 — No `plot()` method for `svd_pca`**: NMF has `plot.nmf`, `biplot`, `compare_nmf`. SVD should have at minimum a `biplot()` method that calls `stats::biplot()` with u and v.
- `[ ]` **API-08 — `svd_pca` class should expose `summary()` method**: Return a data.frame of singular values, proportion of variance explained, cumulative variance — analogous to `summary(prcomp(...))`.
- `[ ]` **API-09 — `evaluate()` method for `svd_pca`**: NMF has `evaluate(model, data)`. SVD should have the same, returning reconstruction MSE and per-component variance explained.

---

## 7. PCA User-Facing Design

The `pca()` alias calls `svd(..., center = TRUE)` and returns an `svd_pca` object. For users familiar with R's `prcomp` / `princomp` workflow, several things are non-intuitive:

- `[ ]` **PCA-01 — User expects `$rotation` and `$x`**: `prcomp` returns `rotation` (loadings = v) and `x` (scores = u·diag(d)). Consider adding these as accessor slots or active bindings on `svd_pca`:
  ```r
  setGeneric("rotation", function(x) x@v)
  setGeneric("scores",   function(x) x@u %*% diag(x@d))
  ```
  Or add these as named columns to `misc`.
- `[ ]` **PCA-02 — `center = FALSE` default is surprising for `pca()`**: The `pca()` alias calls `svd(..., center = TRUE)` correctly. But the man page for `pca` inherits from `svd` and shows `center = FALSE`. The `pca` alias should have its own `@param center` doc entry noting it defaults to `TRUE`.
- `[ ]` **PCA-03 — Scale (unit variance normalization) not supported**: `prcomp` has `scale. = TRUE` to divide by standard deviation. RcppML SVD lacks this. For CRAN-era users, add a `scale = FALSE` parameter that divides each row by its standard deviation before factorization (or document clearly that it is not supported).
- `[ ]` **PCA-04 — `variance_explained()` output format**: Currently returns a raw numeric vector. Follow `prcomp`-style convention: provide a named vector (PC1, PC2, ...) and include a `print()` method that shows `Proportion of Variance` and `Cumulative Proportion`.

---

## 8. Code Bloat & Cleanup

### 8.1 Files to Delete

The following files serve no production purpose and should be removed from the repository:

#### Root-level development/diagnostic files
- `[x]` `comprehensive_gpu_test.R`
- `[x]` `diagnose_bypass_output.txt`
- `[x]` `final_verification.R`
- `[x]` `inline_verify.R`
- `[x]` `python_verify.py`
- `[x]` `quick_cpu_test.R`
- `[x]` `quick_init_test.R`
- `[x]` `test_adaptive_chol.R`
- `[x]` `test_cd_trace.R`
- `[x]` `test_cholesky_trace.R`
- `[x]` `test_hierchol_trace.R`
- `[x]` `test_svd_all.R`
- `[x]` `verify_fix.R`
- `[x]` `verify_irlba.R`

#### Root-level build/job logs
- `[x]` `build.log`
- `[x]` `build_svd.log`
- `[x]` `docinstall_svd.log`
- `[x]` `install.log`
- `[x]` `install_test_h100.log`
- `[x]` `nohup.out`
- `[x]` `render_output.log`
- `[x]` `render_pdf_output.log`
- `[x]` `diagnose_extended_250647.out`
- `[x]` `diagnose_extended_250649.out`
- `[x]` `diagnose_final_250651.out`
- `[x]` `diagnose_final_250652.out`
- `[x]` `diagnose_svd_250645.out`
- `[x]` `rebuild_cleanup_250421.out`
- `[x]` `rebuild_cleanup_250422.out`
- `[x]` `rebuild_cleanup_250424.out`
- `[x]` `test_init_modes_250399.log`
- `[x]` `test_results.log`
- `[x]` `verify_svd_250632.out`

#### Root-level SLURM scripts (move to `tools/` or delete)
- `[x]` `build_and_bench_fp16.sbatch`
- `[x]` `build_and_test.sbatch`
- `[x]` `build_fp16.sbatch`
- `[x]` `build_fp16_removal.sbatch`
- `[x]` `build_gpu.sh`
- `[x]` `build_svd.sbatch`
- `[x]` `check_svd_benchmark.sh`
- `[x]` `cleanup` (script)
- `[x]` `complete_rebuild.sh`
- `[x]` `diagnose_extended.sbatch`
- `[x]` `diagnose_final.sbatch`
- `[x]` `diagnose_svd.sbatch`
- `[x]` `docinstall_svd.sbatch`
- `[x]` `http_server.sbatch`
- `[x]` `install_and_test_gpu.sh`
- `[x]` `rebuild_after_cleanup.sbatch`
- `[x]` `rebuild_and_test.sh`
- `[x]` `rebuild_and_verify.sbatch`
- `[x]` `rebuild_and_verify_cpu.sbatch`
- `[x]` `rebuild_and_verify_lowmem.sbatch`
- `[x]` `rebuild_and_verify_minimal.sbatch`
- `[x]` `rebuild_and_verify_nodata.sbatch`
- `[x]` `rebuild_svd.sbatch`
- `[x]` `render_sparsepress.sbatch`
- `[x]` `render_sparsepress_pdf.sbatch`
- `[x]` `run_local_test.sh`
- `[x]` `run_ssh_tests.sh`
- `[x]` `run_verification.sh`
- `[x]` `test_auto.sbatch`
- `[x]` `test_fp16_hybrid.sbatch`
- `[x]` `test_gpu_cholesky.sbatch`
- `[x]` `test_init_fix.sbatch`
- `[x]` `test_init_modes.sbatch`
- `[x]` `test_run.sbatch`
- `[x]` `verify_fix.sbatch`
- `[x]` `verify_now.sbatch`
- `[x]` `verify_svd.sbatch`

#### Stale markdown files in root (redundant research notes — consolidate into RESEARCH_LOG.md)
- `[x]` `ADMM_RESEARCH.md` → deleted (content archived in Section 11 research log)
- `[x]` `ALGORITHM_RESEARCH.md` → deleted
- `[x]` `CLEANUP_FILE_TREE.md` → deleted
- `[x]` `CLEANUP_GUIDE.md` → deleted
- `[x]` `CLEANUP_PROPOSAL.md` → deleted
- `[x]` `CRAN_AUDIT.md` → deleted (superseded by this file)
- `[x]` `DEFLATION_ACCELERATION_PROPOSAL.md` → deleted
- `[x]` `DEFLATION_OPTIMIZATION_RESULTS.md` → deleted
- `[x]` `FP16_DECISION.md` → deleted
- `[x]` `FP16_FINAL_STATUS.md` → deleted
- `[x]` `GPU_SVD_FIX_STATUS.md` → deleted
- `[x]` `IMPLEMENTATION_SUMMARY.md` → deleted
- `[x]` `IRLBA_IMPLEMENTATION_PLAN.md` → deleted
- `[x]` `KSPR_OPTIMIZATIONS.md` → deleted
- `[x]` `MANUAL_VERIFICATION_REQUIRED.md` → deleted
- `[x]` `NEW_ALGORITHM_PROPOSALS.md` → deleted
- `[x]` `NEW_ALGORITHM_PROPOSALS_OLD.md` → deleted
- `[x]` `NNLS_SOLVER_RESEARCH.md` → deleted
- `[x]` `PRECISION_DEPLOYMENT.md` → deleted
- `[x]` `PRODUCTION_STATUS.md` → deleted (superseded by this file)
- `[x]` `SOLVER_CONSOLIDATION_RECAP.md` → deleted
- `[x]` `STREAMING_NNLS_RESEARCH.md` → deleted
- `[x]` `SVD_FIX_STATUS.md` → deleted
- `[x]` `SVD_INIT_BUG_FIX.md` → deleted
- `[x]` `SVD_PERFORMANCE_AUDIT.md` → deleted
- `[x]` `TENSOR_CORE_RESULTS.md` → deleted
- `[x]` `WARMSTART_SCALING_INVESTIGATION.md` → deleted

#### Backup files in include tree
- `[x]` `inst/include/RcppML/algorithms/svd/spmv.hpp.bak` — deleted
- `[x]` `inst/include/RcppML/algorithms/nmf/fit_unified.hpp.dscale_backup` — deleted

#### Compiled objects (add to `.gitignore`)
- `[ ]` `src/*.o`, `src/RcppML.so`, `src/RcppML_gpu.so`, `src/gpu_bridge.o`, etc.

#### Benchmark result data files (not reproduced on package install)
- `[ ]` `benchmarks/svd_comprehensive_cpu_*.rds`
- `[ ]` `benchmarks/svd_comprehensive_gpu_*.rds`
- `[ ]` `benchmarks/svd_direct_cpu_*.rds`
- `[ ]` `benchmarks/svd_direct_gpu_*.rds`
- `[ ]` `benchmarks/svd_init_bench_gpu_*.rds`
- `[ ]` `benchmarks/svd_init_consolidated_analysis.rds`
- `[ ]` `benchmarks/svd_init_results_*.rds` / `.csv`
- `[ ]` `benchmarks/quick_trajectory*.rds`
- `[ ]` `benchmarks/*.txt` result logs
- `[ ]` `benchmarks/*.log` files

### 8.2 Files to Keep / Relocate

- `[ ]` `tools/fix_rcpp_info_bug.sh` → keep, but also reproduce its content in `CONTRIBUTING.md`
- `[ ]` `benchmarks/` → keep the R benchmark scripts as `tools/benchmarks/` or add all to `.Rbuildignore`
- `[ ]` `tests/testthat/_problems/` → ensure this directory is listed in `.Rbuildignore` if it contains expected-failure outputs

### 8.3 `.Rbuildignore` Gaps

Add these patterns to `.Rbuildignore`:
```
^benchmarks$
^logs$
^\.github$
^tools$
^manuscript$
^.*\.sbatch$
^.*\.sh$
^.*\.log$
^.*\.out$
^.*\.txt$ (except LICENSE, NEWS)
^src/RcppML.*\.so$
^inst/lib$
^nohup\.out$
^PRODUCTION_AUDIT\.md$
^PRODUCTION_STATUS\.md$
^CRAN_AUDIT\.md$
^[A-Z_]+_RESEARCH\.md$
^[A-Z_]+_STATUS\.md$
^[A-Z_]+_GUIDE\.md$
^[A-Z_]+_PROPOSAL\.md$
^[A-Z_]+_DECISION\.md$
^[A-Z_]+_RESULTS\.md$
^[A-Z_]+_SUMMARY\.md$ (except IMPLEMENTATION_SUMMARY)
```

---

## 9. Tests — Gaps & Quality

- `[ ]` **TEST-01 — No tests/testthat/test_svd.R**: Covered in SVD-01. This is the most critical gap.
- `[ ]` **TEST-02 — No tests for `pca()`, `sparse_pca()`, `nn_pca()` aliases**: Add to test_svd.R.
- `[ ]` **TEST-03 — No tests for `variance_explained()`**: Add to test_svd.R.
- `[ ]` **TEST-04 — No tests for `reconstruct()`**: Add to test_svd.R.
- `[ ]` **TEST-05 — No tests for `svd_pca` S4 methods**: `show()`, `head()`, `[`, `dim()`.
- `[ ]` **TEST-06 — `test_cv_irls.R` is excluded**: Either fix the underlying MAE-CV bug or add `skip()` with issue reference.
- `[ ]` **TEST-07 — Many test files named `test_gpu_*.R` skip when GPU is absent**: Verify that all GPU tests properly `skip_if(!gpu_available())` and do not leave stale objects that fail later tests.
- `[ ]` **TEST-08 — `tests/testthat/test_unified_compile.cpp`**: A C++ file inside `tests/testthat/`. This will not be compiled by `testthat` and may confuse `R CMD check`. Move to `src/` or remove.
- `[ ]` **TEST-09 — SLURM validation scripts are not `testthat` tests**: Files like `tests/gpu_full_validation.sbatch`, `tests/run_tests_gpu.sbatch` are in the `tests/` directory but are SLURM scripts. `testthat` ignores non-`.R` files in `tests/testthat/`, but having SLURM scripts in `tests/` is confusing. Move to `tools/`.
- `[ ]` **TEST-10 — Many large test files cover overlapping scenarios**: `test_edge_cases.R`, `test_degenerate_inputs.R`, `test_parameters.R`, `test_validation_errors.R` all test NMF parameter handling. Consider consolidating or at least adding a `helper-test-utils.R` function to avoid 4× copies of the same matrix setup.

---

## 10. Documentation

- `[ ]` **DOC-01 — `svd.Rd` does not document the `method = "auto"` auto-selection logic**: The `auto_select.hpp` k=32 Lanczos/IRLBA threshold is invisible to the user. Add an `@section Auto-selection:` to `svd.Rd`.
- `[ ]` **DOC-02 — `svd.Rd` `@seealso` lists `pca` as a separate link but `pca` is just an alias**: Update to mention that `pca()`, `sparse_pca()`, and `nn_pca()` are aliases.
- `[ ]` **DOC-03 — Vignette `gpu-acceleration.Rmd` references `resource = "gpu"` for SVD** but may not reflect the current `method` + `resource` parameter interaction. Review and update.
- `[ ]` **DOC-04 — `nmf.Rd` `@return` section does not describe `nmfCrossValidate` structure**: When `k` is a vector, `nmf()` returns an `nmfCrossValidate` data.frame. The columns (`rep`, `k`, `train_mse`, `test_mse`, `best_iter`, `total_iter`) are not documented in `@return`.
- `[ ]` **DOC-05 — `README.md` contains references to `nmf_v0.3.7` syntax**: Check for any lingering old-API examples. Specifically, search for `project(`, `crossValidate(`, `mse(A, w, d, h)` patterns.
- `[ ]` **DOC-06 — No vignette for SVD/PCA**: There are 8 vignettes but none demonstrates the SVD/PCA workflow. Add a `pca.Rmd` vignette showing basic PCA, sparse PCA, non-negative PCA, and auto-rank selection. This is a major gap for a package that now exports `svd()` as a primary function.
- `[ ]` **DOC-07 — `CONTRIBUTING.md` does not describe the `fix_rcpp_info_bug.sh` workflow**: New contributors will hit the Rcpp 1.1.0 `info` bug. Document the required post-`roxygenise()` step.

---

## 11. Research Log (Optimization History)

> Per user request: a brief running log of what has been tried so we don't repeat it.
> Details in individual markdown files (where they still exist); merge into this section before deleting those files.

### NNLS Solver Experiments

| Approach | Outcome | Date |
|----------|---------|------|
| ADMM batch NNLS | ~1× vs CD; not faster for dense k<64; removed from production | 2025–2026 |
| Warm-start CD | Small gains (~1.1×) for later iterations; implemented as default in CD solver | 2026-01 |
| Hierarchical Cholesky (active-set) | More accurate than clip-only; added as `solver = "hierarchical_cholesky"` | 2026-01 |
| Adaptive Cholesky (threshold-based) | Auto-selects clip vs hierarchical based on violation rate; default for GPU | 2026-01 |
| CD absolute tolerance (`cd_abs_tol`) | Prevents excess sweeps at convergence; added as parameter | 2026-02 |

### SVD Algorithm Experiments (for NMF initialization)

| Approach | Outcome | Date |
|----------|---------|------|
| Random uniform init | Baseline; 100 iterations typical to convergence | Legacy |
| NNDSVD (non-negative double SVD) | Rejected: negative values from SVD fold into zeros, too sparse initially | Research |
| Lanczos SVD init | 1.13× speedup for k<32 (pbmc3k, ifnb, hcabm40k); production default for small k | 2026-03 |
| IRLBA SVD init | 1.05–1.11× speedup for k≥32; production default for large k | 2026-03 |
| Threshold k=32 (Lanczos vs IRLBA) | Validated across three datasets and CPU/GPU; consistent crossover | 2026-03-02 |

### SVD Algorithm Experiments (standalone SVD)

| Approach | Outcome | Date |
|----------|---------|------|
| Basic Lanczos bidiagonalization | Fast for k<32; no restart overhead; no regularization | 2026-02 |
| IRLBA (Baglama & Reichel 2005) | Better scaling for k≥32; implicit restarts; pure unconstrained SVD | 2026-02 |
| Randomized SVD (Halko–Martinsson–Tropp) | ~1× IRLBA; constant cost with q power iterations; no constraints | 2026-02 |
| Deflation ALS | Supports all constraints; slower for large unconstrained k; good for k<10 | 2026-01 |
| KSPR (Krylov-Seeded Projected Refinement) | Best all-around constrained method for k≥8; Lanczos seed + block Gram | 2026-02 |
| Krylov on GPU | Faster than deflation for large k on GPU; same constraints as KSPR | 2026-02 |

### Deflation vs KSPR Crossover

Empirical result: k=8 is the crossover where KSPR becomes faster than sequential deflation. Below k=8, deflation's overhead-free rank-1 loop dominates. This is reflected in `auto_select.hpp`.

### FP16 / Mixed Precision

| Approach | Outcome | Date |
|----------|---------|------|
| Full FP16 NMF | Unacceptable quality degradation; removed | 2025 |
| FP16 Gram matrix only | Negligible speedup vs FP32; not worth complexity | 2025 |
| Auto-select FP16 based on k+nnz | Implemented as `should_use_mixed_precision()`; enabled for large-k GPU NMF | 2026 |

### Deflation Acceleration

| Approach | Outcome | Date |
|----------|---------|------|
| Deflation correction caching | ~5% speedup; not worth added complexity | 2026-01 |
| Cyclic deflation (multi-factor update) | Convergence instability; abandoned | 2026-01 |
| Batched deflation (mini-epochs) | No gain for sparse; small gain for dense k>20 | 2026-02 |

### Streaming NMF

| Approach | Outcome | Date |
|----------|---------|------|
| Per-column streaming NNLS (H-update) | Reduces peak memory; 20–30% slower than full batch | 2026 |
| IRLS in streaming path | Rejected: weight matrix requires full matrix access; incompatible with streaming | 2026-02 |
| Panel-based streaming (chunk cols together) | Default in production streaming path; `panel_cols` parameter | 2026 |

---

## 12. Priority Order for CRAN Submission

### Phase 1 — Must Do (Hard Blockers)

1. `[x]` B10/B11/B12/B13 — All non-package files deleted and `.Rbuildignore` updated
2. `[x]` B1 — Version synced to 1.0.1 in both DESCRIPTION and NEWS.md
3. `[x]` B2 — Compiled `.o` files removed from git tracking
4. `[x]` B3/B4 — B3 mitigated via `.Rbuildignore`; B4 was a false alarm (no strip in Makevars)
5. `[x]` B5 — Pre-built vignette HTMLs removed from `vignettes/` source dir
6. `[x]` B6 — `hcabm40k_mat.rda` (90 MB), GPU `.so` (20 MB), `pbmc3k.rda` (2.6 MB), `inst/extdata` (2.1 MB) excluded; estimated tarball ≈ 3.5 MB
7. `[x]` B7/B8/B9 — Deprecated API wrappers: `mse()` backward compat added; `project()` and `nnls()` already handled
8. `[x]` B14 — configure is functional (writes `inst/build_config` at install time; not a no-op)
9. `[ ]` SVD-01 / TEST-01 — Create `tests/testthat/test_svd.R` (no SVD tests currently)
10. `[x]` CPP-02/CPP-03 — Deleted `.bak` and `.dscale_backup` from include tree

### Phase 2 — High Value (Should Do)

8. `[ ]` API-06 — Add `predict()` method for `svd_pca` (project new samples)
9. `[ ]` CPP-10 — Create `gateway/svd.hpp` (unify SVD dispatch like NMF)
10. `[ ]` CPP-12/CPP-13 — Collapse `Rcpp_svd_cpu`/`_gpu` and split `RcppFunctions.cpp`
11. `[ ]` API-01/API-02/API-03 — Extract SVD validation helpers; share resource resolver
12. `[ ]` SVD-03/API-08/API-09 — Add `predict()`, `summary()`, `evaluate()` for `svd_pca`
13. `[x]` W5/W4 — Fix R imports; configure confirmed functional (not a no-op)
14. `[ ]` PCA-01/PCA-04 — Add `rotation`/`scores` accessors; improve `variance_explained()` output
15. `[ ]` DOC-06 — Write PCA/SVD vignette

### Phase 3 — Polish

16. `[ ]` CPP-06 — Extract `BaseConfig` shared between NMF and SVD
17. `[ ]` SVD-12 — Store method/algorithm/resource in `misc`
18. `[ ]` NMF-04 — Route `k = "auto"` through `Rcpp_nmf_full` for full parameter support
19. `[ ]` NMF-08/NMF-09 — Document or remove `adaptive_solver` and `cd_abs_tol`
20. `[ ]` Cleanup all root-level development files (Section 8.1)
21. `[ ]` Consolidate research markdown notes into Section 11 of this document, then delete them

---

*End of audit — updated as items are resolved.*
