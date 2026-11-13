# RcppML 1.0: findings and proposed plan

13 November 2026. Based on `main` at `49df8da`.

## Recommendation

Hold the CRAN resubmission and ship a correctness release first. The ten open issues are mostly symptoms of two deeper problems. The default solver in 1.0.0 is an approximation that gives worse fits than 0.3.7, and many options are accepted without effect on some fit paths. Both went unnoticed because the test suite compares the package with itself: all 2,161 CPU expectations pass on Clipper while every reported bug reproduces.

The plan below fixes defaults and silent failures first, then the mathematics of individual algorithms, then GPU parity and packaging. Each fix lands with a test against an independent reference, so the suite starts to mean something.

## Evidence

Everything ran on Clipper against a fresh clone (R 4.5.3, GCC 13.3, CUDA 12.9, two V100S cards).

- `R CMD check --as-cran`: 1 error, 2 warnings, 3 notes.
- Full test suite: CPU 0 failures and 155 skips; GPU 9 failures and 3 errors.
- A reproduction script for each issue that describes a bug, including installs of 0.3.7.1 and 0.5.6 for comparison.
- Eight line-by-line code reviews: R layer and packaging, core fit paths, GPU and file format, then five reviews that derived each update from its objective (solvers and penalties, distributions, cross-validation, SVD/PCA, clustering and networks).
- A new correctness harness, `tools/correctness/sweep.R`, that checks the package against reference math written in plain double-precision R. It covers solver exactness, each penalty on 19 fit paths, seven simulated distributions, cross-validation, SVD/PCA against `base::svd` and `prcomp`, clustering and utilities. Result tables are in `tools/correctness/results/`.

Findings are marked **[run]** when a Clipper run confirmed them and **[read]** when they rest on code reading alone. The second kind needs a failing test before any fix.

## The open issues

| Issue | Status | Cause | Fix |
|---|---|---|---|
| #79 fewer feature-specific rows in W | Reproduced **[run]** | `solver = "auto"` picks an unconstrained solve followed by clipping at zero whenever k < 32 and L1 = 0. On pbmc3k at k = 21, 1.0.0 gives 0 such rows and a 2% higher error; `solver = "cd"` gives 62, matching 0.3.7 (71) and 0.5.6 (67). | Make exact NNLS the default (see A1). |
| #82 graph penalty ignored for `loss = "gp"` | Reproduced **[run]**, and wider | The IRLS path rebuilds the Gram matrix after penalties are applied. Graph, L21 and target are dropped for every non-Gaussian loss, for `robust`, for zero inflation and for custom masks. | Apply penalties after the weighted Gram is built; one shared routine for all paths. |
| #75 target regularization ignored on GPU | Reproduced **[run]** | The GPU bridge never passes `target_H`. Masks and zero inflation are dropped the same way. | Pass them, or fall back to CPU with a message. |
| #81 `train_mse` with `track_train_loss = FALSE` | Reproduced **[run]** | Train loss is computed only at the first and last possible iteration and reported as 0 otherwise. | Return `NA` when not tracked. |
| #76 GPU not detected | Reproduced **[run]** | No install route builds the GPU library. `configure` reports "CUDA: found" and stops; the sources are excluded from the tarball. | `install_gpu()` helper and a `gpu_status()` diagnostic (see D). |
| #78 `configure`/`cleanup` not executable | Reproduced **[run]** | Committed as mode 644. `configure` also does nothing: `Makevars.in` has no placeholders. | Delete `configure`, `cleanup`, `Makevars.in`, `inst/build_config`. |
| #86 GPU tests on a simulated GPU | Their failure list matches ours on real hardware **[run]** | Two tests pass arguments `nmf()` rejects; Lanczos returns wrong trailing singular values on a rank-5 matrix; three tolerance misses; seven tests skip because a header is missing from the repo. | Fix the tests and the Lanczos bug (C1). Decide separately whether to adopt their simulator in CI. |
| #77 CRAN timeline | Open | See "Packaging" below. | Reply once the correctness release is tagged. |
| #85 Brunet / KL algorithm | Question | `loss = "gp", dispersion = "none"` is the KL objective, fit by IRLS, not multiplicative updates. It is currently biased on sparse input (B1). | Answer after B1; add a "which loss is which" table to the docs. |
| #72 PyTorch/Triton port | Discussion | — | Leave open; out of scope here. |

## Correctness findings

### A. NMF solvers and penalties

1. **Default solver is not NNLS [run].** Coordinate descent reaches the exact optimum of each half-step and tracks an exact reference iteration for iteration. The clipped-Cholesky solver does not, and the gap grows with rank: 17% higher final error at k = 40. It is the default below k = 32.
2. **Options that do nothing on some paths [run].** `mask = "zeros"` outside cross-validation, `mask = "NA"` (NAs are fit as zeros), `h_init`, the four SVD-based `seed` strings (results identical to random init), `huber_delta`, `convergence`, `sparse`, and `target_H` under cross-validation. The full penalty-by-path matrix is in `results/sweep_B_cpu.csv`.
3. **L21 does not zero out features [run]; the cause is [read].** It never produced an all-zero row at any strength. The reweighting uses the unit-normalized previous iterate, so it acts as a ridge.
4. **Graph regularization is weak and path-dependent [run].** With a Laplacian it smooths on the sparse coordinate-descent path only. The code adds a k×k term to the Gram and never couples neighboring samples **[read]**.
5. **L1 means different things on different paths [read].** Absolute on the main Gaussian path, scaled by the Gram diagonal in cross-validation, IRLS and `nnls()`. `nnls()` with L1 matched none of four candidate definitions **[run]**.
6. **`upper_bound` is applied by clipping after the solve inside `nmf()` [read]**; `nnls()` enforces it correctly inside the solver **[run]**.
7. **`predict()` returns H already multiplied by d and also keeps d [run]**; `refine()` has the same double count **[read]**.
8. **`target_H` of the wrong shape is accepted [run]** and read out of bounds **[read]**. Infinite values give a `NaN` loss without an error **[run]**.
9. `angular` works as a decorrelation heuristic **[run]** but has no stated objective and forces factors non-negative **[read]**. Projective and symmetric NMF are heuristics; projective ignores H-side options **[run]**.

### B. Distributions

1. **Count losses on sparse input are biased [run].** Stored zeros get unit weight instead of the distribution's weight. On simulated negative-binomial data the fit from a sparse matrix has four times the deviance of the fit from the same data stored dense, and is worse than a plain least-squares fit. Poisson at low counts shows the same pattern. Since count data is nearly always sparse, this is the common case.
2. **Inverse Gaussian fits diverge [run]** on correctly simulated data, sparse or dense.
3. **Negative-binomial size estimates hit the upper clamp [run]** (mean above 20,000 for a true value of 5).
4. **Reported loss is not comparable across paths [run].** It is a sum on some paths and a mean on others, covers non-zeros only for sparse input, and goes negative for count losses.
5. **A custom mask with a non-Gaussian loss may solve the Gaussian problem [read].** Unconfirmed: in our run the two fits differed slightly, so this needs a sharper test before anyone acts on it.
6. Zero inflation works on CPU **[run]** but makes fits worse on GPU **[run]**. It densifies the matrix every iteration **[read]**.
7. `robust` works **[run]**. Gamma and Tweedie beat least squares on their own data **[run]**.
8. No input validation: negative data with count losses, zeros with gamma, and any `tweedie_power` are accepted **[run]**.

### C. Cross-validation and rank selection

1. **A cross-validated fit returns a wrong model [run].** H is scaled by d and d is returned too, so `w %*% diag(d) %*% h` is off by a factor of d. Anyone who used the factors from `nmf(..., test_fraction = )` has this.
2. **`k = "auto"` returns the upper bound [run]**: 50 on all three rank-5 data sets tried.
3. **Rank selection with count losses fails [run].** With `gp`, test loss falls monotonically and picks the largest rank; with `nb` it is flat. Least-squares cross-validation behaves sensibly.
4. **The holdout fraction is quantized [run].** Requests between 0.21 and 0.25 hold out the same entries; anything above 0.5 holds out everything. The default 0.1 holds out one ninth **[read]**.
5. `best_iter` exceeds `total_iter` in most rows **[run]**. The reported test loss is the minimum over iterations while the factors are from the last one **[read]**.
6. Coordinate descent inside cross-validation is warm-started without forming the residual **[read]**. This would explain C3 and needs a targeted test.
7. `svd(..., test_fraction = )` fills held-out entries with zeros, which shrinks every singular value by the holdout fraction **[run]**. A custom `mask` in `svd()` is treated as zeros rather than missing **[run]**. Rank selection for SVD nonetheless found the true rank in all six configurations **[run]**.

### D. SVD and PCA

1. **Lanczos and Krylov return too few or wrong singular values in special cases [run].** With repeated singular values they return three factors when five are requested. On an exactly rank-5 matrix the GPU versions return 4.28 and 0.39 where 8 and 4 are correct. The breakdown handler discards a coupling term **[read]**.
2. **IRLBA returns values that are not singular values on flat spectra [run]** (for example 16.97 where the true second value is 16.81). A restart bookkeeping error is the suspected cause **[read]**.
3. Deflation, Lanczos and Krylov match `base::svd` to single precision on well-separated spectra, and orthogonality holds throughout **[run]**. Randomized SVD is 1–4% low on flat spectra, as expected for three power iterations; the docs should say so.
4. `pca()` matches `prcomp` when rows are variables, which contradicts its own example **[run]**. With `scale = TRUE` it uses the population standard deviation **[run]**.
5. L2 and L21 do nothing in deflation **[run]**. Constrained `d`, variance explained and `reconstruct()` are mutually inconsistent **[read]**.
6. The GPU deflation path still lacks the October fix for constrained factors **[read]**.

### E. Clustering, networks, utilities

1. `dclust()` split four clean clusters into eight (adjusted Rand index 0.52) **[run]**. Its distance measure is a similarity ratio that is zero for perfectly separated clusters, so `min_dist` rejects the best splits **[read]**. `bipartition()` itself is correct **[run]**.
2. `cosine(vector, matrix)` is wrong (a missing square root) **[run]**. `bipartiteMatch()` is optimal but errors on negative costs **[run]**. `model[1]`, `model$L1` and `head(model)` error **[run]**.
3. Multi-modal `nmf(list(...))` requires names and forwards only four arguments **[run]**. Deep networks are fit greedily layer by layer, and multi-modal W rows may be assigned to the wrong modality when names are not alphabetical **[read]**.
4. Fitting from an `.spz` file gives results identical to in-memory **[run]**, because the streaming dispatcher is a stub and the file is simply loaded. The chunked code path is unreachable from R and carries its own defects **[read]**.
5. The `.spz` v2 reader does not validate sizes or offsets read from the file **[read]**. A malformed file can write out of bounds, which CRAN's sanitizer checks will also catch.

### F. GPU

Results are not bit-reproducible run to run **[run]**. The GPU library is searched for in the working directory before the installed package **[read]**, which is both a stale-library hazard and a CRAN objection. `st_read_gpu()` is exported but can never work because its header is not in the repo **[run]**.

## Packaging and CRAN

- **Core use.** Examples, tests and vignettes run on all cores (CPU time 10–16× elapsed); CRAN allows two. This alone is grounds for silent rejection.
- **Check time.** About nine minutes with 16 cores. Vignette rebuild is the largest part.
- **`SeuratData` in Suggests** is not on CRAN and is what breaks GitHub Actions today.
- **C++17 is used but not declared**, while `DESCRIPTION` allows R 4.1.
- **One example fails under `--run-donttest`** (an interactive plot in a headless session).
- **Size.** 8.8 MB tarball against a 5 MB guideline; 6.5 MB is data.
- **Attribution.** `RcppHungarian.h` is third-party BSD code with no credit in `Authors@R` or `inst/COPYRIGHTS`.
- **Reverse dependencies.** `project()`, `crossValidate()` and `mse()` were removed without deprecation stubs, and the `nnls()` compatibility shim does not cover the positional call that phytoclass uses **[read]**. None of the five reverse dependencies was run.
- **Repository.** 574 of 1,093 tracked files are a committed pkgdown site; most scripts in `tools/` hard-code a cluster path.

## Proposed plan

**Phase 0 — make failures visible (about a week).** Move the harness into `tests/testthat` as reference-based tests, marked as expected failures where a defect is known. Fix CI by dropping `SeuratData`. Add a pinned notice to the repository naming the three defects users are most likely to hit (A1, B1, C1) with workarounds: `solver = "cd"`, dense input for count losses, and rescaling H from cross-validated fits.

**Phase 1 — defaults and silent failures.** Exact NNLS by default. One validation layer that either applies an option on the chosen path or stops with a clear message, driven by a single support table. Fix the cross-validation return value, the holdout fraction, `train_mse`, NA and zero masks, `predict()`, and input validation. This closes #79, #81, #82 and the CPU half of #75.

**Phase 2 — algorithm fixes.** IRLS weights for sparse input and in cross-validation, inverse Gaussian, dispersion estimates, a single definition of reported loss. Graph, L21, L1 scaling and bounds inside the solver. Lanczos breakdown, IRLBA restarts, masked SVD. `k = "auto"`. `dclust` distance. Each begins with a failing reference test.

**Phase 3 — GPU.** Parity with the CPU support table or explicit fallback, the deflation fix, `install_gpu()` and `gpu_status()`, loading only from the installed location. Closes #75 and #76 and most of #86.

**Phase 4 — CRAN.** Two-core cap in examples, tests and vignettes; precomputed vignettes; declare C++17; remove `configure`; trim data; attribution; deprecation stubs; run all reverse dependencies on Clipper; bounds checks in the `.spz` reader plus a sanitizer run. Then resubmit and answer #77.

**Phase 5 — documentation.** Rewrite the regularization, distributions and cross-validation vignettes to state the objective each option optimizes. Answer #85.

## Decisions needed

1. **Default solver.** Switching to coordinate descent restores 0.3.7 quality at a speed cost (1.4 s against 0.3 s on pbmc3k at k = 21). The alternative is to add the active-set refinement that 0.3.7 used after the Cholesky solve, which keeps most of the speed and is exact. I recommend the second, with coordinate descent as the interim default in Phase 1.
2. **Scope.** Several features are heuristics or unreachable: `angular`, projective NMF, deep and conditional networks, the zero-copy GPU path, `st_read_gpu()`, chunked streaming. I recommend marking them experimental or removing them from 1.0.x, so the release carries only what has a reference test.
3. **Precision.** Everything runs in single precision. That is adequate for NMF but limits SVD accuracy and makes small `tol` values unreachable. Offering a double-precision CPU path is a larger change that I would defer.
4. **Public notice.** Whether to post the Phase 0 notice now, before fixes land.
