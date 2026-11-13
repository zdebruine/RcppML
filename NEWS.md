# RcppML (development version)

This release corrects numerical defects found by comparing every fit path
with reference implementations written in plain R. Results change wherever a
defect was fixed; the sections below say where.

## Solvers and penalties

- The default solver is exact (#79). `solver = "cholesky"`, which
  `solver = "auto"` selects on CPU for `k < 32` without L1, was an
  unconstrained solve clipped at zero. It is now a Cholesky solve followed by
  a correction that satisfies the non-negativity conditions, and agrees with
  coordinate descent.
- Penalties are honored on every fit path (#82). L1, L2, L21, angular, graph
  and bound settings were dropped or applied differently on the IRLS
  (non-Gaussian and robust) and masked paths.
- L1 is an absolute shift of the right-hand side on every path, including
  factors without the non-negativity constraint, where the sign is now
  handled correctly.
- L21 groups are features for W and samples for H.
- The graph penalty uses the symmetric Laplacian as supplied.
- `upper_bound` is a constraint of the solve instead of a clip applied
  afterwards.

## Distributions

- Sparse input with a count loss weights the zero entries correctly.
- IRLS updates use step control.
- The reported loss has one definition on every path: the summed negative
  log-likelihood or deviance over all fitted entries.
- A mask combined with a non-Gaussian loss outside cross-validation is an
  error instead of being ignored.

## Cross-validation

- The model is scaled consistently between the training and test losses, and
  an entry is held out with probability exactly `test_fraction`, identically
  for sparse and dense input.
- `train_mse` is correct when tracked and `NA` when
  `track_train_loss = FALSE`; it was 0 or a last-iteration value (#81).
  `best_iter` is no longer off by one, and the documented column names match
  the returned ones (`train_mse`, `test_mse`).
- A cross-validated fit returns the factors of the best iteration.
- `mask = "zeros"` treats zeros as missing outside cross-validation too.
- `k = "auto"`: the rank search was rewritten and now receives the full
  configuration (loss, masks, all penalties, solver, seeds). Only `L1[1]`,
  `L2[1]`, `tol`, `maxit` and `threads` were forwarded before. The search
  range is clamped to `min(dim(data))`.

## `nmf()` interface

- Missing values. `mask = "NA"`, and the automatic handling of `NA`/`NaN`,
  build a mask from the missing positions. The values used to be replaced by
  zero and fitted as zeros. Missing positions are also added to a
  user-supplied mask.
- Input is validated before fitting, with a clear error for: infinite
  values; a matrix without non-zero values; `k` larger than
  `min(dim(data))`; negative values with any loss other than `"mse"`; zeros
  with `"gamma"`, `"inverse_gaussian"` or `"tweedie"` with
  `tweedie_power >= 2`; `tweedie_power` outside `[1, 3]`; a `target_H`,
  `w_init`, `h_init` or mask matrix of the wrong shape. Non-integer data
  with `"gp"` or `"nb"` gives a warning.
- `projective = TRUE` stops when H-side penalties, `nonneg[2] = FALSE` or a
  mask are supplied. `symmetric = TRUE` stops with a mask, a non-MSE loss or
  an H-side setting that differs from the W side (`graph_H`, `target_H`,
  `angular[2]`, `upper_bound[2]`). These settings were silently ignored.
- Arguments that were accepted and ignored are errors that name the
  replacement: `huber_delta` (use `robust`), `sparse` (use
  `mask = "zeros"`), `convergence` and `panel_cols`. Unnamed extra arguments
  are an error.
- `h_init` is honored: the starting W is the solution of the W update given
  `h_init`. It had no effect because every fit updates H first.
- A numeric `seed` no longer changes the caller's random number stream.
- `options(RcppML.threads = )` is honored by every function. It was
  documented but never read. Precedence: a non-zero `threads` argument, the
  option, all cores. The count is capped at 2 under `R CMD check` and at
  `OMP_THREAD_LIMIT` when that is set.
- A `.spz` path no longer replaces an explicit `resource = "gpu"` with
  `"cpu"`, and `streaming = TRUE` reports once that the file is loaded into
  memory.
- Multi-modal input: unnamed lists are accepted, and the fitted W is split
  back into the modalities in the right order.

## SVD and PCA

- `method = "deflation"` with `nonneg`, `L1` or `upper_bound` uses projection
  deflation: each factor is fitted with earlier constrained directions
  projected out. Hotelling deflation let later factors repeat earlier ones
  (pbmc3k, k = 16: 16 factors spanned 8 directions). Constrained factors are
  returned as fitted, without the Gram-Schmidt step that reintroduced
  negative entries, so they are not mutually orthogonal.
- `method = "auto"` selects `"deflation"` whenever `nonneg`, `L1` or
  `upper_bound` is set; `"krylov"` with these constraints and `k > 1` warns.
- Lanczos handles breakdown (an exhausted Krylov space) and IRLBA restarts
  correctly.
- Masked and held-out entries are treated as missing, so singular values are
  not biased toward zero. Methods that cannot do this (`"lanczos"`,
  `"irlba"`, `"randomized"`) reject a mask or `test_fraction`.
- `scale = TRUE` uses the sample standard deviation, and PCA results match
  `prcomp()` for the same orientation (rows are features, columns samples).
- A warning is issued when a method stops before reaching the tolerance.
- `convergence` accepts `"factor"`, `"loss"` and `"both"`.

## Clustering and utilities

- `dclust()` computes the distance between cluster centers correctly.
- Fixes in `predict()`, `evaluate()`, `align()` (it applied the matching
  permutation in the wrong direction), `cosine()` (vector against matrix),
  the AUC of the classifier metrics, and `refine()`.
- `bipartiteMatch()` accepts negative costs.

## GPU

- Options the GPU code does not support fall back to the CPU with a message
  instead of being ignored (#75).
- `install_gpu()` builds the GPU library from the installed package and
  `gpu_status()` reports what was searched, what was loaded and why not
  (#76). A normal install never built the library, and `gpu_available()`
  returned `FALSE` without explanation.
- The library is looked up in a per-user directory and the installed
  package, no longer in the working directory.

## StreamPress

- The `.spz` reader validates the file before reading it.

## Compatibility with 0.3.7

- `nnls(a, b)`, `nnls(a, b, cd_maxit, cd_tol)` and `nnls(a = , b = )` are
  accepted with a deprecation warning and return the 0.3.7 result, the
  solution of `a x = b` subject to `x >= 0` for symmetric positive definite
  `a`. In 1.0.0 the two-argument form solved a different least-squares
  problem and the four-argument form was an error.
- `project()`, `crossValidate()` and `mse()` are back as deprecated wrappers.
- `nmf(mask_zeros = TRUE)` and `nmf(A = )` are accepted with a deprecation
  warning.

## Installation and packaging

- The `configure` and `cleanup` scripts are gone (#78). They generated a
  `Makevars` identical to the committed one, and a plain checkout failed to
  install because `configure` was not executable.
- C++17 is declared (`CXX_STD = CXX17`, `SystemRequirements`).
- `SeuratData`, `Seurat` and `SeuratObject` are no longer suggested.
- Examples, tests and vignettes use at most two threads. Long tests run only
  when `NOT_CRAN=true`, and vignette code is evaluated only then.
- Added `inst/CITATION`, and `inst/COPYRIGHTS` for the bundled Hungarian
  algorithm, whose authors are listed in `Authors@R`.

# RcppML 1.0.0

## Breaking Changes

### Removed Functions
- `project()`: Use `predict()` for NMF objects, or `nnls()` for raw matrix projection
- `crossValidate()`: Use `nmf()` with `test_fraction` and a vector of ranks
- `lnmf()`: Discontinued

### Renamed Parameters
- `mask_zeros = TRUE` → `mask = "zeros"`

## New Features

### Statistical Distributions via IRLS
NMF now supports six distribution-appropriate loss functions via Iteratively Reweighted Least Squares:
- `"mse"` (Gaussian, default), `"gp"` (Generalized Poisson), `"nb"` (Negative Binomial), `"gamma"`, `"inverse_gaussian"`, `"tweedie"`
- Automatic distribution selection via `auto_nmf_distribution()`
- Zero-inflation models (`zi = "row"` or `zi = "col"`) for ZINB and ZIGP

### StreamPress I/O (`.spz` Format)
- Column-oriented binary format with 10–20× compression via rANS entropy coding
- Streaming NMF directly from `.spz` files for datasets larger than RAM
- `st_write()`, `st_read()`, `st_info()` for reading and writing `.spz` files
- Embedded obs/var metadata tables via `st_read_obs()` and `st_read_var()`
- GPU-direct reading with `st_read_gpu()` for zero-copy GPU NMF

### FactorNet Graph API
- `factor_net()` composes multi-modal, deep, and branching NMF pipelines
- Multi-modal NMF with shared embeddings across data modalities
- Deep NMF with hierarchical decomposition
- Conditional factorization by sample groups
- Cross-validation for graph architectures via `cross_validate_graph()`

### GPU Acceleration
- Optional CUDA backend via cuBLAS/cuSPARSE with automatic CPU fallback
- `gpu_available()` and `gpu_info()` for device queries
- Streaming GPU NMF for matrices exceeding VRAM
- Dense and sparse GPU NMF paths

### Cross-Validation Improvements
- Speckled holdout masks for principled rank selection
- `mask = "zeros"` for recommendation data (only non-zero entries held out)
- Automatic rank search with `k = "auto"`
- Multiple replicates via `cv_seed` vector
- Early stopping with configurable `patience`

### SVD and PCA
- `svd()` for truncated SVD with five methods: deflation, Krylov, Lanczos, IRLBA, randomized
- `pca()` convenience wrapper (centered and optionally scaled SVD)
- Constrained PCA: non-negative (`nonneg = TRUE`), sparse (`L1`), and combined
- `variance_explained()` for scree plots
- `predict()` for SVD objects (out-of-sample projection)

### Projective and Symmetric NMF
- `projective = TRUE`: H is computed as $W^T A$ instead of solved independently, producing more orthogonal factors
- `symmetric = TRUE`: For symmetric matrices, enforces $H = W^T$

### Enhanced NNLS Solver
- `nnls()` now supports `loss`, `L21`, and `angular` parameters (parity with `nmf()`)
- Non-MSE losses delegate through single-iteration NMF with IRLS
- `predict()` for NMF objects now uses the config (L1, L2, upper_bound) stored during fitting

### Additional Regularization
- L1 / L2 penalties with separate values for W and H
- L21 group sparsity for automatic factor selection
- Angular decorrelation penalty (replaces `ortho`)
- Upper bound constraints on factor values
- Graph Laplacian regularization for spatial/network smoothness

### New Datasets
- `golub`: Leukemia gene expression (38 × 5,000 sparse)
- `olivetti`: Olivetti face images (400 × 4,096 sparse)
- `digits`: Handwritten digit images (1,797 × 64 dense)
- `pbmc3k`: PBMC 3k scRNA-seq with cell type annotations (13,714 × 2,638, StreamPress compressed)

### Other
- Multiple random initializations with best-model selection via `seed = c(...)`
- Automatic NA detection and masking
- `consensus_nmf()` for robust factorizations across multiple random starts

## Enhancements
- S4 class system with backward-compatible `$` accessor (migrated from implicit S4)
- `on_iteration` callback for custom per-iteration logging
- Lanczos and IRLBA initialization for faster NMF convergence

## Internal Changes
- Unified NMF configuration into `NMFConfig` struct
- Pure C++ header library (Rcpp-free core algorithms in `inst/include/RcppML/`)
- `DataAccessor<MatrixType>` template for unified sparse/dense dispatch
- Consolidated header structure and eliminated code duplication
- OpenMP parallelization with proper reduction clauses
