## RcppML 1.0.1

### CRAN Compliance

- Reduced public API from 52 to 27 exports; internal helpers (`r_matrix`, `r_sparsematrix`, `r_sample`, `r_unif`, `r_binom`, `mse`, `solve`, `sp_compress`, `sp_decompress`, `sp_read_transpose`) are now `@keywords internal`
- Moved `Matrix` from `Depends` to `Imports`
- All `R CMD check --as-cran` issues resolved (0 ERRORs)

### Documentation

- Added comprehensive algorithm documentation in `docs/factornet/algorithms/` covering
  NMF solver loop, NNLS solvers, cross-validation, loss functions, regularization,
  initialization, scaling, SVD methods, and graph regularization
- Added C++ library documentation: `docs/factornet/README.md`, `ARCHITECTURE.md`,
  `API_REFERENCE.md`, `gpu/README.md`, `io/README.md`
- Full roxygen audit: added `@seealso`, `@return`, `@examples`, and `@param` to all
  exported functions and S3/S4 methods
- Added `@section Unsupported Combinations` to `nmf()`, `svd()`, and `factor_config()`
  documenting Cholesky+IRLS, ZI restrictions, and GPU fallback paths
- Vignette fixes: corrected `sparse` → `mask_zeros` parameter in cross-validation,
  NMF deep dive, and recommendation vignettes; fixed non-existent `precision` parameter
  in GPU vignette; fixed undefined variables in distribution API vignette
- Updated `_pkgdown.yml`: added SVD, Distribution Selection, and missing function
  entries to reference index
- Updated `DESCRIPTION` to reflect full feature set (SVD, distribution-based losses,
  Cholesky solver)
- API parameter consistency audit documented in `docs/dev/API_PARAM_AUDIT.md`

### New Features

- `gpu_available()` and `gpu_info()` are now exported and documented (#GPU)
- `nmf()` gains `streaming` and `panel_cols` parameters for out-of-core GPU NMF on matrices exceeding VRAM
- `predict()` S4 method for `svd_pca` objects supports new projection of test data onto existing left singular vectors
- Graph regularization parameters (`graph_W`, `graph_H`, `graph_lambda`) now clearly document Laplacian vs adjacency semantics; `RcppML` converts raw adjacency A to Laplacian L = D - A internally

### Bug Fixes

- Fixed GCC template deduction failure when `nullptr` passed for `const Scalar*` IRLS parameters (4 call sites in `fit_cpu.hpp`)
- Fixed examples using unexported internal functions (`r_sparsematrix`, `mse`)
- Fixed undocumented parameters (`dispersion`, `theta_init`) in `nmf()` documentation
- Fixed tests referencing removed parameters (`ortho`, `convergence_str`, `sparse`, `convergence`, `precision`)
- Removed duplicate `cv_irls_weight` helper (was shadowing `compute_irls_weight` in `fit_cpu.hpp`)
- Removed duplicate `detail::extract_scaling` (was shadowing `variant::extract_scaling` in `fit_cpu.hpp`)
- Replaced 9 deprecated function pairs (separate sparse/dense variants) with single unified templates, eliminating dead code paths

### Internal

- Unified NMF configuration into single `NMFConfig` struct (Phases 1-2)
- Internalized GPU dispatch from R `.C()` bridge to C++ gateway (Phase 3)
- Consolidated header directory structure: `algorithms/nmf/` + `gateway/` merged into `nmf/`, `algorithms/svd/` merged into `svd/` (Phases 5-7)
- Added `FitHistory` callback logging to all fit loops (Phase 8)
- Mechanical renames: `ortho` -> `angular`, `splitmix64` -> `rng` (Phase 4)
- SVD initialization auto-selection: Lanczos for k<32, IRLBA for k>=32
- **Unified CPU data access layer**: New `DataAccessor<MatrixType>` template (Phase 10) provides a single code path for sparse and dense matrix access, replacing ~40% duplicated sparse/dense update loops in the NMF solver
- **GPU streaming NMF**: Added `fit_gpu_streaming.cuh` for matrices too large for GPU VRAM; `fit_chunked_gpu.cuh` tiles the computation over `panel_cols`-wide column panels
- **CPU W-update cache optimisation**: Pre-computed W^T stored as a contiguous column-major buffer; avoids scattered column reads during the Gram matrix inner loop
- **`fit_cpu.hpp` restructured** (1,442 → 980 lines): Initialization helpers extracted to `nmf_init.hpp`, gram-trick and explicit loss functions to `explicit_loss.hpp`, masked-NNLS code to `masked_nnls.hpp`; all three are `#include`'d before the `RcppML::nmf` namespace
- **`fit_cv.hpp` restructured** (1,845 → 1,032 lines): CV mask generation, IRLS weight helpers, and fold bookkeeping split into focused translation units
- **`gpu_bridge.cu` split** (1,618 lines → 4 focused translation units): `gpu_bridge_common.cuh` (shared headers/helpers), `gpu_bridge_cluster.cu` (bipartition/dclust), `gpu_bridge_nmf.cu` (all NMF variants), `gpu_bridge_svd.cu` (SVD/PCA), `gpu_bridge_utils.cu` (profiler); `Makefile.gpu` updated accordingly
- Development documentation consolidated under `docs/dev/`
- New `tests/testthat/test_chunked_gpu.R`: 5 GPU-specific tests (basic validity, loss convergence, reproducibility, chunked vs non-chunked consistency, panel-size sweep); all skip gracefully on CPU-only nodes

## RcppML 0.5.0 (2022-04-01)

### Major changes

- Launch pkgdown site
- Added the `nmf` S3 class to the result of `nmf` function
- Introduce S3 methods for NMF (`[`, `align`, `biplot`, `dim`, `dimnames`, `head`, `mse`, `predict`, `print`, `prod`, `sort`, `sparsity`, `summary`, `t`)
- New plotting methods for NMF (`biplot.nmf`, `plot.nmfSummary`, `plot.nmfCrossValidation`)
- `mse` is now an S3 method for `nmf` objects
- `project` now handles only projections of `w`, for simplicity
- New vignette on `Getting Started with NMF`!

### Minor changes
- Support for specific sample and feature selections for NMF removed to increase performance on C++ end
- Removed `updateInPlace` advanced parameter for `nmf` because advantages were not convincing
- `mask_zeros` implementation is now specific to sparse matrices, multi-thread parallelization, and projections with transposition
- Added `cosine` function for fast cosine distance calculations
- Condensed and pared down documentation throughout. Advanced usage discussion will be moved to future vignettes.

## RcppML 0.5.1 (2022-09-15)

### Major changes
- three new datasets (`hawaiibirds`, `aml`, and `movielens`)
- Move NMF models and methods from S3 to S4 for stability
- Better random initializations (now using both `rnorm` and `runif` with multiple ranges/shapes, when multiple seeds are specified)
- added L2 regularization to NMF
- Support for masking values
- add `impute` and `perturb` methods to `crossValidate`

### Minor changes
- better random initializations (now using both `rnorm` and `runif` with multiple ranges/shapes)
- New vignette on random restarts
- better "head" and "show" methods
- return "w_init" with model

## RcppML 0.5.2 (2023-03-01)

### Major changes
- add linked NMF
- update all documentation

### Minor changes
- clean up C++ API
- C++ API gets meta-templating

## RcppML 1.0.0 (2026-02-27)

### Breaking Changes

**Major Parameter Refactoring** - The `nmf()` function has been significantly simplified and modernized. See `PARAMETER_CHANGES.md` for detailed migration guide.

#### Removed Parameters
- `reps`: Removed. Use vector of seeds for multiple runs: `seed = c(1, 2, 3)`
- `use_dense_mode`: Removed. Backend automatically selects optimal mode
- `cv_folds`: Removed. Use `cv_seed` vector instead for multiple CV replicates
- `cv_test_fraction`: Removed. Consolidated into `test_fraction`
- `cv_fraction`: Removed. Use `test_fraction` instead
- `cv_init`: Removed. Initialization always optimized
- `cv_tolerance`: Removed. CV always uses exact best rank
- `cv_max_k`: Removed. Use `cv_k_range = c(min, max)` instead
- `cv_k_init`: Removed. Use `cv_k_range = c(min, max)` instead
- `precision`: Removed. Use `fp32 = TRUE/FALSE` instead
- `sparse_mode`: Removed. Use `sparse = TRUE/FALSE` instead

#### Renamed Parameters
- `ortho` → `angular`: More accurate name for angular regularization/penalty
- `sparse_mode` → `sparse`: Simplified, more intuitive name
- `cv_fraction` → `test_fraction`: Clearer, consistent naming across CV functionality
- `precision` → `fp32`: Boolean flag more intuitive than string ("float"/"double")

#### Consolidated Parameters
- `cv_k_init` + `cv_max_k` → `cv_k_range = c(min, max)`: Single vector parameter
- `cv_folds` → `cv_seed` vector: Explicit seed control, length = number of replicates
- `cv_fraction` + `cv_test_fraction` → `test_fraction`: Single unified parameter

### New Features

#### Multiple Initializations
- **Vector of seeds**: Provide multiple random seeds to run models with different initializations and return the best one
  ```r
  # Run 5 models with different random initializations
  model <- nmf(data, k = 5, seed = c(123, 456, 789, 101, 202))
  ```
- **List of custom matrices**: Provide multiple custom initialization matrices to test different starting points
  ```r
  # Run models from 3 different custom initializations
  model <- nmf(data, k = 5, seed = list(init1, init2, init3))
  ```
- **Best model selection**: Automatically selects and returns the model with lowest loss across all initializations
- **Results tracking**: All initialization results stored in `model@misc$all_inits` dataframe

#### Rank Validation
- Automatic validation that custom initialization matrices match specified rank `k`
- Clear error messages when rank mismatch detected
- Works for both single matrix and list of matrices

#### Automatic NA Detection
- Automatically detects NA values in input data
- Provides informative message with count and percentage of NAs
- Auto-creates mask for missing values when detected
- Works with both dense and sparse matrices

### Minor Changes
- Improved error messages throughout parameter validation
- Better documentation with clearer examples
- Enhanced test coverage with 47 comprehensive parameter tests
- Reorganized helper functions for better maintainability
- **Removed deprecated parameters**: `lambda_W`, `lambda_H` (use `graph_lambda`), and `holdout_fraction` (use `cv_fraction`) have been removed.

### Backward Compatibility
Despite the S3→S4 class migration and breaking changes, **backward compatibility is maintained** for existing reverse dependencies:
- **S4 `$` accessor override**: Old code using `model$w` and `model$h` continues to work with new S4 objects. The `$` method transparently accesses S4 slots.
- **S4 `$<-` setter override**: Assignment using `model$w <- value` syntax remains functional.
- **Preserved function signatures**: Core functions `nmf()` and `nnls()` maintain their original parameter names and behaviors.
- **Maintained model structure**: The `w`, `d`, `h`, and `misc` components remain accessible via both `$` (backward compatible) and `@` (S4 native) accessors.
- **No changes to common use cases**: Standard workflows like `model <- nmf(A, k=5); w <- model$w` work unchanged.

This ensures that packages depending on RcppML (GeneNMF, phytoclass, CARDspa, scater, miloR, flashier) will continue to function without modification.

### Major Changes
- **Alternative loss functions**: NMF now supports multiple loss functions via the `loss` parameter:
  - `"mse"` (default): Mean Squared Error (Frobenius norm)
  - `"mae"`: Mean Absolute Error (L1 loss), robust to outliers
  - `"huber"`: Huber loss, blend of MSE and MAE with configurable `huber_delta`
  - `"kl"`: Kullback-Leibler divergence, suitable for count data
  Non-MSE losses use Iteratively Reweighted Least Squares (IRLS) for optimization.
- **Unified graph regularization**: New `graph_lambda = c(w, h)` parameter replaces separate `lambda_W` and `lambda_H` parameters. Accepts single value (applies to both) or vector for independent control.
- **Enhanced seed parameter**: The `seed` parameter now accepts three formats:
  - `"random"` (default): Random initialization
  - Integer: Sets random seed for reproducible initialization
  - Matrix: Custom W initialization matrix (dimensions p × k)
- **Flexible upper bounds**: `upper_bound` parameter now accepts `c(w, h)` vector for independent bounds on W and H matrices. Single value applies to both (backward compatible).
- **Improved CV control**: 
  - New `cv_folds` parameter for k-fold cross-validation (e.g., `cv_folds = 5` for 5-fold CV)
  - Renamed `holdout_fraction` to `cv_fraction` for clarity
  - Auto-enables `track_loss` when CV is active
  - CV disabled when `cv_folds = 0` and `cv_fraction = 0` (default)
- **Unified NMF class**: Merged `nmf_sparse` and `nmf_dense` into a single templated `nmf<Mat>` class using `constexpr if` for compile-time dispatch based on matrix type. This reduces code duplication by ~20% while maintaining full backward compatibility through type aliases.
- **Pure C++ header library**: All headers in `inst/include/RcppML/` are now completely Rcpp-free, using only Eigen and the C++ standard library. This makes the core algorithms portable and usable in non-R C++ projects.
- **Simplified sparse matrix handling**: Removed custom `RcppML::SparseMatrix` and `RcppML::MappedSparseMatrix` typedefs. Now uses `Eigen::SparseMatrix<double>` and `Eigen::Map<const Eigen::SparseMatrix<double>>` directly throughout the codebase.
- **Consolidated header structure**: Merged `RcppMLCommon.hpp` into `RcppML.h` as the single entry point for the C++ library.
- **Template consistency**: Unified template infrastructure across NMF, NNLS, bipartition, and clustering components for consistent precision handling. All core algorithms now support arbitrary scalar types via templates (currently using double precision).
- Better cross-validation, now exclusively using the mean squared error of missing value imputation (random speckled patterns of missing values)
- Complete migration to the S4 system, with backwards compatibility for CRAN version 0.5.0
- New vignettes and built-in datasets

### Code Cleanup for CRAN Submission
- **Removed profiling infrastructure**: Deleted all `ScopedTimer` instrumentation from `nmf.hpp`, `nnls.hpp`, and `baseline_nmf.cpp` (~250 lines of profiling code removed)
- **Removed development artifacts**: Deleted 25+ benchmark and test files from root directory (`benchmark_*.R`, `test_*.R`, verification scripts)
- **Removed unused directory**: Deleted `inst/benchmarks/` directory containing development benchmarking code
- **Cleaned template implementations**: Updated `bipartition.hpp` and `cluster.hpp` to use modern templated CD solvers (`cd_solve2`, `cd_solve2_inplace`) instead of deprecated `nnls2` functions

### New Files
- `inst/include/RcppML/loss.hpp`: Loss function configuration, IRLS weight computation, and helper functions for Huber and KL divergence losses.
- `NEWS.md`: Complete release notes for version 1.0.0

### Removed Dead Code
- Deleted `RcppEigen_bits.h` (redundant with RcppEigen)
- Deleted `distance.hpp` (functions were never called)
- Deleted `SparseMatrix.h` and `MappedSparseMatrix.h` (unused helper classes)
- Removed unexported `align_models()` function
- Removed all profiling-related functions and infrastructure

### Code Organization
- Consolidated all Rcpp bindings into `src/RcppFunctions.cpp` (merged `bipartiteMatch.cpp`)
- Consolidated dataset documentation (`aml`, `hawaiibirds`, `movielens`) into single `R/data.R` file
- Added `inline` specifiers to header functions to prevent multiple definition errors
- **Type traits for sparse/dense dispatch**: New `detail::is_sparse_matrix<T>` trait enables compile-time detection of sparse vs dense matrices for template specialization.

### Documentation Improvements
- Fixed duplicate `plot.nmfCrossValidate` alias (removed from `nmf_plots.R`)
- Regenerated all documentation with roxygen2
- All 154 tests passing
- Updated DESCRIPTION file with current date

### Build System
- Added optimization flags to Makevars: `-O3`, `-mtune=generic`, `-funroll-loops`
- Added diagnostic suppression: `-Wno-ignored-attributes`, `-Wno-deprecated-declarations`
- Added Eigen optimization defines: `EIGEN_INITIALIZE_MATRICES_BY_ZERO`, `EIGEN_NO_DEBUG`

### Bug Fixes
- Fixed deprecated `@docType package` roxygen warning
- Fixed circular include dependencies in header files
- Compatibility with latest version of the `Matrix` package
- Fixed namespace issues with templated `cluster` struct in `cluster.hpp`
- **Fixed `align()` method**: Was using nonexistent `$pairs` field from `bipartiteMatch()`. Now correctly uses `$assignment + 1L` (0-indexed to 1-indexed conversion)
- **Fixed `align()` dimension check**: Changed `all(dim(a) != dim(b))` to `any(dim(a) != dim(b))` so rank mismatches are properly detected
- **Fixed `bipartiteMatch()` documentation**: Return value documented as `$assignment` (0-indexed) instead of incorrect `$pairs`
- **Fixed `dclust()` documentation**: Return value corrected — removed nonexistent `$leaf` and `$dist` fields, added `$size`. Documented that `$samples` is 0-indexed and `$id` is numeric

### Documentation Improvements
- All exported functions now have `@return` tags and `\value{}` sections in .Rd files
- All exported function .Rd files now have `\examples{}` sections
- Organized pkgdown reference page into logical categories
