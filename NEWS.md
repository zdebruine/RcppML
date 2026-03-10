# RcppML 1.0.0

## Breaking Changes

### Removed Deprecated Functions
- `solve()`: Use `nnls()` instead
- `project()`: Use `predict()` for NMF objects, or `nnls()` for raw matrix projection
- `crossValidate()`: Use `nmf()` with `test_fraction` and a vector of ranks
- `sp_write()`, `sp_read()`, `sp_info()`, `sp_write_dense()`, `sp_read_dense()`: Use `st_write()`, `st_read()`, `st_info()`, `st_write_dense()`, `st_read_dense()` instead
- `sp_read_gpu()`, `sp_free_gpu()`: Use `st_read_gpu()`, `st_free_gpu()` instead
- `svd_pca()`, `sparse_pca()`, `nn_pca()`: Use `svd()` with appropriate parameters

### Renamed Classes
- The `svd_pca` S4 class is now `svd`. All methods (`predict`, `reconstruct`, `variance_explained`, `[`, `head`, `show`, `dim`) updated accordingly.

### Renamed Datasets
- `digits_full` is now `digits`

### Renamed Parameters
- `ortho` → `angular` (decorrelation penalty)
- `precision` → `fp32` (boolean flag)
- `cv_fraction` → `test_fraction`

### Removed Parameters
- `reps`: Use vector of seeds for multiple runs: `seed = c(1, 2, 3)`
- `use_dense_mode`, `cv_folds`, `cv_test_fraction`, `cv_fraction`, `cv_init`, `cv_tolerance`, `cv_max_k`, `cv_k_init`, `sparse_mode`

## New Features

### Statistical Distributions via IRLS
NMF now supports six distribution-appropriate loss functions via Iteratively Reweighted Least Squares:
- `"mse"` (Gaussian, default), `"gp"` (Generalized Poisson), `"nb"` (Negative Binomial), `"gamma"`, `"inverse_gaussian"`, `"tweedie"`
- Automatic distribution selection via `auto_nmf_distribution()`
- Zero-inflation models (`zi = "row"` or `zi = "col"`) for ZINB and ZIGP

### StreamPress I/O (`.spz` Format)
- Renamed from SparsePress to StreamPress throughout the package
- Column-oriented binary format with 10–20× compression via rANS entropy coding
- Streaming NMF directly from `.spz` files for datasets larger than RAM
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
- Automatic rank search with `k = "auto"`
- Multiple replicates via `cv_seed` vector
- Early stopping with configurable patience

### Enhanced NNLS Solver
- `nnls()` now supports `loss`, `L21`, and `angular` parameters (parity with `nmf()`)
- Non-MSE losses delegate through single-iteration NMF with IRLS
- `predict()` for NMF objects now uses the config (L1, L2, upper_bound) stored during fitting

### Additional Regularization
- L21 group sparsity for automatic factor selection
- Angular decorrelation penalty
- Graph Laplacian regularization for spatial/network smoothness
- Huber-type robust NMF (less sensitive to outliers)

### SVD/PCA Enhancements
- Five SVD methods: deflation, Krylov, Lanczos, IRLBA, randomized
- Constrained PCA: non-negative, sparse, and combined
- Auto-rank selection for SVD
- `predict()` for SVD objects (out-of-sample projection)

### Other
- Multiple random initializations with best-model selection via `seed = c(...)` 
- Automatic NA detection and masking
- S4 class system (migrated from S3) with backward-compatible `$` accessor
- Consensus clustering via `consensus_nmf()`
- Divisive clustering via `dclust()` and `bipartition()`

## Internal Changes
- Unified NMF configuration into `NMFConfig` struct
- Pure C++ header library (Rcpp-free core algorithms in `inst/include/RcppML/`)
- `DataAccessor<MatrixType>` template for unified sparse/dense dispatch
- Consolidated header structure and eliminated code duplication
- OpenMP parallelization with proper reduction clauses
- Optimization flags: `-O3`, `-mtune=generic`, `-funroll-loops`
