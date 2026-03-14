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
- `mask_zeros` parameter for recommendation data (only non-zero entries held out)
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
