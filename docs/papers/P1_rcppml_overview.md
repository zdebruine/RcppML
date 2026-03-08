# P1: RcppML — High-Performance Non-Negative Matrix Factorization for R

**Target Venue**: Journal of Statistical Software (JSS) or The R Journal  
**Type**: Software paper  
**Estimated Length**: 25–30 pages  

---

## Abstract (Draft)

We present RcppML, an R package for scalable non-negative matrix factorization
(NMF) and related methods. RcppML provides a unified interface for standard NMF
with six exponential-family loss functions (MSE, Generalized Poisson, Negative
Binomial, Gamma, Inverse Gaussian, Tweedie), zero-inflation, cross-validation
rank selection, L1/L2/L2,1/angular/graph regularization, guided NMF, out-of-core
streaming factorization from compressed sparse files, GPU acceleration via CUDA,
constrained SVD/PCA, divisive biclustering, and a composable graph DSL for
multi-modal factorization networks. The C++ backend uses Eigen, OpenMP, and
template metaprogramming to dispatch optimized code paths for sparse/dense
matrices, CPU/GPU backends, and distribution-specific IRLS solvers. On
standard benchmarks, RcppML achieves 5–50× speedup over existing R packages
(NMF, NNLM, singlet) with lower memory footprint.

---

## Key Contributions (5)

1. **Unified NMF API**: Single `nmf()` function dispatches to 200+ backend
   configurations (sparse/dense × CPU/GPU × 6 distributions × CV/standard ×
   streaming/in-memory) with automatic fallback.

2. **IRLS framework for exponential-family NMF**: Iteratively Reweighted Least
   Squares converts distribution-specific likelihoods into weighted NNLS
   subproblems. Supports GP, NB, Gamma, InvGauss, Tweedie with per-row
   dispersion estimation.

3. **Streaming out-of-core NMF**: Column-chunked streaming from SparsePress
   (.spz) compressed files enables factorization of matrices larger than RAM
   with bounded O(chunk_size × k) memory.

4. **Cross-validation with lazy Gram correction**: Speckled holdout mask with
   per-column Gram matrix correction — no materialization of the full mask
   matrix. GPU-accelerated variant achieves near-zero CV overhead.

5. **Composable FactorNet graph DSL**: R-level graph API for building
   multi-layer, multi-modal factorization networks with shared factors,
   conditioning, and cross-layer guides.

---

## Related Packages Comparison

| Feature | RcppML | NMF (R) | NNLM | singlet | scikit-learn |
|---------|--------|---------|------|---------|--------------|
| Distributions | 6 + ZI | MSE only | MSE, KL | MSE | MSE, KL |
| Cross-validation | Built-in speckled | Manual | No | Built-in | Manual |
| GPU | CUDA | No | No | No | No |
| Streaming | SPZ chunks | No | No | No | Mini-batch |
| Regularization | L1, L2, L2,1, angular, graph | L1 | L1, L2 | No | L1 |
| Guided NMF | Classifier, external, callback | No | No | No | No |
| Multi-modal | FactorNet graph DSL | No | No | No | No |
| SVD variants | 5 methods | No | No | No | Randomized |
| Divisive clustering | dclust | No | No | No | No |
| NNLS solvers | CD + Cholesky auto | Lee-Seung MU | CD | Lee-Seung MU | CD |

---

## Benchmark Design

### Benchmark 1: Time vs. Rank (k)
- Datasets: simulated sparse (m=10000, n=5000, density=0.1), AML (dense 824×135)
- Ranks: k ∈ {2, 4, 8, 16, 32, 64, 128}
- Packages: RcppML, NMF, NNLM, singlet, sklearn (via reticulate)
- Metric: Wall time (seconds), 50 iterations each, tol=1e-10

### Benchmark 2: Time vs. Matrix Size (n)
- Fixed k=20, varying n ∈ {1000, 5000, 10000, 50000, 100000}
- Sparse matrices, density=0.05
- Same packages

### Benchmark 3: Memory Scaling
- Peak RSS via `/proc/self/status` VmPeak
- Same configurations as Benchmark 1

### Benchmark 4: Convergence Quality
- Known ground truth (simulated W, H with noise)
- Recovery error: ||W_true - W_est||_F / ||W_true||_F after alignment
- Compare distribution selection (MSE vs GP vs NB) on count data

---

## Figure List

1. **Figure 1**: Package architecture diagram (R → Rcpp bridge → template dispatch → CPU/GPU backends)
2. **Figure 2**: Wall time vs. rank across packages (log-log plot)
3. **Figure 3**: Wall time vs. matrix size (scaling curves)
4. **Figure 4**: Memory usage comparison
5. **Figure 5**: Convergence curves (loss vs. iteration) for different distributions on count data
6. **Figure 6**: Cross-validation curves showing rank selection
7. **Figure 7**: FactorNet graph topology example (multi-modal integration)
8. **Table 1**: API feature comparison (the table above)
9. **Table 2**: Supported distribution parameters

---

## Paper Outline

### 1. Introduction (2 pages)
- NMF background and applications (genomics, topic modeling, image analysis)
- Limitations of existing R packages
- Overview of RcppML contributions

### 2. Software Architecture (3 pages)
- Template-driven C++ backend
- R → Rcpp → Eigen dispatch chain
- Sparse/dense matrix abstraction
- Resource detection and GPU fallback

### 3. NMF Algorithm (4 pages)
- ALS update rules
- NNLS solvers: coordinate descent and Cholesky with clipping
- Gram trick for efficient loss computation
- Normalization and diagonal scaling

### 4. Statistical Distributions (3 pages)
- IRLS framework overview
- Per-distribution weight formulas
- Dispersion estimation (method of moments)
- Zero-inflation E-step

### 5. Cross-Validation (2 pages)
- Speckled holdout mask generation
- Lazy per-column Gram correction
- GPU-accelerated CV

### 6. Regularization and Constraints (2 pages)
- L1, L2, L2,1, angular penalty implementation
- Graph Laplacian regularization
- Guided NMF (classifier, external, callback, reference)

### 7. Streaming and Scalability (2 pages)
- SparsePress file format overview
- Column-chunked streaming NMF algorithm
- Memory-bounded processing

### 8. FactorNet Graph DSL (2 pages)
- Node types and graph compilation
- Multi-modal shared factorization
- Deep factorization chains

### 9. Additional Methods (2 pages)
- Constrained SVD/PCA (5 algorithm backends)
- Divisive biclustering (dclust)
- Standalone NNLS solver

### 10. Benchmarks (4 pages)
- Experimental setup
- Results for Benchmarks 1–4
- Discussion of scaling behavior

### 11. Application: Single-Cell Genomics (2 pages)
- PBMC 3k analysis walkthrough
- Distribution selection with auto_nmf_distribution
- Comparison of MSE vs. NB on count data

### 12. Conclusion (1 page)
- Summary of contributions
- Future work (dense streaming, additional GPU kernels)

---

## Reproducibility

All benchmarks will use:
- `benchmarks/harness/` infrastructure
- Pinned package versions via `renv`
- Fixed seeds for all stochastic operations
- Docker container with CUDA toolkit for GPU benchmarks
