# P6: Constrained SVD — A Unified Framework for Regularized Truncated SVD

**Target Venue**: Computational Statistics / JCGS  
**Type**: Methodology + software paper  
**Estimated Length**: 20–25 pages  

---

## Abstract (Draft)

We present a unified framework for truncated singular value decomposition with
per-component constraints and regularization: L1 (lasso), L2 (ridge),
non-negativity, upper bounds, L2,1 (group sparsity), angular (orthogonality),
and graph Laplacian smoothing. Our implementation provides five algorithm
backends (deflation ALS, Krylov-seeded projected refinement, Lanczos
bidiagonalization, implicitly restarted Lanczos, and randomized SVD) with
automatic method selection based on constraint requirements and matrix
properties. We introduce two novel contributions: (1) a deflation-corrected
constrained ALS algorithm that maintains orthogonality despite proximal
projection steps, and (2) a Krylov-seeded projected refinement (KSPR) method
that combines Lanczos initialization with Gram-level regularization for
simultaneous multi-factor extraction. The framework includes automatic rank
selection via speckled holdout cross-validation, robust Huber-loss
downweighting, and optional GPU acceleration. Implementations are available as
C++ header-only templates callable from R, achieving 2–10× speedup over
existing R packages (irlba, rsvd) for constrained problems.

---

## Key Contributions

1. **Deflation-corrected constrained ALS**: Rank-1 ALS with proximal operators
   for L1/nonneg/bounds, followed by deflation correction that accounts for
   constraint-induced non-orthogonality bias
2. **KSPR algorithm**: Lanczos seed provides near-optimal initialization;
   Gram-solve-then-project refinement incorporates L2, L2,1, angular, and
   graph regularization at the Gram matrix level
3. **Auto-rank selection**: Speckled holdout CV with patience-based early
   stopping, interleaved with factor computation (no refit needed)
4. **Unified constraint API**: Single function handles all 5 methods with
   automatic dispatch based on constraint requirements
5. **Robust SVD**: Huber-loss IRLS reweighting within deflation ALS for
   outlier-robust factor extraction

---

## Algorithm Catalog

### Method 1: Deflation ALS (`method = "deflation"`)

**Algorithm**: Sequential rank-1 extraction via alternating least squares.

For factor $j = 1, \ldots, k$:
1. Initialize $u_j, v_j$ from residual right/left singular vector
2. Repeat until convergence:
   - $\tilde{v}_j \leftarrow A^T u_j / (u_j^T u_j)$
   - $v_j \leftarrow \text{prox}(\tilde{v}_j)$ (L1, nonneg, upper bound)
   - $\tilde{u}_j \leftarrow A v_j / (v_j^T v_j)$
   - $u_j \leftarrow \text{prox}(\tilde{u}_j)$
3. $d_j \leftarrow \|u_j\|$; normalize $u_j$
4. **Deflation**: $A \leftarrow A - d_j u_j v_j^T$

**Deflation correction**: Standard deflation assumes exact orthogonality. When
constraints (nonneg, L1) break this, we apply a correction step:
$u_j \leftarrow u_j - \sum_{i<j} (u_i^T u_j) u_i$ after proximal projection.

**Complexity**: $O(k \cdot \text{nnz} \cdot \text{maxit})$ for sparse; $O(k \cdot mn \cdot \text{maxit})$ for dense.

**Supports**: All constraints + robust + CV + auto-rank.

### Method 2: Krylov-Seeded Projected Refinement (`method = "krylov"`)

**Algorithm**: Block method extracting all $k$ factors simultaneously.

1. **Lanczos seed**: Run Lanczos bidiagonalization for $p$ steps ($p = k + 5$)
   to get approximate $U_0, V_0$
2. **Gram computation**: $G = A^T A$ (or its sparse approximation)
3. **Gram regularization**: $\tilde{G} = G + \lambda_2 I + \lambda_{21} D + \lambda_\alpha P + \lambda_g L_V$
   where $D$ is L2,1 diagonal, $P$ is angular penalty, $L_V$ is graph Laplacian
4. **Projected refinement**: Iterative solve-then-project:
   - $V \leftarrow \tilde{G}^{-1} G V$ (solve with regularized Gram)
   - $V \leftarrow \text{QR}(V)$ (re-orthogonalize)
   - Apply element-wise proximal operators to each column
5. Recover $U = A V D^{-1}$

**Complexity**: $O(\text{nnz} \cdot k + k^2 n)$ per iteration (Gram precomputed).

**Supports**: L2, L2,1, angular, graph regularization. L1/nonneg via post-projection.

### Method 3: Lanczos Bidiagonalization (`method = "lanczos"`)

**Algorithm**: Standard Lanczos bidiagonalization with full reorthogonalization.
Converts the SVD of $A$ to the eigendecomposition of the bidiagonal matrix $B$.

**Complexity**: $O(\text{nnz} \cdot k)$ for sparse; $O(mn \cdot k)$ for dense.

**Supports**: Unconstrained only. No regularization.

### Method 4: Implicitly Restarted Lanczos (`method = "irlba"`)

**Algorithm**: Augmented Lanczos bidiagonalization with implicit restarts.
Better numerical stability than plain Lanczos for larger $k$.

**Complexity**: $O(\text{nnz} \cdot k \cdot \text{restarts})$.

**Supports**: Unconstrained only. No regularization.

### Method 5: Randomized SVD (`method = "randomized"`)

**Algorithm**: Random projection → QR → project → SVD of small matrix.
With $q$ power iterations for improved accuracy.

**Complexity**: $O(\text{nnz} \cdot (k + p) \cdot (q + 1))$ where $p$ is oversampling.

**Supports**: Unconstrained only. No regularization.

---

## Constraint & Regularization Framework

### Element-Level Proximal Operators (Deflation + KSPR)

| Constraint | Proximal Operator | Effect |
|------------|-------------------|--------|
| L1 ($\lambda_1$) | $\text{sign}(x) \max(\|x\| - \lambda_1, 0)$ | Sparsity |
| L2 ($\lambda_2$) | $x / (1 + \lambda_2)$ | Shrinkage |
| Non-negativity | $\max(x, 0)$ | Non-negative factors |
| Upper bound ($b$) | $\min(\max(x, 0), b)$ | Bounded factors |

### Gram-Level Regularization (KSPR Only)

| Regularization | Gram Modification | Effect |
|----------------|-------------------|--------|
| L2 ($\lambda_2$) | $G + \lambda_2 I$ | Ridge shrinkage |
| L2,1 ($\lambda_{21}$) | $G + \lambda_{21} \text{diag}(1/\|v_j\|)$ | Group sparsity |
| Angular ($\lambda_\alpha$) | $G + \lambda_\alpha (V^T V - I)$ | Decorrelation |
| Graph ($\lambda_g, L$) | $G + \lambda_g L$ | Smooth along graph |

### Automatic Method Selection (`method = "auto"`)

```
IF robust: → deflation
ELIF any(L1, nonneg, upper_bound, L21, angular, graph): → krylov (or deflation if k small)
ELIF k < 64 and nnz/mn > 0.5: → randomized
ELIF k < 64: → lanczos
ELSE: → irlba
```

GPU auto-selection uses lower thresholds (Lanczos k<32, Randomized 32≤k<64, IRLBA k≥64).

---

## Auto-Rank Selection

### Algorithm

For `k = "auto"`, integrated cross-validation:

1. Create speckled holdout mask $\Omega$ (fraction `test_fraction`, default 5%)
2. For $j = 1, 2, \ldots, k_{\max}$:
   a. Extract factor $j$ using training entries only
   b. Compute test MSE: $\text{MSE}_j = \frac{1}{|\Omega|} \sum_{(i,l) \in \Omega} (A_{il} - \hat{A}^{(j)}_{il})^2$
   c. If $\text{MSE}_j > \text{MSE}_{\text{best}}$ for `patience` consecutive factors → stop
3. Return model truncated to rank $j^*$

### Deflation Integration

Auto-rank integrates naturally with deflation: each factor is extracted
sequentially, and test MSE is evaluated incrementally. No refitting needed.

### Krylov Integration

For the Krylov method, auto-rank evaluates test MSE after convergence of the
full $k_{\max}$-factor model, then truncates. This is less efficient than
deflation but still avoids multiple complete refits.

---

## Robust SVD via Huber Loss

### Huber Loss

$$L_\delta(r) = \begin{cases} \frac{1}{2} r^2 & \text{if } |r| \leq \delta \\ \delta |r| - \frac{1}{2} \delta^2 & \text{if } |r| > \delta \end{cases}$$

### IRLS Integration

Within each ALS iteration of deflation:
1. Compute residuals $r_{il} = A_{il} - d_j u_{ij} v_{lj}$
2. Compute weights $w_{il} = \min(1, \delta / |r_{il}|)$
3. Solve weighted least squares: $\tilde{v}_j = (U^T W U)^{-1} U^T W a_l$

Converges to the Huber M-estimator of the rank-1 approximation.

---

## Benchmark Design

### Benchmark 1: Unconstrained Speed (vs irlba, rsvd, svds)
- Data: MovieLens 10M (sparse), Olivetti faces (dense), random sparse/dense
- Methods: All 5 RcppML backends vs `irlba::irlba()`, `rsvd::rsvd()`, `RSpectra::svds()`
- Vary: k ∈ {5, 10, 20, 50, 100}, matrix dimensions
- Metric: Wall time, relative error $\|A - U_k D_k V_k^T\|_F / \|A\|_F$

### Benchmark 2: Constrained Problems (vs custom implementations)
- Tasks: Sparse PCA (L1), non-negative SVD (nonneg), graph-smooth PCA
- Compare: RcppML deflation/krylov vs `sparsepca`, `nsprcomp`, manual ADMM
- Vary: L1 ∈ {0.01, 0.1, 0.5, 1.0}, k ∈ {5, 10, 20}
- Metric: Reconstruction, sparsity (% zero loadings), computation time

### Benchmark 3: Auto-Rank Accuracy
- Data: Known-rank synthetic matrices with noise (SNR sweep)
- Ground truth: $A = U_r D_r V_r^T + \sigma E$, $r \in \{5, 10, 20\}$
- Compare: Auto-rank vs scree plot (manual), CV-based rank selection
- Metric: Rank recovery accuracy, MSE of selected rank vs optimal rank

### Benchmark 4: Robust SVD (vs PCPP, robustPCA)
- Data: Synthetic with outlier contamination (5%, 10%, 20%)
- Compare: Huber-robust deflation vs standard SVD vs PCPP vs robustPCA
- Metric: Subspace angle to true factors, breakdown point

### Benchmark 5: GPU Acceleration
- Data: Large sparse matrices (100K × 50K, 1M+ nonzeros)
- Compare: CPU vs GPU for each applicable method
- Vary: Matrix size, density, k
- Metric: Speedup ratio, numerical equivalence

---

## Figure List

1. **Figure 1**: Algorithm decision tree (method = "auto" dispatch logic)
2. **Figure 2**: Unconstrained speed benchmark (log-log time vs matrix size)
3. **Figure 3**: Relative error vs computation time (Pareto frontiers)
4. **Figure 4**: Sparse PCA loadings visualization (L1 sweep)
5. **Figure 5**: Non-negative SVD factors vs standard SVD (face dataset)
6. **Figure 6**: Auto-rank: estimated vs true rank across SNR levels
7. **Figure 7**: Auto-rank: test MSE curves with patience-based stopping
8. **Figure 8**: Robust SVD: subspace angle vs contamination fraction
9. **Figure 9**: GPU speedup curves (speedup ratio vs matrix size)
10. **Table 1**: Method × constraint support matrix
11. **Table 2**: Full benchmark results summary

---

## Paper Outline

### 1. Introduction (2 pages)
- Truncated SVD as the workhorse of dimensionality reduction
- Gap: existing R packages lack constraint support or are slow
- Contribution: unified framework with 5 methods + full constraint catalog

### 2. Background (2 pages)
- Review of truncated SVD algorithms
- Constrained matrix factorization landscape
- Proximal operators and ADMM for structured problems

### 3. Deflation ALS with Constraints (3 pages)
- Rank-1 ALS derivation
- Proximal projection integration
- Deflation correction for non-orthogonality
- Per-component constraint specification

### 4. Krylov-Seeded Projected Refinement (3 pages)
- Lanczos seed initialization
- Gram-level regularization framework
- Solve-then-project iterations
- Convergence guarantees

### 5. Auto-Rank via Cross-Validation (2 pages)
- Speckled holdout design
- Integration with deflation (incremental) and Krylov (post-hoc)
- Patience-based early stopping

### 6. Robust SVD (2 pages)
- Huber loss M-estimation
- IRLS integration within ALS
- Choice of delta parameter

### 7. Implementation (2 pages)
- C++ template design (header-only, CRTP dispatch)
- Sparse/dense template specialization
- GPU backend via CUDA
- R API design

### 8. Experiments (4 pages)
- Benchmarks 1–5
- Discussion of results

### 9. Conclusion (1 page)
- Summary of contributions
- Future work: tensor SVD, online/streaming SVD, adaptive delta

---

## Reproducibility

- Source code: `R/svd.R`, `R/svd_methods.R`, `inst/include/RcppML/svd/`
- Convenience wrappers: `pca()`, `sparse_pca()`, `nn_pca()`
- Datasets: Built-in (`olivetti`, `movielens`) + synthetic generators
- All benchmarks reproducible via `benchmarks/R/` scripts
