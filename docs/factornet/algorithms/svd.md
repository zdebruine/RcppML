# Singular Value Decomposition (SVD) Methods

## Overview

FactorNet provides five SVD methods, ranging from simple unconstrained truncated SVD to regularized constrained SVD with auto-rank selection. The choice depends on the rank $k$, whether constraints are needed, and the compute backend.

### Model

Given $A \in \mathbb{R}^{m \times n}$, compute the truncated rank-$k$ SVD:

$$A \approx U \Sigma V^T$$

where $U \in \mathbb{R}^{m \times k}$ (left singular vectors), $\Sigma = \text{diag}(\sigma_1, \ldots, \sigma_k)$ (singular values), and $V \in \mathbb{R}^{n \times k}$ (right singular vectors).

With constraints (non-negativity, sparsity), the factorization becomes a constrained approximation that may not be a true SVD but optimizes the regularized objective.

---

## Method Summary

| Method | Algorithm | Constraints | Best for | Complexity |
|---|---|---|---|---|
| **Deflation** | Sequential rank-1 ALS | Full (L1, L2, nonneg, L21, angular, graph) | Small $k < 8$, constrained | $O(\text{nnz} \cdot k^2)$ |
| **Krylov** | Lanczos seed + Gram refinement | Gram-level (L2, L21, angular, graph) | Medium $k \geq 8$, constrained | $O(\text{nnz} \cdot k)$ per iter |
| **Lanczos** | Golub-Kahan bidiagonalization | None | $k < 64$, unconstrained | $O(\text{nnz} \cdot k + (m+n) k^2)$ |
| **IRLBA** | Implicitly restarted Lanczos | None | $k \geq 64$, unconstrained | Similar to Lanczos, better restart |
| **Randomized** | Power iteration + sketch | None | Large $k$, GPU, fixed cost | $O(\text{nnz} \cdot 8k)$ |

---

## 1. Deflation — Sequential Rank-1 ALS

### Algorithm

Each factor is computed sequentially by solving rank-1 ALS on the deflated residual:

```
for f = 1 to k:
    R = A - Σ_{i<f} σ_i u_i v_i^T        // deflated residual
    repeat until convergence:
        v = R^T u / (u^T u)               // update v
        apply constraints to v             // L1, nonneg, bounds
        u = R v / (v^T v)                  // update u
        apply constraints to u
    σ_f = ||u|| · ||v||                    // extract singular value
    normalize u, v to unit vectors
```

### Constraint Application (Per-Element)

After each unconstrained least-squares solve:

- **L1 (Lasso)**: Soft-thresholding: $v_j \leftarrow \text{sign}(v_j) \max(0, |v_j| - \lambda / (2\|u\|^2))$
- **L2 (Ridge)**: Shrinkage: $v_j \leftarrow v_j / (1 + \lambda / \|u\|^2)$
- **Non-negativity**: Clipping: $v_j \leftarrow \max(0, v_j)$
- **Upper bound**: Box constraint: $v_j \leftarrow \min(\text{bound}, v_j)$

### When to Use

- Small rank ($k < 8$) where sequential overhead is acceptable
- Full per-element constraint support needed (L1, non-negativity, bounds)
- Auto-rank selection with cross-validation
- Robust SVD with IRLS (outlier downweighting)

### Complexity

- Per factor: $O(\text{nnz} \cdot k)$ for SpMV + constraint application
- Total: $O(\text{nnz} \cdot k^2)$ due to deflation (each subsequent factor operates on residual)

---

## 2. Krylov-Seeded Projected Refinement (KSPR)

### Algorithm

A two-phase hybrid method:

**Phase 1 — Lanczos seed**: Run unconstrained Lanczos bidiagonalization to identify the $k$-dimensional subspace in $O((\text{k+p}) \cdot \text{nnz})$ time.

**Phase 2 — Iterative refinement**:
```
repeat until convergence:
    W = A · V                        // SpMM, O(nnz · k)
    H = A^T · U                      // SpMM, O(nnz · k)
    G = W^T W (or H^T H)            // Gram, O(k² · m)
    G += L2·I + angular·C + graph·L  // Gram-level regularization
    Solve G^{-1} via Cholesky
    Project constraints (nonneg, bounds)
    Orthonormalize columns, extract singular values into d
```

**Early termination**: Falls back to pure Lanczos if no constraints are active.

### When to Use

- Medium rank ($k \geq 8$) where block methods outperform sequential deflation
- Gram-level constraints (L2, L21, angular, graph Laplacian)
- GPU-friendly (uses block SpMM, cuSOLVER Cholesky)

### Complexity

- Phase 1: $O((k+p) \cdot \text{nnz})$
- Phase 2 per iteration: $O(2 \cdot \text{nnz} \cdot k + k^3 + k^2(m+n))$
- Typically converges in 3–5 iterations

---

## 3. Lanczos Bidiagonalization

### Algorithm

Golub-Kahan bidiagonalization builds orthonormal bases $P$ ($n \times j$) and $Q$ ($m \times j$) via orthogonalized sparse matrix-vector products, forming an upper bidiagonal matrix $B$ ($j \times j$) whose singular values approximate those of $A$.

```
p_1 = random unit vector
for step = 1 to j:
    q_step = A · p_step - β_{step-1} · q_{step-1}
    reorthogonalize q against all previous q's
    α_step = ||q_step||; q_step /= α_step
    p_{step+1} = A^T · q_step - α_step · p_step
    reorthogonalize p against all previous p's
    β_step = ||p_{step+1}||; p_{step+1} /= β_step

SVD(B) → Ritz values approximate σ_1, ..., σ_k
U = Q · U_B, V = P · V_B
```

Reorthogonalization uses Classical Gram-Schmidt twice per step for numerical stability.

### When to Use

- **Default for unconstrained SVD** with $k < 64$
- CPU and GPU
- Dense or sparse matrices
- Fast, stable baseline

### Complexity

- Per step: 2 SpMVs $O(\text{nnz})$ + reorthogonalization $O((m+n) \cdot j)$
- Total ($j \approx 3k$): $O(\text{nnz} \cdot k + (m+n) \cdot k^2)$

---

## 4. IRLBA — Implicitly Restarted Lanczos

### Algorithm

Based on Baglama & Reichel (2005). Runs short Lanczos passes (typically `work = k + 7` steps), then **implicitly restarts** by compressing the Krylov subspace to keep only the top $k$ Ritz vectors plus a residual:

```
repeat until convergence:
    Run Lanczos for (work - k) steps from previous restart
    Compute Ritz values from bidiagonal matrix
    Implicit restart: compress to top k + residual
    Check convergence via Ritz residual norms
```

### When to Use

- **Large rank** ($k \geq 64$) where full Lanczos basis is memory-expensive
- Memory-constrained environments
- Unconstrained SVD only

### Complexity

- Per restart: $O(\text{nnz} \cdot \text{work})$ SpMVs + $O((m+n) \cdot \text{work}^2)$ reorthogonalization
- Number of restarts: adaptive (typically 2–5 for well-separated singular values)

---

## 5. Randomized SVD

### Algorithm

Halko-Martinsson-Tropp (2011) randomized range finder with power iteration:

```
l = k + oversampling (default: max(10, k/5))
Ω = random Gaussian matrix (n × l)

// Power iteration (q iterations, default q=3)
for i = 1 to q:
    Y = A · Ω               // SpMM
    QR(Y) → Q₁
    Z = A^T · Q₁            // SpMM
    QR(Z) → Q₂
    Ω = Q₂

Q = QR(A · Ω)               // final range approximation
B = Q^T · A                  // core matrix (l × n)
SVD(B) → top k components
U = Q · U_B
```

### When to Use

- Very large matrices (millions of rows/columns)
- GPU with $32 \leq k < 64$ (cuSPARSE SpMM batches efficiently)
- Fixed computation budget preferred over adaptive iteration
- Slight approximation error acceptable (~1–2% tail singular value error)

### Complexity

- $O(\text{nnz} \cdot l \cdot (2q + 2))$ SpMMs + $O(m \cdot l^2 \cdot q)$ QR + $O(l^2 \cdot n)$ core SVD
- With $q=3$, $l = k + 10$: $\approx O(\text{nnz} \cdot 8k)$

---

## Auto-Method Selection

When `method = "auto"`, the system selects based on rank, backend, and constraints:

### CPU

| Condition | Selected method |
|---|---|
| Constraints + $k < 8$ | Deflation |
| Constraints + $k \geq 8$ | Krylov |
| No constraints + $k < 64$ | Lanczos |
| No constraints + $k \geq 64$ | IRLBA |

### GPU

| Condition | Selected method |
|---|---|
| Constraints active | Krylov or Deflation (same as CPU) |
| No constraints + $k < 32$ | Lanczos |
| No constraints + $32 \leq k < 64$ | Randomized |
| No constraints + $k \geq 64$ | IRLBA |

---

## Constrained SVD via ALS

Both Deflation and Krylov methods support constrained SVD by applying regularization within the ALS/projection framework:

### Per-Element Constraints (Deflation)

Applied after each ALS subproblem solve:

| Constraint | Formula |
|---|---|
| L1 (Lasso) | $v_j \leftarrow \text{sign}(v_j) \max(0, \|v_j\| - \lambda)$ |
| L2 (Ridge) | $v_j \leftarrow v_j / (1 + \lambda)$ |
| Non-negativity | $v_j \leftarrow \max(0, v_j)$ |
| Upper bound | $v_j \leftarrow \min(\text{ub}, v_j)$ |

### Gram-Level Constraints (Krylov)

Applied to the Gram matrix before the Cholesky solve:

| Constraint | Gram modification |
|---|---|
| L2 | $G \leftarrow G + \lambda I$ |
| L21 (Group sparsity) | $G_{ii} \leftarrow G_{ii} + \lambda / \|\text{row}_i\|$ |
| Angular (Orthogonality) | $G \leftarrow G + \lambda \cdot \text{cosine\_matrix}$ |
| Graph Laplacian | $G \leftarrow G + \lambda \cdot L$ |

---

## Auto-Rank Selection via Cross-Validation

SVD supports automatic rank determination using speckled cross-validation:

```r
result <- svd(A, k = "auto", test_fraction = 0.05, patience = 3)
```

### Mechanism

1. Define a speckled holdout mask via lazy PRNG hash (same as NMF CV)
2. For each candidate rank $k = 1, 2, 3, \ldots$:
   - Compute the $k$-th factor on training data
   - Evaluate reconstruction error on held-out test entries
3. **Early stopping**: Stop if test error increases for `patience` consecutive ranks
4. Return the model at the rank that achieved minimum test error

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `test_fraction` | 0.05 | Fraction of entries held out |
| `patience` | 3 | Stop after this many non-improving ranks |
| `mask_zeros` | `FALSE` | Whether zeros can be in the test set |
| `cv_seed` | derived | Seed for holdout pattern |

---

## References

1. Golub, G. & Kahan, W. (1965). Calculating the singular values and pseudo-inverse of a matrix. *SIAM J. Numer. Anal.*, 2(2), 205–224.
2. Baglama, J. & Reichel, L. (2005). Augmented implicitly restarted Lanczos bidiagonalization methods. *SIAM J. Sci. Comput.*, 27(1), 19–42.
3. Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. *SIAM Review*, 53(2), 217–288.
