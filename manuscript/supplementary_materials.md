# Supplementary Materials: RcppML Implementation Details

## S1. Algorithm Pseudocode

### Algorithm S1: Full NMF with Cross-Validation

```
Algorithm: NMF with Integrated Cross-Validation

Input: 
  A ∈ ℝ≥0^(m×n)     - Input matrix
  k                  - Factorization rank
  f                  - Holdout fraction
  τ                  - Convergence tolerance
  max_iter           - Maximum iterations
  patience           - Early stopping patience
  seed               - Random seed for CV masking

Output:
  W ∈ ℝ≥0^(m×k)     - Basis matrix (normalized)
  D ∈ ℝ^(k×k)       - Diagonal scaling matrix
  H ∈ ℝ≥0^(k×n)     - Coefficient matrix (normalized)
  best_iter          - Iteration with lowest test loss

Initialize:
  W ← random(m, k), H ← random(k, n)
  inv_prob ← ⌊1/f⌋
  cv_rng ← SpeckledRNG(seed)
  best_test_loss ← ∞, stale_count ← 0

for iter = 1 to max_iter:
    W_old, H_old ← W, H
    
    # H-update with CV
    G ← W^T × W
    train_loss ← 0, test_loss ← 0
    
    parallel for j = 1 to n:
        G_train, b_train, train_sqnorm ← BuildMaskedSystem(A, W, G, j, cv_rng, inv_prob)
        h_j ← CD_NNLS(G_train, b_train)
        
        # Accumulate losses (streaming, no reconstruction)
        train_loss += train_sqnorm + h_j^T G_train h_j - 2 b_train^T h_j
        for i in HoldoutRows(j, cv_rng, inv_prob):
            test_loss += (a_ij - w_i^T h_j)²
        H[:, j] ← h_j
    
    # W-update (symmetric to H-update on A^T)
    W ← SolveNNLS(A^T, H^T, cv_rng.transpose(), inv_prob)^T
    
    # Convergence check
    max_change ← max(‖W - W_old‖_∞, ‖H - H_old‖_∞)
    
    # Early stopping
    if test_loss < best_test_loss:
        best_test_loss ← test_loss
        best_iter ← iter
        stale_count ← 0
    else if test_loss > prev_test_loss AND train_loss < prev_train_loss:
        stale_count += 1
        if stale_count ≥ patience:
            break
    
    if max_change / max(1, scale) < τ:
        break
    
    prev_train_loss ← train_loss
    prev_test_loss ← test_loss

# Final normalization
for i = 1 to k:
    d_i ← ‖w_i‖ × ‖h_i‖
    w_i ← w_i / ‖w_i‖
    h_i ← h_i / ‖h_i‖

return W, D, H, best_iter
```

### Algorithm S2: Coordinate Descent with IRLS

```
Algorithm: CD-NNLS with IRLS for Robust Losses

Input:
  A        - Input matrix column (m×1)
  W        - Basis matrix (m×k)
  loss     - Loss function type {MSE, MAE, Huber, KL}
  δ        - Huber threshold (if applicable)
  max_irls - Maximum IRLS iterations
  ε_irls   - IRLS convergence tolerance

Output:
  h        - Solution vector (k×1)

Initialize:
  h ← zeros(k)
  weights ← ones(m)

for t = 1 to max_irls:
    h_old ← h
    
    # Compute weighted Gram and target
    G_w ← W^T × diag(weights) × W
    b_w ← W^T × diag(weights) × a
    
    # Solve weighted NNLS
    h ← CD_NNLS(G_w, b_w)
    
    # Check IRLS convergence
    if ‖h - h_old‖_∞ / max(1, ‖h‖_∞) < ε_irls:
        break
    
    # Update IRLS weights
    predicted ← W × h
    residuals ← a - predicted
    
    if loss = MAE:
        weights ← 1 / (|residuals| + ε)
    else if loss = Huber:
        weights ← where(|residuals| < δ, 1, δ / |residuals|)
    else if loss = KL:
        weights ← 1 / max(predicted, ε)

return h
```

### Algorithm S3: Position-Dependent CV Masking

```
Algorithm: SpeckledRNG - Deterministic Position-Based Masking

Class SpeckledRNG:
    state: uint32     - Random seed
    transpose: bool   - Transpose-identical mode

    function is_holdout(i, j, inv_prob):
        if inv_prob = 0: return false
        
        # Enforce transpose identity (optional)
        if transpose AND j > i:
            swap(i, j)
        
        # Cantor pairing for unique (i,j) hash
        ij ← (i+1)×(i+2)/2 + j + 1
        
        # xorshift mixing
        ij ← ij XOR (ij << 13) OR (i << 17)
        ij ← ij XOR (ij >> 7) OR (j << 5)
        ij ← ij XOR (ij << 17)
        
        # Combine with state
        s ← state XOR ij
        s ← s XOR (s << 23)
        s ← s XOR ij XOR (s >> 18) XOR (ij >> 5)
        
        hash ← (s + ij) mod 2³²
        return (hash mod inv_prob) = 0
    
    function transpose():
        return SpeckledRNG(state, NOT transpose)
```

---

## S2. Mathematical Derivations

### S2.1 Gram-Based Loss Computation

For MSE loss, we derive the streaming formula. Let $\mathbf{a}_j$ be the $j$-th column of $\mathbf{A}$ and $\hat{\mathbf{a}}_j = \mathbf{W}\mathbf{h}_j$ be its prediction.

The squared error is:
$$\|\mathbf{a}_j - \mathbf{W}\mathbf{h}_j\|_2^2 = \mathbf{a}_j^\top\mathbf{a}_j - 2\mathbf{a}_j^\top\mathbf{W}\mathbf{h}_j + \mathbf{h}_j^\top\mathbf{W}^\top\mathbf{W}\mathbf{h}_j$$

Substituting $\mathbf{G} = \mathbf{W}^\top\mathbf{W}$ and $\mathbf{b} = \mathbf{W}^\top\mathbf{a}_j$:
$$= \|\mathbf{a}_j\|_2^2 - 2\mathbf{b}^\top\mathbf{h}_j + \mathbf{h}_j^\top\mathbf{G}\mathbf{h}_j$$

All three terms are available after solving the normal equations, with no additional computation.

### S2.2 CV Gram Modification

For CV, let $\mathcal{T}_j$ and $\mathcal{H}_j$ partition the rows for column $j$ into training and holdout sets.

The training Gram matrix is:
$$\mathbf{G}_{\text{train}} = \sum_{i \in \mathcal{T}_j} \mathbf{w}_i\mathbf{w}_i^\top = \mathbf{G} - \sum_{i \in \mathcal{H}_j} \mathbf{w}_i\mathbf{w}_i^\top$$

Each holdout row contributes a rank-1 subtraction, which is $O(k^2)$.

### S2.3 IRLS Derivation for MAE

For MAE loss $\mathcal{L} = \sum_i |a_i - \hat{a}_i|$, we seek weights $w_i$ such that the weighted least squares problem:
$$\min_{\mathbf{h}} \sum_i w_i (a_i - \hat{a}_i)^2$$

has stationary points matching those of the original problem.

Taking gradients:
- Original: $\frac{\partial \mathcal{L}}{\partial \hat{a}_i} = -\text{sign}(a_i - \hat{a}_i)$
- Weighted LS: $\frac{\partial \mathcal{L}_w}{\partial \hat{a}_i} = -2w_i(a_i - \hat{a}_i)$

Equating: $w_i = \frac{1}{2|a_i - \hat{a}_i|}$

Adding $\epsilon$ for numerical stability: $w_i = \frac{1}{|a_i - \hat{a}_i| + \epsilon}$

### S2.4 Orthogonality Gradient

For the orthogonality penalty on $\mathbf{H}$:
$$R_\perp(\mathbf{H}) = \|\mathbf{H}\mathbf{H}^\top - \mathbf{I}\|_F^2$$

The gradient with respect to column $\mathbf{h}_j$ is:
$$\nabla_{\mathbf{h}_j} R_\perp = 4(\mathbf{H}\mathbf{H}^\top - \mathbf{I})\mathbf{h}_j$$

In the linearized approach, we use the previous iterate $\mathbf{H}^{(t-1)}$ to compute:
$$\mathbf{M} = \mathbf{H}^{(t-1)}\mathbf{H}^{(t-1)\top} - \mathbf{I}$$

Then the gradient direction is $\mathbf{g}_j = \mathbf{M}\mathbf{h}_j^{(t-1)}$, which is added to the NNLS target as a linear penalty.

---

## S3. Hyperparameter Sensitivity Analysis

### S3.1 Holdout Fraction

The holdout fraction $f$ controls the bias-variance trade-off in test loss estimation:

| Fraction $f$ | Training Set | Test Precision | Recommendation |
|--------------|--------------|----------------|----------------|
| 0.05 | 95% of entries | Lower | Large matrices (>10⁶ entries) |
| 0.10 | 90% of entries | Moderate | Default for most applications |
| 0.15 | 85% of entries | Higher | Small matrices (<10⁴ entries) |
| 0.20 | 80% of entries | Highest | Very small matrices |

### S3.2 Patience Parameter

The patience parameter controls early stopping sensitivity:

- **patience = 0**: No early stopping; run to convergence or max iterations
- **patience = 3**: Aggressive stopping; may stop prematurely
- **patience = 5**: Balanced (default); good for most use cases
- **patience = 10**: Conservative; ensures stable convergence detection

### S3.3 Coordinate Descent Iterations

The inner CD solver has its own convergence parameters:

| Parameter | Default | Effect |
|-----------|---------|--------|
| cd_max_iter | 1000 | Maximum CD cycles per NNLS solve |
| cd_tol | 1e-8 | Convergence threshold for coordinate changes |

Increasing `cd_max_iter` improves per-iteration accuracy at the cost of speed. For most applications, the default is sufficient as outer NMF iterations compensate for slightly imprecise inner solves.

---

## S4. Memory Layout and Cache Efficiency

### S4.1 Column-Major Storage

RcppML uses column-major (Fortran-style) matrix storage via Eigen:
- Accessing columns is cache-friendly (contiguous memory)
- H-update naturally iterates over columns of A and H
- W-update transposes the problem to maintain efficiency

### S4.2 Thread-Local Accumulators

To avoid false sharing and lock contention, each thread maintains private accumulators:

```cpp
struct CVLossAccumulator {
    double train_sum;    // 8 bytes
    double test_sum;     // 8 bytes
    uint64_t train_count; // 8 bytes
    uint64_t test_count;  // 8 bytes
    // Total: 32 bytes (fits in half a cache line)
};
```

After the parallel region, accumulators are merged sequentially, avoiding synchronization during the hot path.

### S4.3 Gram Matrix Reuse

The Gram matrix $\mathbf{G} = \mathbf{W}^\top\mathbf{W}$ is computed once per NMF iteration ($O(mk^2)$) and reused across all $n$ column solves. This is more efficient than computing per-column products when $n > k$.

---

## S5. Comparison with Related Methods

### S5.1 Multiplicative Update Rules

Lee & Seung's multiplicative updates (MU) are:
$$H_{ij} \leftarrow H_{ij} \frac{[\mathbf{W}^\top\mathbf{A}]_{ij}}{[\mathbf{W}^\top\mathbf{W}\mathbf{H}]_{ij}}$$

Advantages of RcppML's CD-NNLS over MU:
1. **Faster convergence**: CD achieves higher accuracy per iteration
2. **Regularization**: L1/L2 penalties are naturally incorporated
3. **Sparsity**: CD produces exactly sparse solutions (zeros) rather than near-zeros

### S5.2 Hierarchical Alternating Least Squares (HALS)

HALS solves rank-1 updates sequentially, similar to CD but at the factor level. RcppML's element-wise CD is more granular and handles regularization more naturally.

### S5.3 Projected Gradient Methods

Projected gradient methods like Lin's approach use gradient descent with projection onto the non-negative orthant. These are generally slower than second-order methods like CD-NNLS but can handle more general constraints.

---

## S6. Software Dependencies

| Component | Library | Version | Purpose |
|-----------|---------|---------|---------|
| Linear Algebra | Eigen | 3.4+ | Matrix operations, BLAS interface |
| Parallelization | OpenMP | 4.5+ | Thread-level parallelism |
| R Interface | Rcpp | 1.0.10+ | C++/R binding |
| Sparse Matrices | Eigen/RcppEigen | - | CSC format handling |
| Random Numbers | C++ STL | C++17 | xorshift mixing |

---

## S7. Reproducibility Checklist

To ensure reproducible NMF results with RcppML:

1. **Set random seed**: `seed = 42` for initialization
2. **Set CV seed**: `cv_seed = 123` for holdout pattern
3. **Fix thread count**: `options(RcppML.threads = 4)` to avoid non-determinism from thread scheduling
4. **Document tolerance**: Report `tol` and `maxit` used
5. **Report version**: Include `packageVersion("RcppML")` in methods

Note: Floating-point arithmetic may vary slightly across platforms due to compiler optimizations and SIMD instructions. For exact reproducibility, compile with `-ffp-contract=off`.

---

## References (Supplementary)

Eigen Project. (2023). Eigen: A C++ template library for linear algebra. https://eigen.tuxfamily.org/

Franc, V., Hlaváč, V., & Navara, M. (2005). Sequential coordinate-wise algorithm for the non-negative least squares problem. CAIP 2005.

Lin, C. J. (2007). Projected gradient methods for nonnegative matrix factorization. Neural Computation, 19(10), 2756-2779.

OpenMP Architecture Review Board. (2018). OpenMP Application Programming Interface, Version 5.0.
