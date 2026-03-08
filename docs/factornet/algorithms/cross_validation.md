# Cross-Validation for NMF Rank Selection

## Overview

Cross-validation (CV) in NMF partitions the data matrix into **training** and **test** sets using a speckled (element-wise) holdout pattern. The model is fit on training entries only, and predictive performance is evaluated on held-out test entries. This enables principled selection of the factorization rank $k$.

---

## Speckled Mask Design

### Lazy PRNG Hash

The holdout mask is **never materialized** as a matrix. Instead, a deterministic hash function answers $O(1)$ per-entry queries:

$$\text{is\_holdout}(i, j) \iff \text{hash}(\text{seed}, i, j) < \frac{\text{UINT64\_MAX}}{\text{inv\_prob}}$$

where `inv_prob` $= 1 / \text{test\_fraction}$.

The hash uses **SplitMix64**, a fast mixer with excellent avalanche properties:

```
state = seed ⊕ encode(i, j)
state = (state ⊕ (state >> 30)) × 0xBF58476D1CE4E5B9
state = (state ⊕ (state >> 27)) × 0x94D049BB133111EB
state = state ⊕ (state >> 31)
```

**Properties:**
- $O(1)$ per query — pure arithmetic, no memory allocation
- Deterministic — same `(seed, i, j)` always produces the same result
- Works on both CPU and GPU (no atomics or shared state)
- Uniform distribution of holdout entries across the matrix

### `mask_zeros` Semantics

| Setting | Test set definition | Use case |
|---|---|---|
| `mask_zeros = TRUE` | Only **non-zero** entries can be held out | Recommendation systems (zero = unobserved) |
| `mask_zeros = FALSE` | **All** $m \times n$ entries uniformly eligible | Dense matrix reconstruction |

**Test set sizes:**
- `mask_zeros = TRUE`: ~$\text{nnz} \times \text{test\_fraction}$ entries
- `mask_zeros = FALSE`: ~$mn \times \text{test\_fraction}$ entries

This distinction is critical when comparing CV metrics across packages — a "test MSE" computed over only non-zeros can differ by orders of magnitude from one computed over all entries.

---

## Per-Column Gram Correction

### The Problem

Standard NMF computes the global Gram matrix $G = \tilde{W}^T \tilde{W}$ and reuses it for all columns. But with CV masking, each column $j$ has a different set of held-out rows $\mathcal{T}_j$. Training on column $j$ should only use rows **not** in $\mathcal{T}_j$.

### Derivation

The full Gram decomposes as:

$$G = G_{\text{train}}^{(j)} + \sum_{i \in \mathcal{T}_j} w_i w_i^T$$

where $w_i$ is column $i$ of $\tilde{W}^T$ (i.e., row $i$ of $\tilde{W}$). Therefore:

$$G_{\text{train}}^{(j)} = G - \sum_{i \in \mathcal{T}_j} w_i w_i^T$$

This is a rank-$|\mathcal{T}_j|$ downdate of the global Gram.

**Cost:** $O(k^2 \cdot |\mathcal{T}_j|)$ per column via rank-1 outer product subtraction — efficient because $|\mathcal{T}_j|$ is small (typically $\text{test\_fraction} \times m$).

### Similarly for RHS

The right-hand side vector for column $j$ excludes test entries:

$$b_{\text{train}}^{(j)} = \tilde{W}^T a_j - \sum_{i \in \mathcal{T}_j} w_i \cdot a_{ij}$$

For sparse matrices, only the non-zero entries are iterated, and the mask is checked per entry.

---

## Train and Test Loss Computation

### Test Loss

Computed explicitly at held-out entries:

$$\ell_{\text{test}} = \frac{1}{|\mathcal{T}|} \sum_{(i,j) \in \mathcal{T}} (A_{ij} - \hat{A}_{ij})^2$$

where $\hat{A}_{ij} = (\tilde{W} H)_{ij}$ is reconstructed per test entry at cost $O(k)$ each. Total cost: $O(|\mathcal{T}| \cdot k)$.

For non-MSE distributions (IRLS), the test loss uses the distribution-specific deviance rather than squared error.

### Train Loss (Gram Trick)

For MSE, the training loss is computed efficiently without full matrix reconstruction:

$$\ell_{\text{train}} = \text{tr}(A^T A) - 2 \cdot \text{tr}(B^T H) + \text{tr}(G \cdot H H^T)$$

where $B$ and $G$ are from the training-only computations. Cost: $O(k^2)$.

For non-MSE distributions, explicit per-element computation is required at cost $O(\text{nnz} \cdot k)$.

---

## Early Stopping

CV enables early stopping based on test loss:

```
best_test_loss = ∞
patience_count = 0

for iter = 1 to max_iter:
    fit one ALS iteration (on training data)
    compute test_loss

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_iter = iter
        patience_count = 0
    else:
        patience_count += 1

    if patience_count >= patience:
        break  // Test loss has plateaued
```

**Default patience:** 5 iterations. The returned model corresponds to `best_iter` (when test loss was minimum), not the final iteration.

---

## Rank Selection Workflow

### Single Rank with CV

```r
model <- nmf(A, k = 10, test_fraction = 0.1, cv_seed = 42)
# model$misc$test_mse contains held-out loss
```

### Rank Sweep with Replicates

When `k` is a vector and `cv_seed` has multiple values, NMF runs a nested grid:

```r
cv_results <- nmf(A, k = 2:20, test_fraction = 0.1, cv_seed = 1:5)
```

Returns a `data.frame` with columns:
- `k`: factorization rank
- `rep`: replicate index (which seed)
- `train_mse`: training loss at best iteration
- `test_mse`: held-out test loss at best iteration
- `best_iter`: iteration when test loss was minimized
- `total_iter`: total iterations run

**Optimal rank selection:**

```r
# Average test MSE per rank
avg_test <- aggregate(test_mse ~ k, data = cv_results, FUN = mean)
optimal_k <- avg_test$k[which.min(avg_test$test_mse)]
```

### Reproducibility

Each `(seed, rank)` combination produces a deterministic mask and initialization:
- The mask seed controls which entries are held out
- The initialization seed (derived from the mask seed and rank) controls random starting factors
- Multiple seeds (replicates) quantify variance in the CV estimate

---

## Interaction with Other Features

### CV × Distribution

Test loss computation depends on the distribution. For non-MSE losses, test deviance replaces test MSE. The Gram correction applies the same way regardless of distribution.

### CV × Zero-Inflation

Zero-inflation EM skips held-out entries: only training zeros participate in the E-step. Held-out zeros are evaluated at face value in the test loss.

### CV × Solver

Both CD and Cholesky solvers work with CV. The per-column Gram correction happens identically in both cases — only the downstream solve differs.

### CV × Streaming

For streaming (SPZ) NMF, the lazy mask works identically since it only requires `(i, j)` coordinates. Panel boundaries are transparent to the mask.

---

## Complexity

| Component | Cost per iteration | Notes |
|---|---|---|
| H-update with Gram correction | $O(k^2 n_{\text{test}} + \text{nnz} \cdot k)$ | Correction + standard NNLS |
| W-update (symmetric) | Same | Via $A^T$ |
| Test loss | $O(|\mathcal{T}| \cdot k)$ | Explicit reconstruction at test entries |
| Train loss (Gram trick) | $O(k^2)$ | Reuses cached quantities |
| Mask queries | $O(1)$ each, $O(\text{nnz})$ total | Hash-based, no memory |

**Overhead vs. standard NMF:** The Gram correction adds $O(k^2 \cdot \text{test\_fraction} \cdot m \cdot n)$ work per iteration, typically 10–20% overhead for `test_fraction = 0.1`.

---

## Common Pitfalls

1. **Comparing CV metrics across packages**: Different packages define "test set" differently. Verify whether zeros are included before comparing MSE values.

2. **Test fraction too large**: Beyond ~20%, the Gram correction overhead becomes significant and training data may be insufficient for convergence.

3. **Too few replicates**: A single seed can produce a biased mask. Use at least 3 replicates for reliable rank selection.

4. **Ignoring early stopping**: Without patience, the model may overfit on training data while test loss increases.

---

## References

1. Owen, A. B. & Perry, P. O. (2009). Bi-cross-validation of the SVD and the nonnegative matrix factorization. *Ann. Appl. Stat.*, 3(2), 564–594.
2. DeBruine, Z. J. et al. (2024). Speckled cross-validation for non-negative matrix factorization. *bioRxiv*.
