# Non-negative Matrix Factorization (NMF)

## Mathematical Formulation

### Model

Given a non-negative matrix $A \in \mathbb{R}_{\geq 0}^{m \times n}$, NMF seeks a rank-$k$ factorization:

$$A \approx W \cdot \text{diag}(d) \cdot H$$

where:

- $W \in \mathbb{R}_{\geq 0}^{m \times k}$ — feature (basis) matrix
- $d \in \mathbb{R}_{>0}^{k}$ — diagonal scaling vector
- $H \in \mathbb{R}_{\geq 0}^{k \times n}$ — coefficient (encoding) matrix

The explicit diagonal $d$ separates factor scale from factor shape, ensuring:
1. Convex L1 regularization (regularization acts on unit-norm factors)
2. Consistent factor scales across iterations
3. Natural factor ordering by importance ($d$ sorted descending)
4. Symmetric treatment of $W$ and $H$ for symmetric inputs

### Objective Function (MSE / Frobenius Norm)

The default objective minimizes the squared Frobenius norm:

$$\min_{W, d, H \geq 0} \|A - W \cdot \text{diag}(d) \cdot H\|_F^2$$

For non-Gaussian distributions, the objective generalizes to a weighted least squares formulation via IRLS (see [irls.md](irls.md)).

---

## Alternating Least Squares (ALS) Update Rules

NMF is solved by alternating between updating $H$ (with $W$ fixed) and updating $W$ (with $H$ fixed). Each subproblem is a non-negative least squares (NNLS) problem.

### H Update

With $\tilde{W} = W \cdot \text{diag}(d)$, each column $h_j$ of $H$ is updated independently:

$$\min_{h_j \geq 0} \frac{1}{2} h_j^T G h_j - b_j^T h_j$$

where:

- **Gram matrix**: $G = \tilde{W}^T \tilde{W} \in \mathbb{R}^{k \times k}$
- **Right-hand side**: $b_j = \tilde{W}^T a_j \in \mathbb{R}^k$ (column $j$ of $A$)

The Gram matrix $G$ is computed once and reused for all $n$ columns, reducing complexity from $O(mnk)$ per column to $O(k^2 n)$ total for all RHS vectors plus $O(k^2 m)$ for the Gram.

### W Update

By transposing the problem, each row $w_i^T$ of $W$ is updated:

$$\min_{w_i \geq 0} \frac{1}{2} w_i^T G' w_i - b_i'^T w_i$$

where:

- $G' = H H^T \in \mathbb{R}^{k \times k}$
- $b_i' = H a_i'^T$ where $a_i'$ is row $i$ of $A$ (accessed via pre-transposed $A^T$)

### Diagonal Extraction and Normalization

After each $W$ update, the scaling is extracted into $d$ and $W$ is renormalized:

**L1 normalization** (default):

$$d_i = \sum_{j=1}^m W_{ji}, \quad W_{\cdot i} \leftarrow W_{\cdot i} / d_i$$

**L2 normalization**:

$$d_i = \|W_{\cdot i}\|_2, \quad W_{\cdot i} \leftarrow W_{\cdot i} / d_i$$

This maintains well-conditioned Gram matrices across iterations and ensures the diagonal captures all scale information.

---

## Convergence

### Efficient Loss Computation (Gram Trick)

The MSE loss is computed without reconstructing $W \cdot \text{diag}(d) \cdot H$ explicitly, using the identity:

$$\|A - \tilde{W} H\|_F^2 = \text{tr}(A^T A) - 2 \cdot \text{tr}(B^T H) + \text{tr}(G \cdot H H^T)$$

where $B = \tilde{W}^T A$ and $G = \tilde{W}^T \tilde{W}$ are already computed during the update step. This makes loss evaluation essentially **free** — $O(k^2)$ from the trace operations.

### Relative Tolerance with Patience

Convergence is assessed after each full ALS iteration:

$$\text{rel\_change} = \frac{|\ell_{t-1} - \ell_t|}{|\ell_{t-1}| + \epsilon}$$

where $\ell_t$ is the loss at iteration $t$ and $\epsilon = 10^{-15}$ prevents division by zero.

**Convergence rule**: The algorithm converges when `rel_change < tol` for `patience` consecutive iterations (default: `tol = 1e-4`, `patience = 5`).

This patience mechanism prevents premature termination due to noise or temporary plateaus in the loss landscape.

### Iteration Limits

- `maxit` (default 100): Hard upper bound on outer ALS iterations
- The algorithm terminates when either the patience condition is met or `maxit` is reached

---

## NMF Variants

### Standard NMF

$$A \approx W \cdot \text{diag}(d) \cdot H$$

Both $W$ and $H$ are solved via NNLS. This is the default mode.

### Projective NMF

$$A \approx W \cdot \text{diag}(d) \cdot W^T A$$

The $H$ matrix is computed deterministically as $H = \text{diag}(d) \cdot W^T A$ — no NNLS solve is needed for $H$. This:

- Provides a **1.5–2× speedup** (eliminates one Gram computation and one NNLS pass per iteration)
- Enforces a parts-based representation where $H$ lies in the column space of $W$
- Ignores any $H$-side regularization parameters
- Is appropriate when the reconstruction should be expressible entirely in terms of $W$

### Symmetric NMF

$$A \approx W \cdot \text{diag}(d) \cdot W^T$$

For square symmetric matrices ($A = A^T$), the constraint $H = W^T$ is enforced after each $W$ update. This:

- Provides a **~2× speedup** (only one factor to update)
- Is appropriate for similarity/kernel matrices, adjacency matrices, and covariance matrices
- Requires that the input matrix $A$ is square and symmetric

---

## Regularization

All regularization parameters accept a length-2 vector `c(W_reg, H_reg)` to control each factor independently.

### L1 (Lasso) — Element Sparsity

$$\text{penalty} = \lambda_{L1} \sum_{ij} |W_{ij}|$$

Applied per-element as soft-thresholding during the NNLS solve. Achieves element-level sparsity with $\lambda_{L1} \in [0, 1]$ (fraction of maximum regularization).

### L2 (Ridge) — Shrinkage

$$G \leftarrow G + \lambda_{L2} I$$

Augments the Gram matrix diagonal, providing Tikhonov regularization. Prevents ill-conditioning and encourages small factor values.

### L21 (Group Lasso) — Row Sparsity

$$\text{penalty} = \lambda_{L21} \sum_i \|W_{i \cdot}\|_2$$

Encourages entire rows of a factor matrix to be driven to zero, yielding feature selection. Applied via row-norm-dependent Gram modification.

### Angular — Orthogonality

$$G \leftarrow G + \lambda_{\text{ang}} \cdot (I - \hat{H}\hat{H}^T)$$

where $\hat{H}$ is the row-normalized $H$ matrix. Penalizes high cosine similarity between factors, encouraging orthogonal (decorrelated) components.

### Graph Laplacian — Smoothness

$$G \leftarrow G + \lambda_{\text{graph}} \cdot L$$

where $L$ is the graph Laplacian of a user-supplied adjacency matrix. Encourages factors to be smooth with respect to the graph structure (e.g., gene regulatory networks, spatial proximity).

### Upper Bound — Element Clipping

$$W_{ij} \leftarrow \min(W_{ij}, \text{upper\_bound})$$

Applied post-NNLS as a simple element-wise ceiling.

### Application Order

Regularization terms are applied to the Gram matrix $G$ **before** the NNLS solve in the following order:

1. L2 (diagonal augmentation)
2. Angular (off-diagonal correction)
3. Graph Laplacian (structural smoothness)
4. L21 (row-norm penalty)
5. L1 (applied within NNLS solver)
6. Upper bound (post-NNLS clipping)

---

## Initialization

### Random Initialization

$W$ and $H$ are filled with uniform random values in $[0, 1)$, controlled by the `seed` parameter for reproducibility.

### SVD-based Initialization

Compute the truncated SVD $A \approx U \Sigma V^T$ at rank $k$, then set:

$$W = |\,U\,| \sqrt{\Sigma}, \quad H = \sqrt{\Sigma}\, |\,V^T\,|$$

where $|\cdot|$ denotes element-wise absolute value (to ensure non-negativity). Two SVD methods are available:

| Method | Best for | Algorithm |
|--------|----------|-----------|
| Lanczos | Low $k$ ($k \leq 32$) | Lanczos bidiagonalization |
| IRLBA | Moderate $k$ ($k \geq 32$) | Implicitly restarted Lanczos |

SVD initialization typically reduces the number of ALS iterations needed for convergence by 30–50% compared to random initialization.

### User-supplied Initialization

A pre-computed $W$ matrix (or full $W$, $d$, $H$) can be passed via the `seed` parameter for warm-starting from a previous factorization.

---

## Fused RHS+NNLS Optimization

For sparse inputs under standard conditions (no masking, no IRLS, no guides), the right-hand side computation and NNLS solve are **fused** into a single parallel loop over columns:

1. Compute $b_j = \tilde{W}^T a_j$ from the sparse column
2. Immediately solve the NNLS subproblem with warm-start from the previous $h_j$

This eliminates the global $B = \tilde{W}^T A$ matrix from memory, significantly reducing memory pressure for large sparse problems while maintaining the same numerical result.

---

## Complexity Analysis

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Gram ($W^T W$) | $O(k^2 m)$ | Once per H-update |
| RHS ($W^T A$) | $O(\text{nnz} \cdot k)$ (sparse) or $O(mnk)$ (dense) | Once per H-update |
| NNLS (all columns) | $O(k^2 n \cdot c)$ | $c$ = CD iterations (~5–20) |
| Loss (Gram trick) | $O(k^2)$ | Free from cached Gram/RHS |
| Total per iteration | $O(k^2(m+n) + \text{nnz} \cdot k)$ | Sparse case |
| Total per iteration | $O(k^2(m+n) + mnk)$ | Dense case |

### Memory

| Component | Size | Notes |
|-----------|------|-------|
| $W$ | $m \times k$ | Stored transposed ($k \times m$) for cache efficiency |
| $H$ | $k \times n$ | Column-major |
| $d$ | $k$ | Diagonal scaling |
| Gram | $k \times k$ | Recomputed each half-iteration |
| $A^T$ | $\text{nnz}$ (sparse) or $mn$ (dense) | Pre-transposed for W-update |

$W$ is stored internally as $k \times m$ (transposed) to enable cache-efficient column-major iteration during the NNLS solve. The output is transposed back to $m \times k$ before returning to the user.

---

## References

1. Lee, D. & Seung, H. (1999). Learning the parts of objects by non-negative matrix factorization. *Nature*, 401, 788–791.
2. Kim, J. & Park, H. (2011). Fast nonnegative matrix factorization: An active-set-like method and comparisons. *SIAM J. Sci. Comput.*, 33(6), 3261–3281.
3. DeBruine, Z. J., Melber, K., & Bhambhani, K. (2024). Fast, scalable, and flexible non-negative matrix factorization. *bioRxiv*.
