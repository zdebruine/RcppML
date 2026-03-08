# Non-Negative Least Squares (NNLS) Solvers

## Overview

The NNLS subproblem is the inner loop of NMF. At each ALS half-iteration, every column of $H$ (or row of $W$) requires solving:

$$\min_{x \geq 0} \frac{1}{2} x^T G x - b^T x$$

where $G = \tilde{W}^T \tilde{W}$ is the $k \times k$ Gram matrix and $b = \tilde{W}^T a_j$ is the right-hand side for column $j$. FactorNet provides two NNLS solvers with different performance profiles.

---

## Solver Summary

| Solver | Mechanism | Best for | IRLS compatible | Cost per column |
|---|---|---|---|---|
| **Coordinate Descent (CD)** | Iterative Gauss-Seidel | $k \leq 24$, warm-start, IRLS | Yes | $O(k^2 \cdot s)$ |
| **Cholesky + Clip** | Direct factorization | $k \geq 24$, MSE only | No | $O(k^2)$ amortized |

---

## Coordinate Descent (CD) NNLS

### Algorithm

CD solves the NNLS problem by cycling through coordinates (factors) and performing optimal one-coordinate updates with non-negativity projection:

```
Input: G (k×k Gram), b (k RHS, overwritten as residuals),
       x (k solution, warm-started), L1, L2, upper_bound
Output: Updated x satisfying KKT conditions

for sweep = 1 to cd_maxit:
    convergence_sum = 0
    for i = 0 to k-1:
        if G[i,i] ≤ 0: continue

        step = b[i] / G[i,i]         // unconstrained direction
        step -= L1                    // L1 soft-thresholding
        step += L2 · x[i]            // L2 correction

        new_val = x[i] + step

        // Non-negativity projection
        if new_val < 0:
            actual_step = -x[i]       // clamp to zero
            x[i] = 0
        else if upper_bound > 0 and new_val > upper_bound:
            actual_step = upper_bound - x[i]
            x[i] = upper_bound
        else:
            actual_step = step
            x[i] = new_val

        // Update residuals
        for r = 0 to k-1:
            b[r] -= G[r,i] · actual_step

        convergence_sum += |actual_step| / (|x[i]| + cd_abs_tol)

    if convergence_sum / k < cd_tol:
        break  // converged
```

### L1 Soft-Thresholding

The L1 penalty is applied directly within the coordinate update as a one-shot reduction:

$$x_i^{\text{new}} = \max\left(0,\; x_i + \frac{b_i}{G_{ii}} - \lambda_{L1} + \lambda_{L2} \cdot x_i\right)$$

This is equivalent to the proximal operator for a single CD step. Combined with the non-negativity projection, it produces sparse solutions at moderate $\lambda_{L1}$ values.

### Convergence Criterion

After each full sweep over all $k$ coordinates:

$$\text{metric} = \frac{1}{k} \sum_{i=0}^{k-1} \frac{|\Delta x_i|}{|x_i| + \epsilon_{\text{abs}}} < \text{cd\_tol}$$

- `cd_tol` (default $10^{-8}$): relative convergence tolerance
- `cd_abs_tol` (default $10^{-15}$): prevents division by near-zero (pure relative mode)
- `cd_maxit` (default 100): hard iteration cap

With warm-start, convergence typically occurs in 2–5 sweeps.

### Warm-Start Mechanism

At the batch level, the RHS matrix $B$ is converted to residual form before parallel column solves:

$$B \leftarrow B - G \cdot X_{\text{prev}} \quad \text{(BLAS-3 GEMM, } O(k^2 n) \text{)}$$

Each column's residual $b_j = B_j - G \cdot x_j^{\text{prev}}$ drives the CD solver to refine from the previous solution. This avoids cold-start re-computation and dramatically reduces the number of sweeps needed.

### Complexity

| Component | Cost | Notes |
|---|---|---|
| Per sweep, per column | $O(k^2)$ | $k$ coordinates × $k$ residual updates |
| Typical sweeps (warm-start) | 3–10 | Exponential convergence |
| Batch warm-start | $O(k^2 n)$ | Single BLAS-3 GEMM |
| **Total batch** | $O(k^2 n s)$ | $s$ = average sweeps |

---

## Cholesky + Clip

### Algorithm

Solves the unconstrained least squares problem via Cholesky factorization, then projects the solution to the non-negative orthant:

```
Input: G (k×k Gram), b (k RHS), L1, L2, upper_bound
Output: x ≥ 0

if L1 > 0: b -= L1 · 1          // shift RHS for L1
if L2 > 0: G += L2 · I          // ridge penalty on diagonal

L = cholesky(G)                  // O(k³/3)
y = solve_lower(L, b)           // O(k²)
x = solve_upper(L^T, y)         // O(k²)

x = max(0, x)                   // clip negatives
if upper_bound > 0:
    x = min(x, upper_bound)     // clip to ceiling
```

### Properties

- **One-shot solve**: No iteration — single factorization + two triangular solves
- **Approximate KKT**: Clipping after solve means the solution satisfies non-negativity but may not exactly satisfy the gradient optimality conditions
- **Amortized Cholesky**: For batch NNLS, the Cholesky factorization $O(k^3)$ is done once and the $O(k^2)$ triangular solves are applied to each column

### When to Use

- **High rank** ($k \geq 24$) where batch Cholesky is cheaper than $k^2 \cdot s$ per column
- **MSE loss only** (incompatible with IRLS, which recomputes the Gram per column)
- When approximate KKT compliance is acceptable

### IRLS Incompatibility

IRLS distributions recompute the Gram matrix $G_w$ at each IRLS iteration for each column (weights change). CD handles this naturally since it's iterative. Cholesky would require re-factorization per column per IRLS iteration, negating its batch amortization advantage.

**Rule**: IRLS distributions always use CD (enforced by configuration validation).

### Complexity

| Component | Cost | Notes |
|---|---|---|
| Cholesky factorization | $O(k^3 / 3)$ | Once per batch |
| Triangular solve (per column) | $O(k^2)$ | Two backsolves |
| **Total batch** | $O(k^3 + k^2 n)$ | Amortized |

---

## CD vs. Cholesky Crossover

For batch NNLS over $n$ columns:

$$\text{CD cost} = O(k^2 \cdot n \cdot s), \quad \text{Cholesky cost} = O(k^3 + k^2 \cdot n)$$

Cholesky dominates when $n > k \cdot s$ (many columns relative to rank). For typical warm-start scenarios ($s \approx 3$):

- $k = 16, n = 10{,}000$: CD ≈ 77M flops, Cholesky ≈ 2.7M flops → **Cholesky 25× faster**
- $k = 4, n = 100$: CD ≈ 5K flops, Cholesky ≈ 1.6K flops → Similar

The crossover rank is approximately $k \approx 24$ for typical matrix sizes.

---

## Auto-Selection

The default solver mode is Cholesky + Clip (`solver_mode = 1`). The system overrides to CD when:

1. **IRLS is required** (any non-MSE distribution or `robust_delta > 0`)
2. **User explicitly requests CD** (`solver = "cd"`)
3. **GPU with $k \leq 32$** (CD kernels are highly optimized for register-resident solutions)

---

## GPU NNLS Implementations

### GPU CD: Template Per k (k ≤ 32)

Specialized CUDA kernel with the solution vector $h$ in **registers** and the Gram $G$ in **shared memory**:

- **1 thread per column**: Perfect parallelism with no synchronization
- **Zero branch divergence**: Fixed iteration count (no adaptive early termination)
- **Compile-time $k$**: Template parameter enables full loop unrolling
- **Occupancy**: ~6 blocks/SM at $k = 32$ (40 registers/thread)

### GPU CD: Dynamic Rank ($k > 32$)

Uses dynamic shared memory allocation with runtime $k$. Falls back to local memory for the solution vector when $k > 64$.

### GPU Cholesky: cuSOLVER Pipeline

Pure device-side pipeline for high-rank batch NNLS:

1. `cusolverDnDpotrf` — Cholesky factorization on device
2. `cublasDtrsm` — Batch triangular solve (forward + backward)
3. Custom kernel — Non-negativity clipping

Zero host-device transfers during the solve.

---

## Batch Parallelization (CPU)

The CPU batch NNLS parallelizes over columns using OpenMP:

```cpp
#pragma omp parallel for schedule(dynamic) num_threads(n_threads)
for (int j = 0; j < n; ++j) {
    cd_nnls_col_fixed(G, B.col(j), X.col(j), ...);
}
```

- **`schedule(dynamic)`**: Load balancing for variable convergence speeds
- **No synchronization**: Each column is fully independent
- **Column-major layout**: $B_{:,j}$ and $X_{:,j}$ are contiguous in memory

---

## Default Parameters

| Parameter | Default | Description |
|---|---|---|
| `solver` | `"auto"` | Auto-select CD or Cholesky |
| `cd_maxit` | 100 | Max CD sweeps per column |
| `cd_tol` | $10^{-8}$ | Relative convergence tolerance |
| `cd_abs_tol` | $10^{-15}$ | Absolute tolerance floor |
| `upper_bound` | 0 (none) | Element-wise ceiling |

---

## References

1. Kim, J. & Park, H. (2011). Fast nonnegative matrix factorization: An active-set-like method and comparisons. *SIAM J. Sci. Comput.*, 33(6), 3261–3281.
2. Franc, V., Hlaváč, V., & Navara, M. (2005). Sequential coordinate-wise algorithm for the non-negative least squares problem. *ICSC*, 407–414.
