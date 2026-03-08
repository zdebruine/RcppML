# Iteratively Reweighted Least Squares (IRLS)

## Overview

IRLS extends the standard MSE-based NMF to non-Gaussian distributions by converting the maximum likelihood problem into a sequence of weighted least squares subproblems. At each IRLS iteration, observation-specific weights are recomputed from the current reconstruction $\mu = W \cdot \text{diag}(d) \cdot H$, then a standard NNLS solve is performed on the reweighted system.

This framework unifies six distribution families (Generalized Poisson, Negative Binomial, Gamma, Inverse Gaussian, Tweedie, and MSE with robustness) under a single algorithmic structure.

---

## Mathematical Foundation

### Exponential Dispersion Family

Each supported distribution belongs to the exponential dispersion family, characterized by a **variance function** $V(\mu)$:

| Distribution | Variance Function $V(\mu)$ | Power $p$ |
|---|---|---|
| Gaussian (MSE) | $1$ | $0$ |
| Poisson (KL) | $\mu$ | $1$ |
| Generalized Poisson | $(\mu + \theta\mu)^2 / \mu$ | — |
| Negative Binomial | $\mu + \mu^2 / r$ | — |
| Gamma | $\mu^2$ | $2$ |
| Inverse Gaussian | $\mu^3$ | $3$ |
| Tweedie | $\mu^p$ | $p$ (user-specified) |

### IRLS Weight Derivation

Given observation $y$ with predicted mean $\mu = (WdH)_{ij}$, the IRLS weight is:

$$w = \frac{1}{V(\mu)}$$

This converts the distribution-specific likelihood into a weighted least squares objective:

$$\min_{h \geq 0} \sum_{i} w_i (y_i - \tilde{W}_i^T h)^2$$

which is equivalent to the standard NNLS with modified Gram and RHS:

$$G_w = \tilde{W}^T \text{diag}(w) \tilde{W}, \quad b_w = \tilde{W}^T \text{diag}(w) y$$

---

## Per-Distribution Weight Formulas

### Poisson / KL Divergence

$$w = \frac{1}{\mu}$$

The classical KL divergence objective for count data. Special case of GP with $\theta = 0$.

### Generalized Poisson (GP)

$$w = \frac{1}{s^2} + \frac{(y - 1)}{(s + \theta y)^2}$$

where $s = \mu / (1 + \theta)$ is the reparameterized mean and $\theta \geq 0$ is the overdispersion parameter. When $\theta = 0$, this reduces to Poisson.

A `gp_blend` parameter provides geometric interpolation between KL and GP weights, useful for smooth transition during optimization.

### Negative Binomial (NB)

$$w = \frac{r}{\mu(r + \mu)}$$

where $r > 0$ is the size (inverse-dispersion) parameter. As $r \to \infty$, NB $\to$ Poisson. Variance is $\mu + \mu^2/r$ (quadratic mean-variance relationship).

### Gamma

$$w = \frac{1}{\mu^2}$$

For positive continuous data with variance proportional to $\mu^2$.

### Inverse Gaussian

$$w = \frac{1}{\mu^3}$$

For positive continuous data with heavy right skew. Variance proportional to $\mu^3$.

### Tweedie

$$w = \frac{1}{\mu^p}$$

Generalized power-law variance. Interpolates between Gaussian ($p=0$), Poisson ($p=1$), Gamma ($p=2$), and Inverse Gaussian ($p=3$).

---

## Robust Delta Modifier

The `robust_delta` parameter ($\delta > 0$) applies a Huber-like downweighting to outliers, composable with **any** distribution:

1. Compute the Pearson residual:
$$r_P = \frac{y - \mu}{\sqrt{V(\mu)}} = (y - \mu) \sqrt{w_{\text{dist}}}$$

2. Apply Huber clipping to get robustness weight:
$$w_{\text{robust}} = \min\left(1, \frac{\delta}{|r_P|}\right)$$

3. Final weight:
$$w_{\text{final}} = w_{\text{dist}} \times w_{\text{robust}}$$

**Properties:**
- As $\delta \to \infty$: no effect ($w_{\text{robust}} = 1$ for all observations)
- As $\delta \to 0$: approaches MAE-like behavior (extreme downweighting)
- Acts on **standardized** residuals, so the $\delta$ threshold has consistent meaning across distributions

This replaces the legacy MAE and Huber loss types, which are fully subsumed by `robust_delta` applied to any distribution.

---

## IRLS Loop Structure

### Per-Column Algorithm

For each column $j$ of $H$, the IRLS procedure is:

```
Input: Gram G, column a_j, current solution h_j, parameters (θ, r, etc.)
Output: Updated h_j

for iter = 1 to irls_max_iter (default 5):
    1. Reconstruct: μ = W_T^T · h_j
    2. Compute weights w_i = weight(y_i, μ_i, θ_i) for each observation
       - Sparse path: only non-zero entries (O(nnz_j))
       - Dense path: all m entries (O(m))
    3. Apply robust_delta modifier if active
    4. Build weighted Gram: G_w = W_T · diag(w) · W_T^T     [O(k²m)]
    5. Build weighted RHS: b_w = W_T · (w ⊙ a_j)             [O(nnz_j · k)]
    6. Solve NNLS: h_j^new = argmin_{h≥0} ½h^T G_w h - b_w^T h
    7. Convergence check:
       max_change = max_f |h_f^new - h_f^old| / (|h_f^old| + ε)
       if max_change < irls_tol: break
```

### Convergence Criterion

IRLS converges when the **solution vector** $h_j$ stabilizes:

$$\max_f \frac{|h_f^{(t)} - h_f^{(t-1)}|}{|h_f^{(t-1)}| + 10^{-12}} < \text{irls\_tol}$$

Default parameters:
- `irls_max_iter = 5`
- `irls_tol = 1e-4`

Typical convergence speeds:
- Gamma/InvGauss: 1–2 iterations (smooth weight functions)
- KL/GP: 2–5 iterations (sensitive to θ)
- NB: 2–4 iterations

---

## Integration with Outer ALS

The IRLS-NMF iteration structure is:

```
for iter = 1 to max_iter:
    1. H update: IRLS-NNLS for each column (parallelized)
    2. W update: IRLS-NNLS for each row (via A^T, parallelized)
    3. Dispersion update: re-estimate θ, r, φ via Method-of-Moments
    4. Loss computation & convergence check
```

**Key design decision**: Dispersion parameters ($\theta$, $r$, $\phi$) are updated **once per ALS iteration**, not within the inner IRLS loop. This avoids circular dependencies between weight computation and dispersion estimation while allowing the dispersion to adapt over the course of optimization.

---

## Dispersion Parameter Estimation

### GP Theta ($\theta$) — Iterative Quadratic Solution

For each row $i$ (or globally/per-column depending on `gp_dispersion` mode):

$$\alpha_i \theta^2 + \beta_i \theta - \gamma_i = 0$$

Solved by the quadratic formula:

$$\theta_i = \frac{-\beta_i + \sqrt{\beta_i^2 + 4\alpha_i \gamma_i}}{2\alpha_i}$$

where the coefficients $\alpha$, $\beta$, $\gamma$ are computed from the data and current reconstruction. This is run for 5 inner sub-iterations per dispersion update to refine the estimate.

**Bounds**: $\theta_{\min} \leq \theta_i \leq \theta_{\max}$ (default: $[0, 5]$)

### NB Size ($r$) — Method-of-Moments

$$r_i = \frac{\sum_j \mu_{ij}^2}{\max\left(\sum_j \left[(y_{ij} - \mu_{ij})^2 - \mu_{ij}\right],\; \epsilon\right)}$$

**Bounds**: $r_{\min} \leq r_i \leq r_{\max}$

### Gamma/InvGauss/Tweedie Phi ($\phi$) — Pearson Estimator

$$\phi_i = \frac{1}{n_i} \sum_j \frac{(y_{ij} - \mu_{ij})^2}{V(\mu_{ij})}$$

**Bounds**: $\phi_{\min} \leq \phi_i \leq \phi_{\max}$

### Dispersion Scope

| Scope | Parameters estimated | Use case |
|---|---|---|
| `NONE` | Fixed at defaults | When dispersion is known or irrelevant |
| `GLOBAL` | Single scalar | Homogeneous data |
| `PER_ROW` | Vector of length $m$ | Feature-level heterogeneity (e.g., gene-specific overdispersion) |
| `PER_COL` | Vector of length $n$ | Sample-level heterogeneity |

### Newton-Raphson Refinement

Optional `nr_theta_refine > 0` iterations refine $\theta$ after the MM estimate, particularly important under zero-inflation where MM estimates can have ~25% bias.

---

## The `weight_zeros` Parameter

Controls treatment of zero entries in sparse matrices:

| Setting | Behavior | Complexity | Use case |
|---|---|---|---|
| `weight_zeros = false` (default) | Only compute weights at non-zero entries; zeros get $w = 1$ | $O(\text{nnz})$ | In-memory NMF (fast approximation) |
| `weight_zeros = true` | Compute correct distribution weights for **all** entries including zeros | $O(m)$ per column | Streaming/chunked NMF (statistically correct) |

The approximation is accurate when the data is sufficiently sparse (most entries are zero and contribute little to the weighted Gram). For dense data or streaming applications, `weight_zeros = true` is required for correct results.

---

## Complexity

| Component | Per-column cost | Notes |
|---|---|---|
| Reconstruction $\mu$ | $O(km)$ | Full dot products |
| Weight computation | $O(\text{nnz}_j)$ or $O(m)$ | Depends on `weight_zeros` |
| Weighted Gram $G_w$ | $O(k^2 m)$ | **Dominates** — recomputed per column per IRLS iter |
| Weighted RHS $b_w$ | $O(\text{nnz}_j \cdot k)$ | Sparse |
| CD NNLS solve | $O(k^2 \cdot c)$ | $c$ = CD iterations |
| **Total per column** | $O(I \cdot k^2 m)$ | $I$ = IRLS iterations (1–5) |

The per-column weighted Gram recomputation makes IRLS roughly $I \times$ more expensive than standard MSE-NMF per ALS iteration. The total overhead is typically 3–5× for count data distributions.

---

## References

1. Green, P. J. (1984). Iteratively reweighted least squares for maximum likelihood estimation and some robust and resistant alternatives. *J. R. Stat. Soc. B*, 46(2), 149–192.
2. Févotte, C. & Idier, J. (2011). Algorithms for nonnegative matrix factorization with the β-divergence. *Neural Computation*, 23(9), 2421–2456.
