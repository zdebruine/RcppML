# Zero-Inflated NMF (ZIGP / ZINB)

## Overview

Zero-inflated NMF models data that contains **excess zeros** beyond what the count distribution alone predicts. This is common in:

- Single-cell RNA sequencing (dropout events)
- Recommendation systems (unobserved vs. disliked)
- Survey data (structural non-response)

The model assumes each zero is drawn from a two-component mixture: a **structural zero** (dropout) with probability $\pi$, or a **sampling zero** from the count distribution with probability $1 - \pi$.

---

## Mathematical Model

### Two-Component Mixture

For each entry $(i, j)$ where $Y_{ij} = 0$:

$$P(Y_{ij} = 0) = \pi_{ij} + (1 - \pi_{ij}) \cdot P_{\text{dist}}(0 \mid \mu_{ij}, \theta_i)$$

where:
- $\pi_{ij}$ is the probability that the zero is structural (dropout)
- $\mu_{ij} = (W \cdot \text{diag}(d) \cdot H)_{ij}$ is the predicted mean
- $P_{\text{dist}}(0 \mid \mu, \theta)$ is the count distribution's probability mass at zero

For non-zero entries, $Y_{ij} > 0$ implies the observation is **not** a structural zero.

### Supported Base Distributions

**ZIGP** (Zero-Inflated Generalized Poisson):

$$P(0 \mid \mu, \theta) = \exp\left(-\frac{\mu}{1 + \theta}\right)$$

**ZINB** (Zero-Inflated Negative Binomial):

$$P(0 \mid \mu, r) = \left(\frac{r}{r + \mu}\right)^r$$

---

## Zero-Inflation Modes

Three parameterizations control the scope of the dropout model:

| Mode | $\pi_{ij}$ | Parameters | Interpretation |
|---|---|---|---|
| `zi = "row"` | $\pi_i$ | $m$ values | Feature-level dropout (e.g., gene detection probability) |
| `zi = "col"` | $\pi_j$ | $n$ values | Sample-level dropout (e.g., sequencing depth) |
| `zi = "twoway"` | $1 - (1-\pi_i)(1-\pi_j)$ | $m + n$ values | Independent row and column dropout effects |

### Twoway Derivation

The twoway model assumes row and column dropout are independent mechanisms:

$$\pi_{ij} = 1 - (1 - \pi_i^{\text{row}})(1 - \pi_j^{\text{col}})$$

This means the probability of **not** being a structural zero requires both the row mechanism and column mechanism to not fire. The combined dropout rate is always at least as large as either individual rate.

---

## EM Algorithm

Zero-inflation is estimated via an Expectation-Maximization algorithm embedded within the outer ALS loop.

### Algorithm Structure

```
for iter = 1 to max_iter:
    1. Update H via NNLS (using A_imputed from previous iteration)
    2. Update W via NNLS (using A_imputed)
    3. Update dispersion parameters (θ, r) via Method-of-Moments
    4. EM loop (zi_em_iters iterations, default 1):
       a. E-step: compute posterior z_ij for each zero entry
       b. M-step: update π_i and/or π_j
    5. Soft imputation: set A_imputed[i,j] = z_ij · μ_ij for zeros
    6. Convergence check on outer loss
```

### E-Step: Posterior Dropout Probability

For each zero entry $(i, j)$:

$$z_{ij} = \frac{\pi_{ij}}{\pi_{ij} + (1 - \pi_{ij}) \cdot P(0 \mid \mu_{ij}, \theta_i)}$$

**Interpretation of $z_{ij}$:**

| Value | Meaning | Effect |
|---|---|---|
| $z_{ij} \approx 1$ | Almost certainly a structural zero (dropout) | Entry imputed to $\mu_{ij}$ |
| $z_{ij} \approx 0.5$ | Ambiguous | Partially imputed |
| $z_{ij} \approx 0$ | Almost certainly a true count zero | Entry remains 0 |

### M-Step: Update Dropout Rates

**Row mode:**

$$\pi_i^{\text{new}} = \frac{1}{n} \sum_{j=1}^{n} z_{ij} \cdot \mathbb{1}(Y_{ij} = 0)$$

**Column mode:**

$$\pi_j^{\text{new}} = \frac{1}{m} \sum_{i=1}^{m} z_{ij} \cdot \mathbb{1}(Y_{ij} = 0)$$

**Twoway mode** (with damping for stability):

To prevent double-counting in the twoway model, the $z_{ij}$ posterior is attributed proportionally:

$$z_i^{\text{row}} = z_{ij} \cdot \frac{\pi_i^{\text{row}}}{\pi_{ij}}, \quad z_j^{\text{col}} = z_{ij} \cdot \frac{\pi_j^{\text{col}}}{\pi_{ij}}$$

Updates are damped with exponential moving average (70% new, 30% old) and step-limited to $\pm 0.05$ per iteration.

### Soft Imputation

After the EM step, the working data matrix is updated:

$$A_{\text{imputed}}[i,j] = \begin{cases} z_{ij} \cdot \mu_{ij} & \text{if } Y_{ij} = 0 \\ Y_{ij} & \text{if } Y_{ij} > 0 \end{cases}$$

**Effect**: Structural zeros (high $z_{ij}$) are imputed to the model's prediction $\mu_{ij}$, so they don't penalize the reconstruction. True zeros remain and are handled normally by the count distribution's IRLS weights.

---

## Twoway Instability Analysis

The twoway mode can exhibit oscillation due to **feedback coupling** between row and column parameters:

$$\pi_i^{(t+1)} = f(\pi_1^{(t)}, \ldots, \pi_n^{(t)}), \quad \pi_j^{(t+1)} = g(\pi_1^{(t)}, \ldots, \pi_m^{(t)})$$

### Causes

1. High sparsity (>99%) creates ambiguous attribution between row and column effects
2. Row and column updates compete for the same $z_{ij}$ posterior mass
3. Without damping, rapid oscillation between over-attributing to rows vs. columns

### Stabilization Mechanisms

1. **EMA damping**: Updates weighted 70% new / 30% old
2. **Step limiting**: Each $\pi$ changes by at most $\pm 0.05$ per iteration
3. **Tight bounds**: $\pi \in [0.001, 0.95]$ for twoway (tighter than the $[0.001, 0.999]$ used for row/col modes)
4. **Conservative initialization**: $\pi = \min(0.5 \times \text{zero\_rate}, 0.3)$

With these mechanisms, twoway mode typically stabilizes within 5–10 ALS iterations. However, for extremely sparse data (>99.9% zeros), consider using row-only or column-only mode.

---

## Initialization

Dropout rates are initialized from the observed zero rate:

$$\pi_{\text{init}} = \min(0.5 \times \text{zero\_rate}, 0.3)$$

This is conservative: it attributes at most half of the observed zeros to dropout, capped at 30%. For dense data (low zero rate), $\pi \approx 0$ and the ZI mechanism has negligible effect.

---

## Interaction with Dispersion Estimation

ZI and dispersion ($\theta$ for GP, $r$ for NB) jointly explain excess zeros:

- **Without ZI**: The dispersion parameter absorbs all excess zeros → inflated $\theta$
- **With ZI**: Dropout is modeled separately → more accurate $\theta$ estimate

Optional Newton-Raphson refinement (`nr_theta_refine > 0`) corrects ~25% bias in $\theta$ that can occur when ZI is active, because the Method-of-Moments estimator doesn't account for the imputed values.

---

## Computational Complexity

| Operation | Sparse data | Dense data |
|---|---|---|
| E-step (compute $z_{ij}$ for zeros) | $O(Z \cdot k)$ | $O(mn \cdot k)$ |
| M-step (update $\pi$) | $O(m + n)$ | $O(m + n)$ |
| Soft imputation | $O(Z)$ | $O(mn)$ |
| **Total per EM iteration** | $O(Z \cdot k)$ | $O(mn \cdot k)$ |

where $Z = mn - \text{nnz}$ is the number of zero entries.

For highly sparse data (e.g., 33K × 3K at 99.9% sparsity), $Z \approx 10^8$, making the E-step a significant fraction of total iteration cost. The parallelized implementation uses thread-local accumulators to maintain efficiency.

---

## When to Use Zero-Inflation

**Recommended:**
- scRNA-seq data with known dropout effects
- Data where the zero rate exceeds what the count distribution predicts
- When separating technical zeros from biological zeros matters

**Not recommended:**
- Dense data (zero rate < 10% — ZI has negligible effect)
- When all zeros are genuine (no dropout mechanism)
- When computational cost of the E-step is prohibitive
- When using `mask = "zeros"` (CV already treats zeros specially)

---

## Configuration Parameters

| Parameter | Default | Description |
|---|---|---|
| `zi` | `"none"` | ZI mode: `"none"`, `"row"`, `"col"`, `"twoway"` |
| `zi_em_iters` | 1 | EM iterations per ALS iteration |
| `gp_theta_min` | 0.0 | Minimum theta floor (prevents EM from collapsing all zeros to dropout) |
| `nr_theta_refine` | 0 | Newton-Raphson iterations to refine theta under ZI |

---

## References

1. Risso, D. et al. (2018). A general and flexible method for signal extraction from single-cell RNA-seq data. *Nature Communications*, 9, 284.
2. Lambert, D. (1992). Zero-inflated Poisson regression, with an application to defects in manufacturing. *Technometrics*, 34(1), 1–14.
