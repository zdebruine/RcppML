# Statistical Distributions

## Overview

FactorNet supports six distribution families for NMF, each defining a different relationship between the observed data $y$ and predicted mean $\mu = (WdH)_{ij}$. The choice of distribution controls the IRLS weight function, which determines how each observation influences the factorization.

---

## Distribution Summary

| Distribution | Variance $V(\mu)$ | IRLS Weight $w$ | Dispersion | Use Case |
|---|---|---|---|---|
| Gaussian (MSE) | $1$ | $1$ (no IRLS) | None | General / dense data |
| Generalized Poisson (GP) | $\mu(1+\theta)^2$ | See below | $\theta$ (overdispersion) | Count data, Var > Mean |
| Negative Binomial (NB) | $\mu + \mu^2/r$ | $r / (\mu(r+\mu))$ | $r$ (size) | scRNA-seq, quadratic variance |
| Gamma | $\mu^2$ | $1/\mu^2$ | $\phi$ (dispersion) | Positive continuous |
| Inverse Gaussian | $\mu^3$ | $1/\mu^3$ | $\phi$ (dispersion) | Heavy right-skew positive |
| Tweedie | $\mu^p$ | $1/\mu^p$ | $p$ (power parameter) | Flexible power-law variance |

---

## Gaussian / MSE (Default)

### Probability Density

$$f(y \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y - \mu)^2}{2\sigma^2}\right)$$

### Variance Function

$$V(\mu) = 1$$

### Loss

$$\ell = \sum_{ij} (y_{ij} - \mu_{ij})^2$$

The Frobenius norm. No IRLS needed — standard NNLS directly.

### When to Use

- Continuous data with approximately constant variance
- Dense matrices
- Default choice when distribution is unknown

---

## Generalized Poisson (GP)

### Probability Mass Function

$$P(Y = y \mid \mu, \theta) = \left(\frac{\mu}{1+\theta}\right)^y \frac{(1 + \theta y)^{y-1}}{y!} \exp\left(-\frac{\mu(1+\theta y)}{1+\theta}\right)$$

with reparameterized mean $s = \mu / (1 + \theta)$.

### Variance Function

$$V(\mu) = \mu(1 + \theta)^2$$

When $\theta = 0$, this reduces to the Poisson distribution ($V(\mu) = \mu$, KL divergence).

### IRLS Weight

$$w = \frac{1}{s^2} + \frac{(y - 1)}{(s + \theta y)^2}$$

A `gp_blend` parameter provides geometric interpolation between pure KL ($\theta = 0$) and full GP weights, enabling smooth transition during optimization.

### Dispersion Estimation (Method-of-Moments)

The overdispersion parameter $\theta$ is estimated via an iterative quadratic solution with 5 inner sub-iterations per ALS step:

$$\alpha_i \theta^2 + \beta_i \theta - \gamma_i = 0 \implies \theta_i = \frac{-\beta_i + \sqrt{\beta_i^2 + 4\alpha_i\gamma_i}}{2\alpha_i}$$

**Bounds**: $\theta \in [\theta_{\min}, \theta_{\max}]$ (default: $[0, 5]$)

### Zero Probability

$$P(Y = 0 \mid \mu, \theta) = \exp\left(-\frac{\mu}{1 + \theta}\right)$$

Used in zero-inflation E-step.

### When to Use

- Count data with overdispersion (variance exceeds mean)
- $\theta = 0$ gives Poisson/KL — good baseline for count data
- More flexible than Poisson, lighter than NB

---

## Negative Binomial (NB)

### Probability Mass Function

$$P(Y = y \mid \mu, r) = \binom{y + r - 1}{y} \left(\frac{r}{r + \mu}\right)^r \left(\frac{\mu}{r + \mu}\right)^y$$

### Variance Function

$$V(\mu) = \mu + \frac{\mu^2}{r}$$

As $r \to \infty$, NB $\to$ Poisson (quadratic term vanishes).

### IRLS Weight

$$w = \frac{r}{\mu(r + \mu)}$$

### Dispersion Estimation (Method-of-Moments)

$$r_i = \frac{\sum_j \mu_{ij}^2}{\max\left(\sum_j \left[(y_{ij} - \mu_{ij})^2 - \mu_{ij}\right],\; \epsilon\right)}$$

**Bounds**: $r \in [r_{\min}, r_{\max}]$

### Zero Probability

$$P(Y = 0 \mid \mu, r) = \left(\frac{r}{r + \mu}\right)^r$$

### When to Use

- scRNA-seq and other genomic count data
- When quadratic mean-variance relationship is expected
- Standard in single-cell genomics (DESeq2, edgeR compatibility)
- More appropriate than GP when variance grows quadratically with mean

---

## Gamma

### Probability Density

$$f(y \mid \mu, \phi) = \frac{1}{\Gamma(1/\phi)} \left(\frac{y}{\phi\mu}\right)^{1/\phi} \frac{1}{y} \exp\left(-\frac{y}{\phi\mu}\right)$$

### Variance Function

$$V(\mu) = \mu^2$$

This is the Tweedie family with power $p = 2$.

### IRLS Weight

$$w = \frac{1}{\mu^2}$$

### Dispersion Estimation

$$\phi_i = \frac{1}{n_i} \sum_j \frac{(y_{ij} - \mu_{ij})^2}{\mu_{ij}^2}$$

### When to Use

- Positive continuous data where standard deviation is proportional to mean
- Insurance claims, waiting times, rainfall amounts
- When log-normal seems appropriate but a GLM framework is preferred

---

## Inverse Gaussian

### Probability Density

$$f(y \mid \mu, \phi) = \sqrt{\frac{1}{2\pi\phi y^3}} \exp\left(-\frac{(y - \mu)^2}{2\phi\mu^2 y}\right)$$

### Variance Function

$$V(\mu) = \mu^3$$

This is the Tweedie family with power $p = 3$.

### IRLS Weight

$$w = \frac{1}{\mu^3}$$

### When to Use

- Positive continuous data with heavy right skew
- Data where variance grows cubically with the mean
- Failure time modeling, spatial distances

---

## Tweedie

### Variance Function

$$V(\mu) = \mu^p$$

where $p \in [0, 3]$ is the power parameter. The Tweedie family interpolates between:

| Power $p$ | Distribution |
|---|---|
| $p = 0$ | Gaussian |
| $p = 1$ | Poisson |
| $p = 2$ | Gamma |
| $p = 3$ | Inverse Gaussian |

### IRLS Weight

$$w = \frac{1}{\mu^p}$$

### When to Use

- When the mean-variance relationship is unknown and needs to be estimated
- As a flexible alternative when no specific distribution is theoretically motivated
- For data that doesn't fit neatly into Poisson, Gamma, or Inverse Gaussian

---

## Dispersion Modes

Each distribution (except Gaussian and Tweedie) supports four dispersion scopes:

| Mode | Parameters | Description |
|---|---|---|
| `"none"` | Fixed at default | No dispersion estimation ($\theta = 0$ for GP, $r = \infty$ for NB) |
| `"global"` | 1 scalar | Single dispersion for entire matrix |
| `"per_row"` | Vector of $m$ | Feature-level heterogeneity (e.g., gene-specific overdispersion) |
| `"per_col"` | Vector of $n$ | Sample-level heterogeneity |

Dispersion parameters are updated **once per ALS iteration** via Method-of-Moments, not within the inner IRLS loop. This avoids circular dependencies while allowing the dispersion to adapt.

---

## Robustness: The `robust_delta` Modifier

The `robust_delta` parameter provides Huber-like outlier downweighting composable with **any** distribution:

1. Compute Pearson residual: $r_P = (y - \mu) / \sqrt{V(\mu)}$
2. Robustness weight: $w_{\text{robust}} = \min(1, \delta / |r_P|)$
3. Final weight: $w_{\text{final}} = w_{\text{dist}} \times w_{\text{robust}}$

This replaces the legacy MAE and Huber loss types. Setting `robust_delta > 0` with MSE loss gives Huber regression; with GP loss gives robust GP-NMF.

---

## Distribution Selection Guide

| Data Type | Recommended Distribution | Dispersion |
|---|---|---|
| Real-valued, continuous | MSE (Gaussian) | — |
| Counts, mild overdispersion | GP with $\theta = 0$ (KL) | `"none"` |
| Counts, moderate overdispersion | GP | `"per_row"` |
| scRNA-seq | NB | `"per_row"` |
| Positive continuous, CV ∝ mean | Gamma | `"per_row"` |
| Positive continuous, heavy skew | Inverse Gaussian | `"per_row"` |
| Unknown variance structure | Tweedie | — |
| Any distribution + outliers | Add `robust_delta > 0` | — |

---

## Configuration Parameters

| Parameter | Default | Distributions | Description |
|---|---|---|---|
| `distribution` | `"mse"` | All | Distribution family |
| `dispersion` | `"none"` | GP, NB, Gamma, IG | Dispersion scope |
| `robust_delta` | 0 | All | Huber robustness threshold (0 = off) |
| `gp_theta_max` | 5.0 | GP | Maximum theta |
| `gp_theta_min` | 0.0 | GP | Minimum theta |
| `gp_blend` | 0 | GP | KL↔GP interpolation |
| `nb_size_min` | 0.01 | NB | Minimum size parameter |
| `nb_size_max` | 1e6 | NB | Maximum size parameter |
| `gamma_phi_min` | 0.01 | Gamma, IG | Minimum phi |
| `gamma_phi_max` | 100 | Gamma, IG | Maximum phi |
| `tweedie_power` | 1.5 | Tweedie | Variance power $p$ |

---

## Solver Compatibility

| Distribution | CD NNLS | Cholesky+Clip | GPU |
|---|---|---|---|
| MSE | Yes | Yes | Yes |
| GP | Yes (IRLS) | No | Yes |
| NB | Yes (IRLS) | No | Yes |
| Gamma | Yes (IRLS) | No | Yes |
| Inverse Gaussian | Yes (IRLS) | No | Yes |
| Tweedie | Yes (IRLS) | No | Yes |

All non-MSE distributions require IRLS, which is only compatible with the CD solver (weights change per column per IRLS iteration, making Cholesky re-factorization too expensive).

---

## References

1. Consul, P. C. & Jain, G. C. (1973). A generalization of the Poisson distribution. *Technometrics*, 15(4), 791–799.
2. Hilbe, J. M. (2011). *Negative Binomial Regression* (2nd ed.). Cambridge University Press.
3. Jørgensen, B. (1987). Exponential dispersion models. *J. R. Stat. Soc. B*, 49(2), 127–162.
