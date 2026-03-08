# P3: IRLS-Based NMF for Exponential Family Distributions

**Target Venue**: Biostatistics  
**Type**: Methodology paper  
**Estimated Length**: 20–25 pages  

---

## Abstract (Draft)

We present a unified framework for non-negative matrix factorization under
exponential dispersion family distributions using Iteratively Reweighted Least
Squares (IRLS). Standard NMF minimizes the squared Frobenius norm, implicitly
assuming Gaussian observation noise. For count data (scRNA-seq, spatial
transcriptomics, text corpora), this assumption is inappropriate — the
variance-mean relationship is non-constant, leading to suboptimal factorizations
that overweight high-count entries. Our framework converts the maximum likelihood
estimation problem for six distribution families (Generalized Poisson, Negative
Binomial, Gamma, Inverse Gaussian, Tweedie, and zero-inflated variants) into a
sequence of weighted non-negative least squares (NNLS) subproblems. Each IRLS
iteration recomputes observation-specific weights from the current reconstruction,
then solves a standard weighted NNLS. We derive the weight functions, prove
convergence of the alternating IRLS-NNLS updates under mild regularity conditions,
and demonstrate that the framework achieves lower test-set deviance than both
standard MSE NMF and Poisson (KL) NMF on real single-cell RNA-seq datasets.

---

## Key Contributions

1. **Unified IRLS-NNLS framework**: A single algorithmic template that handles
   all six distributions through distribution-specific weight functions.

2. **Per-row dispersion estimation**: Method-of-moments estimation of dispersion
   parameters (θ for GP, r for NB, φ for Gamma/InvGauss, p for Tweedie) with
   iterative refinement integrated into the ALS loop.

3. **Zero-inflation extension**: EM-style E-step for zero-inflation probability
   π, interleaved with the ALS-IRLS updates, supporting row-level and
   column-level ZI modes.

4. **Convergence analysis**: Proof that the alternating IRLS scheme is a
   majorization-minimization (MM) algorithm and converges to a stationary point
   of the penalized NLL.

5. **Automatic distribution selection**: Score test diagnostics and AIC/BIC
   comparison for choosing the best-fitting distribution from data.

---

## Distribution Support Table

| Distribution | Variance $V(\mu)$ | IRLS Weight $w$ | Dispersion | Zero Prob $P(0)$ |
|---|---|---|---|---|
| Gaussian (MSE) | $1$ | $1$ | $\sigma^2$ | N/A |
| Gen. Poisson (GP) | $\mu(1+\theta)^2$ | $\frac{1}{s^2} + \frac{y-1}{(s+\theta y)^2}$ | $\theta$ | $e^{-s}$ |
| Neg. Binomial (NB) | $\mu + \mu^2/r$ | $\frac{r}{\mu(r+\mu)}$ | $r$ | $(r/(r+\mu))^r$ |
| Gamma | $\mu^2$ | $1/\mu^2$ | $\phi$ | N/A |
| Inverse Gaussian | $\mu^3$ | $1/\mu^3$ | $\phi$ | N/A |
| Tweedie | $\mu^p$ | $1/\mu^p$ | $p$ | N/A |

### Zero-Inflated Variants (GP-ZI, NB-ZI)

For count distributions, the zero-inflated model adds a point mass at zero:

$$P(Y=y) = \pi \cdot \mathbf{1}_{y=0} + (1-\pi) \cdot f(y|\mu)$$

The E-step computes posterior zero-inflation probability:

$$z_{ij} = \frac{\pi_j}{\pi_j + (1-\pi_j) \cdot P(Y_{ij}=0|\mu_{ij})}$$

for observed zeros, and $z_{ij} = 0$ for nonzeros.

---

## IRLS Algorithm

```
Algorithm: IRLS-NMF(A, k, distribution, maxit)

Input: Data matrix A (m×n), rank k, distribution family
Output: W (m×k), d (k), H (k×n)

1. Initialize W, H randomly (or via SVD seeding)
2. Compute μ = W · diag(d) · H
3. For iter = 1, ..., maxit:
   a. Compute weights: w_ij = 1 / V(μ_ij)        [distribution-specific]
   b. If zero-inflated:
      - E-step: update z_ij for all zero entries
      - Set w_ij = 0 for entries with z_ij > 0.5
   c. H-update:
      - G = W^T · diag_col(w) · W                  [weighted Gram]
      - For each column j:
        b_j = W^T · (w_·j ⊙ a_j)                  [weighted RHS]
        Solve: min_{h_j ≥ 0} ||G h_j - b_j||²     [NNLS]
   d. W-update:
      - G = H · diag_row(w) · H^T                  [weighted Gram]
      - For each column i of W^T:
        b_i = H · (w_i· ⊙ a_i·^T)
        Solve: min_{w_i ≥ 0} ||G w_i - b_i||²
   e. Normalize: d = colnorms(W); W = W / d
   f. Estimate dispersion (method of moments)
   g. Update μ = W · diag(d) · H
   h. Compute loss; check convergence
```

### Convergence Proof Sketch

The IRLS weight construction ensures that at each iteration, the weighted
least squares objective is a **quadratic majorizer** of the true NLL:

$$Q(\theta | \theta^{(t)}) = \sum_{ij} w_{ij}^{(t)} (y_{ij} - \mu_{ij})^2 + C^{(t)}$$

satisfies:
- $Q(\theta^{(t)} | \theta^{(t)}) = \ell(\theta^{(t)})$ (touching condition)
- $Q(\theta | \theta^{(t)}) \geq \ell(\theta)$ for all $\theta$ (majorization)

Therefore each IRLS step decreases the true objective, and the sequence
converges to a stationary point by the MM convergence theorem.

---

## Benchmark Design

### Benchmark 1: Distribution Selection on Simulated Data
- Generate data from known GP/NB/Gamma distributions
- Fit NMF with all 6 distributions + MSE baseline
- Metric: Test-set deviance, BIC, factor recovery (cosine similarity)
- Expected: Correct distribution yields lowest deviance

### Benchmark 2: scRNA-seq (PBMC 3k)
- Compare MSE vs GP vs NB distributions
- Metric: 10-fold CV test loss, cell type classification accuracy from H
- Expected: NB or GP outperforms MSE on count data

### Benchmark 3: IRLS Convergence Speed
- Iterations to convergence for each distribution
- Fraction of IRLS sub-iterations needed (inner loop count)
- Wall time comparison: IRLS overhead vs quality gain

### Benchmark 4: Comparison with External Poisson NMF
- Packages: RcppML (GP, NB), scikit-learn NMF(solver='mu', beta_loss='kullback-leibler')
- Metric: Per-element test deviance, wall time

### Benchmark 5: Zero-Inflation Detection and Recovery
- Simulated ZI data with known π
- Compare: NB vs NB-ZI, GP vs GP-ZI
- Metric: Recovery of true π, factor quality

---

## Figure List

1. **Figure 1**: IRLS weight functions for each distribution (w vs μ plots)
2. **Figure 2**: Convergence curves (NLL vs iteration) for six distributions on count data
3. **Figure 3**: Score test diagnostic on PBMC 3k (T-stat vs variance power)
4. **Figure 4**: AIC/BIC comparison across distributions on scRNA-seq
5. **Figure 5**: Factor quality (cosine similarity) vs distribution choice on simulated data
6. **Figure 6**: Zero-inflation: observed vs expected zeros, excess zero rate
7. **Figure 7**: Cell type classification accuracy: MSE vs GP vs NB on PBMC embeddings
8. **Table 1**: Distribution parameter table (the table above)
9. **Table 2**: Test deviance comparison across real datasets

---

## Paper Outline

### 1. Introduction (2 pages)
- Count data in genomics: discrete, overdispersed, zero-inflated
- Why Frobenius NMF fails on count data
- Contribution: unified IRLS framework for exponential family NMF

### 2. Background (2 pages)
- Exponential dispersion families
- NMF as maximum likelihood estimation
- Prior work: Lee & Seung (KL), Cemgil (Gamma NMF), Gouvert et al. (NB NMF)

### 3. IRLS-NMF Framework (5 pages)
- General weight derivation from variance function
- Per-distribution weight formulas and implementation details
- Sparse matrix optimizations (iterate only over nonzeros)
- NNLS solver interaction (CD vs Cholesky with weights)

### 4. Dispersion Estimation (2 pages)
- Method of moments for each distribution
- GP: iterative quadratic θ estimation
- NB: moment-based r with Newton refinement
- Tweedie: p selection via profile likelihood

### 5. Zero-Inflation Extension (2 pages)
- EM formulation for ZI-NMF
- Row-level vs column-level ZI modes
- Integration with IRLS weights

### 6. Convergence Analysis (3 pages)
- MM interpretation of IRLS
- Proof of monotone decrease
- Convergence rate bounds

### 7. Automatic Distribution Selection (1.5 pages)
- Score test approach (variance power)
- AIC/BIC comparison
- Diagnostic visualization

### 8. Experiments (4 pages)
- Benchmarks 1–5 results
- Real data applications

### 9. Discussion (1.5 pages)
- When to use which distribution
- Limitations of moment-based dispersion
- Future work: adaptive distribution selection during fitting

---

## Reproducibility

- Code: `benchmarks/harness/suites/nmf_distributions.R`
- Datasets: Simulated (Poisson/NB/GP generators) + PBMC 3k
- Auto-selection: `auto_nmf_distribution()`, `score_test_distribution()`
