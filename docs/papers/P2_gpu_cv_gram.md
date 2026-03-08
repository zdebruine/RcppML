# P2: GPU-Accelerated Cross-Validation for NMF via Per-Column Gram Correction

**Target Venue**: Bioinformatics or Journal of Computational and Graphical Statistics (JCGS)  
**Type**: Methods paper  
**Estimated Length**: 12–15 pages  

---

## Abstract (Draft)

Cross-validation (CV) for non-negative matrix factorization (NMF) typically
requires expensive per-iteration reconstruction to evaluate test loss on held-out
entries, creating a significant computational bottleneck. We present a novel
approach that avoids full reconstruction by lazily correcting the Gram matrix
on a per-column basis. Given a speckled holdout mask M where each entry is
independently masked with probability p, the standard Gram matrix G = H^T H
can be corrected for each column j by subtracting the rank-1 contribution of
the masked entries in that column. This per-column correction costs O(k² |mask_j|)
compared to O(mnk) for full reconstruction, where |mask_j| is the number of
masked entries in column j. We further show that this correction is trivially
parallelizable on GPU, achieving near-zero overhead for CV compared to standard
NMF fitting.

---

## Mathematical Derivation

### Standard NMF Update (Column j of H)

The NNLS subproblem for column h_j of H is:

$$\min_{h_j \geq 0} \|W^T a_j - G h_j\|^2$$

where $G = W^T W$ (the Gram matrix) and $a_j$ is column $j$ of the input matrix $A$.

### Cross-Validation Mask

Let $M \in \{0, 1\}^{m \times n}$ be a binary mask where $M_{ij} = 0$ indicates
a held-out (test) entry. The training objective only considers non-masked entries:

$$\min_{h_j \geq 0} \sum_{i: M_{ij}=1} \left(a_{ij} - w_i^T h_j\right)^2$$

### Naive Approach: Per-Column Masked Gram

The naive implementation computes a separate Gram matrix for each column:

$$G_j = \sum_{i: M_{ij}=1} w_i w_i^T = G - \sum_{i: M_{ij}=0} w_i w_i^T$$

where $w_i$ is row $i$ of $W$ (as a column vector). This is the **key insight**:
the masked Gram for column $j$ is the full Gram minus a low-rank correction.

### Lazy Correction

For a speckled mask with fraction $p$ held out, each column has approximately
$pm$ masked entries. The correction is:

$$G_j = G - \sum_{i \in \text{mask}_j} w_i w_i^T$$

**Cost**: $O(k^2 \cdot |\text{mask}_j|)$ per column, where $|\text{mask}_j| \approx pm$.
Total cost over all columns: $O(k^2 \cdot pm \cdot n) = O(k^2 mn p)$.

For typical $p = 0.05$, this is a 20× reduction compared to recomputing masked
Gram matrices from scratch.

### Similarly for RHS

$$b_j = \sum_{i: M_{ij}=1} a_{ij} w_i = W^T a_j - \sum_{i: M_{ij}=0} a_{ij} w_i$$

The RHS correction is also $O(k \cdot |\text{mask}_j|)$ per column.

### Test Loss Computation

After solving for $h_j$ on the training set, the test loss for column $j$ is:

$$\ell_j^{\text{test}} = \sum_{i: M_{ij}=0} (a_{ij} - w_i^T h_j)^2$$

This requires iterating only over the $O(pm)$ masked entries per column.

---

## GPU Implementation

### Kernel Design

1. **Gram correction kernel**: One CUDA block per column. Each block loads the
   mask indices for column $j$, fetches the corresponding rows of $W$, and
   performs the rank-1 subtraction using shared memory accumulation.

2. **RHS correction kernel**: Same block structure. Loads masked entries and
   their values, subtracts from the precomputed $W^T a_j$.

3. **NNLS solve kernel**: Per-column batch NNLS with the corrected Gram and RHS.
   Uses the same coordinate descent or Cholesky solver as the standard path.

4. **Test loss kernel**: Per-column reduction over masked entries computing
   $(a_{ij} - w_i^T h_j)^2$.

### Memory Layout

- Mask stored as CSC-like structure: column pointers + row indices (no values needed)
- W stored in k×m transposed layout for coalesced GPU memory access
- Gram matrix G (k×k) fits entirely in shared memory for k ≤ 128

### Parallelism

- **Column-level**: n columns processed in parallel (one block each)
- **Within-column**: k threads cooperate on the k×k Gram correction
- **Warp-level**: Reduction primitives for test loss accumulation

---

## Benchmark Design

### Benchmark 1: CV Overhead (CPU)
- Dataset: Simulated sparse (10000 × 5000, density 0.05)
- Compare: `nmf(k=20, maxit=50)` vs `nmf(k=20, maxit=50, test_fraction=0.1)`
- Expected: < 15% overhead for CV on CPU

### Benchmark 2: CV Overhead (GPU)
- Same dataset
- Compare: GPU standard vs GPU CV
- Expected: < 5% overhead on GPU (correction is memory-bound, tiny vs compute-bound NNLS)

### Benchmark 3: CV Quality
- Known-rank simulated data (true k=10)
- CV curves for k ∈ {2, 4, 6, 8, 10, 12, 16, 20}
- Compare: holdout fraction p ∈ {0.01, 0.05, 0.10, 0.20}
- Metric: Does the elbow correctly identify k=10?

### Benchmark 4: Scaling with Rank
- CV overhead as a function of k ∈ {4, 8, 16, 32, 64}
- Theory: O(k²) correction should dominate for large k

### Benchmark 5: Comparison with External CV
- Compare against singlet cross-validation, manual k-fold CV
- Metrics: wall time, rank selection accuracy, variance of CV estimate

---

## Figure List

1. **Figure 1**: Schematic of the lazy Gram correction (full Gram minus low-rank correction)
2. **Figure 2**: CV overhead vs. standard NMF (CPU and GPU, bar chart)
3. **Figure 3**: CV loss curves for different holdout fractions
4. **Figure 4**: Rank selection accuracy across holdout fractions
5. **Figure 5**: Wall time scaling with rank k (standard vs CV)
6. **Figure 6**: GPU kernel occupancy and memory throughput analysis

---

## Paper Outline

### 1. Introduction (1.5 pages)
- Rank selection as the key hyperparameter in NMF
- Existing CV approaches: k-fold (expensive), held-out test set (cheap but noisy)
- Contribution: lazy Gram correction + GPU acceleration

### 2. Background (1.5 pages)
- NMF as alternating NNLS
- Gram matrix and the normal equations
- Prior work on NMF cross-validation (singlet, Owen & Perry 2009)

### 3. Method: Lazy Gram Correction (3 pages)
- Speckled holdout mask design
- Per-column Gram and RHS correction derivation
- Computational complexity analysis
- Connection to matrix completion literature

### 4. GPU Implementation (2 pages)
- CUDA kernel design
- Memory access patterns and shared memory usage
- Theoretical throughput analysis

### 5. Experiments (3 pages)
- Benchmarks 1–5 description and results
- Analysis of overhead, quality, and scaling

### 6. Application: Single-Cell Rank Selection (2 pages)
- PBMC 3k dataset: CV rank selection vs. cophenetic coefficient
- MovieLens recommendations: sparse=TRUE CV

### 7. Discussion (1 page)
- Limitations (mask_zeros interaction, non-MSE losses)
- Future work (block-sparse masks, distributed CV)

---

## Reproducibility

- Code: `benchmarks/harness/suites/nmf_cv.R`
- Datasets: Simulated (generators.R) + built-in package data
- Hardware: Document GPU model, CUDA version, CPU cores
