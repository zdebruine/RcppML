# P5: FactorNet — A Graph DSL for Multi-Modal Matrix Factorization

**Target Venue**: JMLR or NeurIPS  
**Type**: Systems/methodology paper  
**Estimated Length**: 12–18 pages (conference) or 25–30 pages (journal)  

---

## Abstract (Draft)

We present FactorNet, a composable graph-based domain-specific language (DSL)
for specifying and fitting multi-modal matrix factorization networks. FactorNet
allows users to build arbitrary directed acyclic graphs (DAGs) of factorization
layers (NMF, SVD) with shared factors, conditioning variables, concatenation,
element-wise addition, per-factor regularization, and cross-layer coupling via
reference guides. The graph is compiled into an execution plan that reuses
existing NMF/SVD solvers, enabling multi-modal integration (shared-H across
data modalities), deep factorization (hierarchical W decomposition), residual
factorization (skip connections), and guided/supervised factorization — all from
a single R API. We demonstrate that FactorNet achieves competitive or superior
performance compared to MOFA+ and LIGER on standard multi-modal integration
benchmarks while offering significantly greater architectural flexibility.

---

## Graph DSL Design

### Node Types

| Node | Constructor | Description |
|------|-------------|-------------|
| Input | `factor_input(data)` | Wraps a data matrix (dense, sparse, or SPZ path) |
| NMF Layer | `nmf_layer(input, k)` | Non-negative factorization: A ≈ W·diag(d)·H |
| SVD Layer | `svd_layer(input, k)` | Signed factorization: A ≈ U·diag(d)·V |
| Shared | `factor_shared(x1, x2, ...)` | Shared-H across multiple inputs |
| Concat | `factor_concat(x1, x2, ...)` | Row-bind H factors (k1 + k2 + ...) |
| Add | `factor_add(x1, x2, ...)` | Element-wise addition of H factors |
| Condition | `factor_condition(input, Z)` | Append conditioning metadata to H |

### Per-Factor Configuration

Each layer's W and H can be independently configured:

```r
nmf_layer(inp, k = 20,
  W = W(L1 = 0.1, nonneg = TRUE, graph = laplacian),
  H = H(L2 = 0.01, guide = guide_classifier(labels))
)
```

### Global Configuration

```r
factor_config(
  maxit = 100, tol = 1e-4, loss = "gp",
  test_fraction = 0.1, resource = "auto"
)
```

### Network Compilation

```r
net <- factor_net(
  inputs = list(rna_input, atac_input),
  output = final_layer,
  config = cfg
)
result <- fit(net)
```

---

## Architecture Patterns

### Pattern 1: Multi-Modal Integration (Shared-H)

```r
rna <- factor_input(rna_matrix, "RNA")
atac <- factor_input(atac_matrix, "ATAC")
shared <- factor_shared(rna, atac)
out <- nmf_layer(shared, k = 20)
net <- factor_net(list(rna, atac), out)
```

Execution: `rbind(RNA, ATAC) ≈ rbind(W_rna, W_atac) · diag(d) · H_shared`

### Pattern 2: Deep Factorization (Hierarchical)

```r
inp <- factor_input(data)
L1 <- nmf_layer(inp, k = 64, name = "coarse")
L2 <- nmf_layer(L1, k = 16, name = "fine")
net <- factor_net(inp, L2)
```

Execution: `A ≈ W1 · d1 · H1`, then `H1 ≈ W2 · d2 · H2`, yielding
`A ≈ W1 · d1 · W2 · d2 · H2` (progressively refined representation).

### Pattern 3: Residual/Skip Connection

```r
inp <- factor_input(data)
branch1 <- nmf_layer(inp, k = 10)
branch2 <- nmf_layer(inp, k = 10)
out <- factor_add(branch1, branch2)
net <- factor_net(inp, out)
```

### Pattern 4: Conditioning (Batch Correction)

```r
inp <- factor_input(data)
L1 <- nmf_layer(inp, k = 20)
conditioned <- factor_condition(L1, batch_matrix)
L2 <- nmf_layer(conditioned, k = 10)
net <- factor_net(inp, L2)
```

### Pattern 5: Guided Factorization

```r
inp <- factor_input(data)
out <- nmf_layer(inp, k = 20,
  H = H(guide = guide_classifier(cell_type_labels, lambda = 1.0))
)
net <- factor_net(inp, out)
```

---

## Execution Engine

### Graph Compilation

`factor_net()` performs:
1. **DAG validation**: Check for cycles, verify input/output connectivity
2. **Config inheritance**: Layer defaults < global config < per-factor overrides
3. **Dimension inference**: Propagate m, n, k through the graph
4. **Resource planning**: Check GPU availability for each layer

### Fitting (`fit()`)

The `fit()` function executes the compiled graph:
1. **Topological sort**: Order layers for sequential updating
2. **Per-layer ALS**: Each layer runs its own NMF/SVD solver
3. **Shared-H coupling**: When layers share H, the update alternates between
   the shared NNLS and per-layer W updates
4. **Cross-layer guides**: Reference guides fetch the current factor from
   another layer as the target, updated each iteration
5. **Deep chains**: Output H of layer $l$ becomes input A for layer $l+1$

### Convergence

Global convergence is monitored via the sum of per-layer reconstruction losses.
Early stopping via `patience` parameter.

---

## Comparison with Existing Multi-Modal Tools

| Feature | FactorNet | MOFA+ | LIGER | scVI | Seurat v5 |
|---------|-----------|-------|-------|------|-----------|
| Model class | NMF/SVD graph | Bayesian factor | iNMF | VAE | CCA/RPCA |
| Architecture | Arbitrary DAG | Fixed | Fixed | Fixed | Fixed |
| Distributions | 6 + ZI | Gaussian/Poisson | MSE | NB | N/A |
| Per-factor config | Full | Per-view | No | No | No |
| Conditioning | Graph node | Covariates | No | Covariates | No |
| Cross-validation | Built-in | No | No | No | No |
| GPU | CUDA | No | No | PyTorch | No |
| Streaming | SPZ | No | No | No | No |
| Language | R/C++ | R/Python | R | Python | R |
| Guides/supervision | 4 types | No | No | Labels | No |

---

## Benchmark Design

### Benchmark 1: Multi-Modal Integration (PBMC Multi-ome)
- Data: 10x Multiome (scRNA + scATAC, matched cells)
- Task: Cell type identification from shared H
- Compare: FactorNet shared-H vs MOFA+ vs LIGER iNMF
- Metrics: Cell type classification accuracy, silhouette score, batch mixing

### Benchmark 2: Deep vs Shallow Factorization
- Data: AML dataset (824 × 135)
- Compare: k=20 single-layer vs k=64→k=20 two-layer
- Metrics: Reconstruction error, factor interpretability (specificity metric)

### Benchmark 3: Guided vs Unguided NMF
- Data: PBMC 3k with known cell type labels
- Compare: Standard NMF, classifier-guided NMF, LIGER
- Metrics: Classification accuracy on held-out labels, ARI

### Benchmark 4: Scalability
- Data: Simulated multi-modal (increasing m, n, k)
- Metric: Wall time, memory, convergence speed

---

## Figure List

1. **Figure 1**: FactorNet graph topology examples (5 patterns)
2. **Figure 2**: Execution flow: R DSL → compiled graph → per-layer ALS
3. **Figure 3**: Multi-modal integration UMAP/t-SNE embeddings
4. **Figure 4**: Cell type classification accuracy comparison
5. **Figure 5**: Deep factorization: hierarchical factor structure visualization
6. **Figure 6**: Guided NMF: factor alignment with known cell types
7. **Figure 7**: Scalability curves (time and memory)
8. **Table 1**: Feature comparison (the table above)
9. **Table 2**: Integration benchmark results

---

## Paper Outline

### 1. Introduction (2 pages)
- Multi-modal data explosion in genomics
- Limitations of fixed architectures
- Contribution: composable graph DSL for factorization

### 2. Related Work (2 pages)
- MOFA/MOFA+: Bayesian group factor analysis
- LIGER: integrative NMF with online learning
- scVI/scANVI: variational autoencoder approaches
- Multi-modal integration landscape

### 3. FactorNet DSL (3 pages)
- Node types and graph construction
- Per-factor configuration system
- Network compilation and validation

### 4. Execution Engine (3 pages)
- Topological sort and layer scheduling
- Shared-H update algorithm
- Deep chain execution
- Cross-layer reference guides

### 5. Guide System (2 pages)
- Classifier guide (centroid-based)
- External target guide
- Callback guide (user-defined)
- Reference guide (cross-layer coupling)

### 6. Experiments (4 pages)
- Benchmarks 1–4 results
- Visualization and interpretation

### 7. Discussion (2 pages)
- Expressiveness vs complexity tradeoff
- When to use deep vs shallow
- Future: auto-architecture search, distributed multi-node

---

## Reproducibility

- Code: `R/factor_net.R`, `R/factor_methods.R`, `R/guides.R`
- Cross-validation: `cross_validate_graph()`
- Datasets: built-in (AML, PBMC 3k) + external (10x Multiome)
