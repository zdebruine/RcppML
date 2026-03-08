# FactorNet Graph Composition Model

## Overview

FactorNet provides a composable graph API for building complex factorization pipelines. Instead of calling `nmf()` or `svd()` directly, users construct a **directed acyclic graph (DAG)** of factorization layers with per-factor configuration, multi-modal data fusion, conditioning, and branching. The graph is then compiled and executed as a single unit.

---

## Core Concepts

### Node Types

| Node Type | Constructor | Description |
|---|---|---|
| **Input** | `factor_input(data)` | Wraps a data matrix (dense, sparse, or SPZ path) |
| **NMF Layer** | `nmf_layer(input, k)` | Non-negative factorization: $A \approx WdH$ |
| **SVD Layer** | `svd_layer(input, k)` | Unconstrained factorization (signed factors) |
| **Shared** | `factor_shared(x1, x2, ...)` | Multi-modal shared-$H$ factorization |
| **Concat** | `factor_concat(x1, x2, ...)` | Row-bind $H$ factors from branches |
| **Add** | `factor_add(x1, x2, ...)` | Element-wise $H$ addition (residual connection) |
| **Condition** | `factor_condition(input, Z)` | Append conditioning metadata to $H$ |

### Factor Configuration

Per-factor regularization is specified with `W()` and `H()` constructors:

```r
nmf_layer(x, k = 64,
  W = W(L1 = 0.1, nonneg = TRUE, graph = laplacian, graph_lambda = 0.5),
  H = H(L1 = 0.2, upper_bound = 1)
)
```

Available per-factor settings: `L1`, `L2`, `L21`, `angular`, `upper_bound`, `nonneg`, `guide`, `graph`, `graph_lambda`. Layer-level defaults apply to both factors unless overridden.

---

## Graph Construction

### Single-Layer Pipeline

```r
x <- factor_input(A, name = "rna")
out <- x |> nmf_layer(k = 32, L1 = 0.1)

net <- factor_net(inputs = x, output = out)
result <- fit(net)
```

### Hierarchical Factorization (Stacked Layers)

```r
x <- factor_input(A)
L1 <- x |> nmf_layer(k = 64)
L2 <- L1 |> nmf_layer(k = 16)

net <- factor_net(inputs = x, output = L2)
result <- fit(net)
```

The first layer decomposes $A \approx W_1 d_1 H_1$, then the second layer further decomposes $H_1 \approx W_2 d_2 H_2$, producing a two-level hierarchy.

### Multi-Modal Shared Factorization

```r
rna <- factor_input(rna_matrix, name = "rna")
protein <- factor_input(protein_matrix, name = "protein")

# Shared H across modalities
shared <- factor_shared(rna, protein)
out <- shared |> nmf_layer(k = 32)

net <- factor_net(inputs = list(rna, protein), output = out)
result <- fit(net)
```

Execution concatenates inputs row-wise and runs a single NMF:

$$\begin{bmatrix} A_{\text{rna}} \\ A_{\text{protein}} \end{bmatrix} \approx \begin{bmatrix} W_{\text{rna}} \\ W_{\text{protein}} \end{bmatrix} \cdot \text{diag}(d) \cdot H$$

The shared $H$ matrix captures common sample-level patterns across modalities, while each modality gets its own $W$ (feature loadings).

### Branching and Concatenation

```r
x <- factor_input(A)

branch1 <- x |> nmf_layer(k = 16, L1 = 0.5)   # sparse factors
branch2 <- x |> nmf_layer(k = 16, L1 = 0)      # dense factors

combined <- factor_concat(branch1, branch2)      # k_total = 32
out <- combined |> nmf_layer(k = 8)

net <- factor_net(inputs = x, output = out)
```

`factor_concat` row-binds $H$ factors: $H_{\text{out}} = [H_1; H_2]$ with combined rank $k_1 + k_2$.

### Residual Connections

```r
x <- factor_input(A)
L1 <- x |> nmf_layer(k = 32)
L2 <- L1 |> nmf_layer(k = 32)

residual <- factor_add(L1, L2)  # H = H_1 + H_2 (same rank required)
```

`factor_add` performs element-wise addition of $H$ factors. All branches must have the same rank $k$.

### Conditioning

```r
x <- factor_input(A)
L1 <- x |> nmf_layer(k = 32)
conditioned <- factor_condition(L1, Z = batch_covariates)
L2 <- conditioned |> nmf_layer(k = 16)
```

`factor_condition` appends a metadata matrix $Z$ (e.g., batch indicators, covariates) to $H$. The downstream layer's $W$ learns to separate the signal of interest from the conditioning variables.

---

## Compilation

The `factor_net()` function validates and compiles the graph:

```r
net <- factor_net(
  inputs = list(rna, protein),
  output = final_layer,
  config = factor_config(
    maxit = 100,
    tol = 1e-4,
    loss = "gp",
    solver = "auto",
    resource = "auto"
  )
)
```

### Compilation Steps

1. **Graph traversal**: Walk from output to inputs, collecting all layers
2. **Input validation**: Verify all declared inputs appear in the graph
3. **Config inheritance**: Layer-level settings override global; factor-level (`W()`, `H()`) overrides layer
4. **Name assignment**: Auto-name unnamed layers as `L1`, `L2`, ...
5. **Topology check**: Ensure DAG structure (no cycles)

### Global Configuration

`factor_config()` sets network-wide defaults:

| Parameter | Default | Description |
|---|---|---|
| `maxit` | 100 | Max ALS iterations per layer |
| `tol` | 1e-4 | Convergence tolerance |
| `loss` | `"mse"` | Distribution/loss function |
| `norm` | `"L1"` | Factor normalization |
| `solver` | `"auto"` | NNLS solver selection |
| `resource` | `"auto"` | CPU/GPU dispatch |
| `holdout_fraction` | 0 | CV holdout fraction |
| `mask_zeros` | FALSE | CV mask semantics |
| `cv_patience` | 5 | Early stopping patience |

---

## Execution

```r
result <- fit(net)
```

### Execution Flow

1. **Traverse layers** in topological order (bottom-up)
2. For each layer:
   - Resolve the input (raw data or previous layer's $H$)
   - Build NmfConfig/SvdConfig from merged settings
   - Call the C++ solver (`nmf()` or `svd()`)
   - Store $W$, $d$, $H$ in the result
3. For merge nodes (shared/concat/add):
   - Execute all upstream branches
   - Combine $H$ factors according to merge type
   - Pass combined result downstream

### Result Structure

`fit()` returns a `factor_net_result` containing per-layer results accessible by name or index:

```r
result$L1$w   # W matrix from layer 1
result$L1$d   # diagonal scaling
result$L1$h   # H matrix
result$L2$w   # W from layer 2
```

For multi-modal shared factorization, the result includes per-modality $W$ matrices:

```r
result$rna$w      # RNA feature loadings
result$protein$w  # protein feature loadings
result$shared$h   # shared sample factors
```

---

## Cross-Validation for Graphs

Graph-level cross-validation uses the same speckled mask mechanism as single-layer NMF:

```r
net <- factor_net(
  inputs = x,
  output = out,
  config = factor_config(holdout_fraction = 0.1, cv_patience = 5)
)

result <- fit(net)  # includes train/test loss per iteration
```

For multi-layer networks, the test loss is evaluated at the **final output layer** — the holdout mask applies to the original input data, and the reconstruction error is computed through the entire pipeline.

---

## Design Principles

1. **Composability**: Any node output can be the input to another layer, shared node, or merge
2. **Config inheritance**: Global → Layer → Factor, with each level overriding the previous
3. **Thin wrapper**: Graph execution delegates to the same optimized C++ NMF/SVD solvers
4. **Streaming support**: Input nodes accept SPZ file paths for out-of-core computation
5. **Backend transparency**: CPU/GPU dispatch is handled automatically based on `resource` setting

---

## Multi-Modal Use Cases

| Scenario | Graph Structure |
|---|---|
| RNA + Protein (CITE-seq) | `factor_shared(rna, protein) |> nmf_layer(k)` |
| RNA + ATAC (multiome) | `factor_shared(rna, atac) |> nmf_layer(k)` |
| Hierarchical decomposition | `input |> nmf_layer(k1) |> nmf_layer(k2)` |
| Batch correction | `layer |> factor_condition(batch) |> nmf_layer(k)` |
| Sparse + dense factors | `factor_concat(sparse_branch, dense_branch)` |
