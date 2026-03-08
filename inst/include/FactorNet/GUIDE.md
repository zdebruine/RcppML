# FactorNet C++ API Reference

**FactorNet** is a header-only C++ library for building and executing factorization networks — directed acyclic graphs of NMF and SVD layers connected by combinators (shared, concat, add, condition). The network graph is the central abstraction; NMF and SVD are building blocks within it.

## Table of Contents

1. [Network Graphs](#1-network-graphs)
2. [Node Types](#2-node-types)
3. [Graph Execution](#3-graph-execution)
4. [NMF Layer API](#4-nmf-layer-api)
5. [SVD Layer API](#5-svd-layer-api)
6. [Per-Factor Configuration](#6-per-factor-configuration)
7. [Regularization Features](#7-regularization-features)
8. [Guides (Semi-Supervised Learning)](#8-guides-semi-supervised-learning)
9. [Loss Functions](#9-loss-functions)
10. [Backends & Primitives](#10-backends--primitives)
11. [Clustering](#11-clustering)
12. [Streaming & I/O](#12-streaming--io)
13. [GPU Support](#13-gpu-support)
14. [RNG](#14-rng)
15. [Building & Integration](#15-building--integration)

---

## 1. Network Graphs

A FactorNet graph describes a multi-layer factorization pipeline as a DAG. You create nodes, wire them together, compile the graph, and execute it.

**Headers:**

```cpp
#include <FactorNet/graph/graph_all.hpp>   // single-include umbrella
// — or individual headers —
#include <FactorNet/graph/node.hpp>        // node types
#include <FactorNet/graph/graph.hpp>       // FactorGraph
#include <FactorNet/graph/fit.hpp>         // fit()
#include <FactorNet/graph/result.hpp>      // GraphResult, LayerResult
```

**Namespace:** `FactorNet::graph`

### Single-Layer NMF

The simplest graph: one input, one NMF layer.

```cpp
using namespace FactorNet;
using namespace FactorNet::graph;

Eigen::SparseMatrix<float> A = load_data();  // m × n

InputNode<float, SparseMatrix<float>> inp(A, "X");
NMFLayerNode<float> layer(&inp, 20, "factors");
layer.H_config.L1 = 0.01f;  // sparse H

FactorGraph<float> net({&inp}, &layer);
net.max_iter = 100;
net.tol      = 1e-5f;
net.seed     = 42;
net.compile();

auto result = fit(net, A);
// result["factors"].W  → m × 20
// result["factors"].d  → 20
// result["factors"].H  → 20 × n
```

For single-layer graphs, `fit()` delegates directly to `nmf::fit()`, preserving full access to GPU, streaming, and all other backends.

### Deep NMF (Hierarchical)

Stack NMF layers to learn a hierarchy of features. Each layer factorizes the previous layer's H.

```cpp
InputNode<float, SparseMatrix<float>> inp(A, "X");
NMFLayerNode<float> encoder(&inp, 64, "encoder");
NMFLayerNode<float> bottleneck(&encoder, 10, "bottleneck");

FactorGraph<float> net({&inp}, &bottleneck);
net.max_iter = 200;
net.tol      = 1e-5f;
net.compile();

auto result = fit(net, A);
// result["encoder"].H     → 64 × n (intermediate embedding)
// result["bottleneck"].H  → 10 × n (final compact embedding)
```

Multi-layer graphs use outer ALS: each iteration fixes all layers except one, runs a single NMF update, then rotates. Convergence is tracked on the total reconstruction loss across all layers.

### Multi-Modal (Shared H)

Jointly factorize multiple data matrices that share the same samples (columns).

```cpp
SparseMatrix<float> RNA  = load_rna();   // genes × cells
SparseMatrix<float> ATAC = load_atac();  // peaks × cells

InputNode<float, SparseMatrix<float>> rna_in(RNA, "RNA");
InputNode<float, SparseMatrix<float>> atac_in(ATAC, "ATAC");
SharedNode<float> shared({&rna_in, &atac_in});
NMFLayerNode<float> joint(&shared, 20, "joint");

FactorGraph<float> net({&rna_in, &atac_in}, &joint);
net.compile();

auto result = fit(net, RNA);  // first input drives single-layer dispatch
// result["joint"].H                → 20 × n (shared cell embedding)
// result["joint"].W_splits["RNA"]  → genes × 20
// result["joint"].W_splits["ATAC"] → peaks × 20
```

The inputs are row-concatenated internally: `[RNA; ATAC]`, producing a single NMF. The resulting W is split back by input dimensions.

### Branched Network with Concat

Run parallel branches with different factorization types, then merge.

```cpp
InputNode<float, SparseMatrix<float>> inp(A, "X");

// Branch 1: NMF captures non-negative structure
NMFLayerNode<float> branch1(&inp, 32, "nmf_branch");

// Branch 2: SVD captures signed/linear structure
SVDLayerNode<float> branch2(&inp, 16, "svd_branch");

// Merge: row-bind the two H matrices → 48 × n
ConcatNode<float> merged({&branch1, &branch2});

// Final layer operates on the 48-dim merged embedding
NMFLayerNode<float> final_layer(&merged, 10, "output");

FactorGraph<float> net({&inp}, &final_layer);
net.compile();

auto result = fit(net, A);
// result["nmf_branch"].H  → 32 × n
// result["svd_branch"].H  → 16 × n
// result["output"].H      → 10 × n (fed by 48-dim concat)
```

### Residual / Skip Connection with Add

Element-wise addition of H factors from branches (all must have the same rank).

```cpp
InputNode<float, SparseMatrix<float>> inp(A, "X");
NMFLayerNode<float> main_path(&inp, 20, "main");
NMFLayerNode<float> skip_path(&inp, 20, "skip");

AddNode<float> residual({&main_path, &skip_path});
NMFLayerNode<float> output(&residual, 10, "output");

FactorGraph<float> net({&inp}, &output);
net.compile();

auto result = fit(net, A);
// output layer's input = main.H + skip.H (element-wise)
```

### Conditioning (Covariate Correction)

Append metadata columns to H before passing to the next layer, so the downstream W learns to factor out covariates.

```cpp
InputNode<float, SparseMatrix<float>> inp(A, "X");
NMFLayerNode<float> L1(&inp, 30, "L1");

// Z: p × n matrix of covariates (batch, sex, age, etc.)
DenseMatrix<float> Z = load_covariates();
ConditionNode<float> conditioned(&L1, Z);

NMFLayerNode<float> L2(&conditioned, 10, "L2");

FactorGraph<float> net({&inp}, &L2);
net.compile();

auto result = fit(net, A);
// L2 input is [L1.H; Z] → (30+p) × n
// L2.W learns to separate signal from covariates
```

---

## 2. Node Types

All nodes derive from `Node<Scalar>` and are identified by `NodeType`.

```cpp
enum class NodeType {
    INPUT, NMF_LAYER, SVD_LAYER, SHARED, CONCAT, ADD, CONDITION
};

template<typename Scalar>
struct Node {
    NodeType type;
    std::string name;
    bool is_layer() const;       // NMF_LAYER or SVD_LAYER
    bool is_combinator() const;  // SHARED, CONCAT, or ADD
};
```

### InputNode

```cpp
template<typename Scalar, typename MatrixType>
struct InputNode : Node<Scalar> {
    const MatrixType* data;  // non-owning pointer
    int rows, cols;
    InputNode(const MatrixType& mat, const std::string& name = "");
};
```

### NMFLayerNode

```cpp
template<typename Scalar>
struct NMFLayerNode : Node<Scalar> {
    Node<Scalar>* input;          // upstream node
    int k;                         // factorization rank
    FactorConfig<Scalar> W_config; // W-side regularization
    FactorConfig<Scalar> H_config; // H-side regularization
    // Defaults: nonneg = true for both W and H
    NMFLayerNode(Node<Scalar>* in, int rank, const std::string& name = "");
};
```

### SVDLayerNode

```cpp
template<typename Scalar>
struct SVDLayerNode : Node<Scalar> {
    Node<Scalar>* input;
    int k;
    FactorConfig<Scalar> W_config;
    FactorConfig<Scalar> H_config;
    // Defaults: nonneg = false for both (signed factors)
    SVDLayerNode(Node<Scalar>* in, int rank, const std::string& name = "");
};
```

### SharedNode

```cpp
template<typename Scalar>
struct SharedNode : Node<Scalar> {
    std::vector<Node<Scalar>*> inputs;  // ≥ 2 input nodes
    explicit SharedNode(std::vector<Node<Scalar>*> ins);
};
```

### ConcatNode

```cpp
template<typename Scalar>
struct ConcatNode : Node<Scalar> {
    std::vector<Node<Scalar>*> inputs;  // ≥ 2 layer nodes
    explicit ConcatNode(std::vector<Node<Scalar>*> ins);
};
```

### AddNode

```cpp
template<typename Scalar>
struct AddNode : Node<Scalar> {
    std::vector<Node<Scalar>*> inputs;  // ≥ 2 layer nodes (same k)
    explicit AddNode(std::vector<Node<Scalar>*> ins);
};
```

### ConditionNode

```cpp
template<typename Scalar>
struct ConditionNode : Node<Scalar> {
    Node<Scalar>* input;
    DenseMatrix<Scalar> Z;  // p × n conditioning matrix
    ConditionNode(Node<Scalar>* in, const DenseMatrix<Scalar>& conditioning);
};
```

---

## 3. Graph Execution

### FactorGraph

```cpp
template<typename Scalar>
class FactorGraph {
public:
    // ─── Global config (applies to all layers) ─────
    int      max_iter;     // default: 100
    Scalar   tol;          // default: 1e-4
    bool     verbose;      // default: false
    int      threads;      // default: 0 (all available)
    uint32_t seed;         // default: 0
    LossConfig<Scalar> loss;
    NormType norm_type;    // L1, L2, or None
    int      solver_mode;  // 0 = CD, 1 = Cholesky+clip
    std::string resource;  // "auto", "cpu", "gpu"

    // ─── Construction ───────────────────────────────
    FactorGraph(std::vector<Node<Scalar>*> inputs, Node<Scalar>* output);

    // ─── Compilation ────────────────────────────────
    void compile();           // validate topology, build execution plan
    bool is_compiled() const;
    int  n_layers() const;
    const std::vector<Node<Scalar>*>& layers() const;  // topological order

    // ─── Config builder ─────────────────────────────
    NMFConfig<Scalar> build_layer_config(
        const Node<Scalar>* layer,
        int maxit_override = -1) const;
};
```

### fit() — Top-Level Dispatch

```cpp
template<typename Scalar, typename MatrixType>
GraphResult<Scalar> fit(const FactorGraph<Scalar>& graph, const MatrixType& data);
```

Dispatches to:
- **Single-layer**: `nmf::fit()` directly (GPU/MPI/streaming available)
- **Multi-layer**: outer ALS with per-layer `nmf::fit()` calls

### GraphResult

```cpp
template<typename Scalar>
struct GraphResult {
    std::map<std::string, LayerResult<Scalar>> layers;
    int    total_iterations;
    Scalar total_loss;
    bool   converged;

    const LayerResult<Scalar>& operator[](const std::string& name) const;
};

template<typename Scalar>
struct LayerResult {
    DenseMatrix<Scalar> W;     // m × k
    DenseVector<Scalar> d;     // k
    DenseMatrix<Scalar> H;     // k × n
    std::map<std::string, DenseMatrix<Scalar>> W_splits;  // multi-modal only
    int    iterations;
    Scalar loss;
    bool   converged;
};
```

---

## 4. NMF Layer API

For standalone NMF (without the graph layer), use the NMF gateway directly.

**Header:** `#include <FactorNet/nmf/fit.hpp>`

```cpp
template<typename Scalar, typename MatrixType>
NMFResult<Scalar> nmf::fit(
    const MatrixType& A,
    const NMFConfig<Scalar>& config,
    const DenseMatrix<Scalar>* W_init = nullptr,
    const DenseMatrix<Scalar>* H_init = nullptr);
```

**Behavior:**
1. Detects available resources (CPU cores, GPUs, MPI ranks)
2. Selects backend (auto, or honor `config.resource_override`)
3. Dispatches to standard NMF or cross-validation NMF based on `config.is_cv()`
4. GPU errors gracefully fall back to CPU

### NMFConfig

```cpp
template<typename Scalar>
struct NMFConfig {
    // Core
    int rank, max_iter;
    Scalar tol;
    int patience;
    uint32_t seed;
    bool verbose;
    int threads, max_gpus;

    // Per-factor config
    FactorConfig<Scalar> W;
    FactorConfig<Scalar> H;

    // Algorithm variants
    bool projective;  // H = diag(d)·W^T·A
    bool symmetric;   // A ≈ W^T·diag(d)·W
    int  init_mode;   // 0 = random, 1 = Lanczos SVD

    // NNLS solver
    int solver_mode;  // 0 = CD, 1 = Cholesky+clip
    int cd_max_iter;
    Scalar cd_tol, cd_abs_tol;

    // Loss (MSE, MAE, Huber, KL, GP, NB)
    LossConfig<Scalar> loss;
    int irls_max_iter;
    Scalar irls_tol;

    // GP/NB dispersion
    DispersionMode gp_dispersion;
    Scalar gp_theta_init, gp_theta_max;

    // Zero inflation
    ZIMode zi_mode;
    int zi_em_iters;

    // Cross-validation
    Scalar holdout_fraction;
    uint32_t cv_seed;
    bool mask_zeros;
    int cv_patience;

    // User-supplied mask
    const SparseMatrix<Scalar>* mask;

    // Post-processing
    NormType norm_type;
    bool sort_model;

    // Resource
    std::string resource_override;
};
```

### NMFResult

```cpp
template<typename Scalar>
struct NMFResult {
    DenseMatrix<Scalar> W;   // m × k
    DenseVector<Scalar> d;   // k
    DenseMatrix<Scalar> H;   // k × n

    // Dispersion (GP/NB)
    DenseVector<Scalar> theta, pi_row, pi_col;

    // Convergence
    int iterations;
    bool converged;
    Scalar final_tol, train_loss, test_loss;
    Scalar best_test_loss;
    int best_iter;

    // History
    FitHistory<Scalar> history;
    double wall_time_ms;
    ResourceDiagnostics diagnostics;

    // Methods
    DenseMatrix<Scalar> reconstruct() const;
    void normalize(NormType);
    void sort_by_d(bool descending = true);
};
```

### Algorithm Variants

| Variant | Config | Description |
|---------|--------|-------------|
| Standard NMF | defaults | A ≈ W·diag(d)·H, W ≥ 0, H ≥ 0 |
| Semi-NMF | `W.nonneg = false` | Signed W, nonneg H |
| Projective NMF | `projective = true` | H = diag(d)·W^T·A — no H-side NNLS |
| Symmetric NMF | `symmetric = true` | A ≈ W^T·diag(d)·W, A must be square |
| Box-constrained | `H.upper_bound = 1` | 0 ≤ h ≤ 1 |

### Cross-Validation

```cpp
config.holdout_fraction = 0.1;
config.cv_seed          = 42;
config.mask_zeros       = true;   // recommendation-style (nonzeros only)

auto result = nmf::fit(A, config);
// result.test_loss, result.best_test_loss, result.best_iter
```

Uses a **lazy speckled mask** — holdout status is determined by `SplitMix64::is_holdout(seed, i, j, frac)` in O(1), no mask matrix stored. The Gram matrix receives a per-column delta-G correction to exclude held-out rows exactly.

---

## 5. SVD Layer API

**Header:** `#include <FactorNet/svd/gateway.hpp>`

```cpp
template<typename Scalar, typename MatrixType>
SVDResult<Scalar> svd::svd_gateway(
    const MatrixType& A,
    const SVDConfig<Scalar>& config,
    const std::string& method = "auto");
```

### SVDConfig

```cpp
template<typename Scalar>
struct SVDConfig {
    int k_max;
    Scalar tol;
    int max_iter;
    bool center, scale;  // PCA / correlation PCA
    uint32_t seed;
    int threads;

    // Per-component regularization
    Scalar L1_u, L1_v, L2_u, L2_v;
    bool nonneg_u, nonneg_v;
    Scalar upper_bound_u, upper_bound_v;

    // Gram-level (deflation/Krylov only)
    Scalar L21_u, L21_v;
    Scalar angular_u, angular_v;
    const SparseMatrix<Scalar>* graph_u, *graph_v;
};
```

### SVDResult

```cpp
template<typename Scalar>
struct SVDResult {
    DenseMatrix<Scalar> U;     // m × k
    DenseVector<Scalar> d;     // k (singular values)
    DenseMatrix<Scalar> V;     // k × n
    DenseRowVector<Scalar> center;  // column means (PCA)
    DenseRowVector<Scalar> scale;   // column stds (correlation PCA)
};
```

### Methods

| Method | String | Best for |
|--------|--------|----------|
| Deflation | `"deflation"` | Small k, full regularization |
| Lanczos | `"lanczos"` | Medium k, stable convergence |
| IRLBA | `"irlba"` | Large k, fast |
| Randomized | `"randomized"` | Very large matrices |
| Krylov | `"krylov"` | Custom hybrid solver |

`"auto"` selects based on matrix dimensions, rank, and sparsity.

### PCA Mode

```cpp
SVDConfig<float> cfg;
cfg.k_max  = 50;
cfg.center = true;   // subtract column means
cfg.scale  = true;   // divide by column std (correlation PCA)
cfg.L1_v   = 0.05f;  // sparse loadings

auto pca = svd::svd_gateway(A, cfg, "deflation");
// pca.U, pca.d, pca.V, pca.center, pca.scale
```

---

## 6. Per-Factor Configuration

**Header:** `#include <FactorNet/core/factor_config.hpp>`

`FactorConfig<Scalar>` controls regularization, constraints, and guides for a single factor matrix (W or H).

```cpp
template<typename Scalar>
struct FactorConfig {
    // Tier 1 — O(k) per column
    Scalar L1 = 0;             // lasso (sparsity)
    Scalar L2 = 0;             // ridge (shrinkage)

    // Tier 2 — O(k²·n), Gram-level
    Scalar L21     = 0;        // group sparsity (feature selection)
    Scalar angular = 0;        // orthogonality penalty

    // Constraints
    bool   nonneg      = true; // x ≥ 0
    Scalar upper_bound = 0;    // box constraint (0 = none)

    // Graph Laplacian
    const SparseMatrix<Scalar>* graph = nullptr;
    Scalar graph_lambda = 0;

    // Semi-supervised guides
    std::vector<guides::Guide<Scalar>*> guides;

    // Queries
    bool has_tier2_features() const;
    bool has_any_reg() const;
    bool has_guides_active() const;
};
```

Used in two places:
- **NMFConfig**: `config.W` and `config.H` for standalone NMF
- **Graph nodes**: `NMFLayerNode.W_config` and `NMFLayerNode.H_config` for per-layer control

---

## 7. Regularization Features

All regularization operates on the small k×k Gram matrix G and k×n RHS matrix B, applied between primitive computations and the NNLS solve. Cost: O(k²), negligible versus O(nnz·k) primitives.

**Apply order:** L2 → Angular → Graph → L21 → L1 (inside NNLS) → Guides → NNLS → Bounds

### L1 / L2 (Lasso / Ridge)

```cpp
config.H.L1 = 0.01;  // promotes sparsity in H
config.H.L2 = 0.001; // shrinks factor magnitudes
// L2: G.diagonal() += L2
// L1: b_j -= L1 (soft-threshold shift)
```

### L21 (Group Sparsity)

Drives entire rows of a factor to zero (feature selection):

```cpp
config.H.L21 = 0.1;
// G(i,i) += L21 / ||row_i|| (small rows get stronger penalty)
```

### Angular (Orthogonality)

Penalizes factor overlap:

```cpp
config.H.angular = 0.05;
// overlap = cosine similarity of H columns → G += angular * overlap
```

### Graph Laplacian

Smooths factors along a graph (spatial neighbors, gene networks):

```cpp
SparseMatrix<double> L = compute_laplacian(adjacency);
config.H.graph        = &L;
config.H.graph_lambda = 0.1;
// G += λ · (L^T · L)
```

### Box Constraints

```cpp
config.H.upper_bound = 1.0;  // 0 ≤ h ≤ 1
config.W.nonneg = false;       // signed W (semi-NMF)
```

---

## 8. Guides (Semi-Supervised Learning)

Guides inject prior knowledge by modifying the Gram and RHS to attract (or repel) factors toward targets. Mathematically: add λ·‖h_j − t_j‖² to the per-column NNLS objective.

**Header:** `#include <FactorNet/guides/guide.hpp>`

### Guide Base Class

```cpp
template<typename Scalar>
struct Guide {
    Scalar lambda = 1;  // positive = attract, negative = repel

    virtual void apply(DenseMatrix<Scalar>& G, DenseMatrix<Scalar>& B) const = 0;
    virtual void update(const DenseMatrix<Scalar>& factor, int iteration) {}
};
```

### ClassifierGuide

**Header:** `#include <FactorNet/guides/classifier_guide.hpp>`

Steers factors toward per-class centroids (computed each iteration from labeled samples):

```cpp
guides::ClassifierGuide<float> guide;
guide.lambda = 0.5f;
guide.set_labels(labels, num_classes);

config.H.guides.push_back(&guide);
```

### ExternalGuide

**Header:** `#include <FactorNet/guides/external_guide.hpp>`

Fixed target matrix (transfer learning, prior knowledge, cross-layer coupling):

```cpp
guides::ExternalGuide<float> guide;
guide.lambda = 0.3f;
guide.target_matrix = &T;  // k × n

config.H.guides.push_back(&guide);
```

### Custom Guide

```cpp
template<typename Scalar>
struct MyGuide : public guides::Guide<Scalar> {
    void apply(DenseMatrix<Scalar>& G, DenseMatrix<Scalar>& B) const override {
        G.diagonal().array() += std::abs(this->lambda);
        // steer B toward your target...
    }
    void update(const DenseMatrix<Scalar>& factor, int iteration) override {
        // recompute targets from current factor state
    }
};
```

---

## 9. Loss Functions

| Loss | `LossType` | IRLS | Use case |
|------|------------|------|----------|
| MSE | `MSE` | No | Gaussian noise (default) |
| MAE | `MAE` | Yes | Robust to outliers |
| Huber | `HUBER` | Yes | Smooth MSE→MAE transition |
| KL divergence | `KL` | Yes | Count data, topic models |
| Generalized Poisson | `GP` | Yes | Overdispersed counts |
| Negative Binomial | `NB` | Yes | Highly overdispersed (scRNA-seq) |

Non-MSE losses use IRLS: per-element weights transform the objective into weighted least-squares solved by the same NNLS machinery.

### LossConfig

```cpp
enum class LossType { MSE, MAE, HUBER, KL, GP, NB };

template<typename Scalar>
struct LossConfig {
    LossType type = LossType::MSE;
    Scalar huber_delta = 1.0;
    bool requires_irls() const;
};
```

### Zero Inflation (ZIGP / ZINB)

```cpp
config.loss.type   = LossType::GP;
config.zi_mode     = ZIMode::ZI_ROW;
config.zi_em_iters = 3;
```

Each iteration runs E-step (posterior probability each zero is structural) + M-step (update π), then soft-imputes structural zeros.

---

## 10. Backends & Primitives

**Header:** `#include <FactorNet/primitives/primitives.hpp>`

Four primitive operations abstract the compute backend. All NMF/SVD algorithms call these — never raw BLAS or CUDA directly.

```cpp
namespace primitives {
template<typename Resource, typename Scalar>
void gram(const DenseMatrix<Scalar>& H, DenseMatrix<Scalar>& G);

template<typename Resource, typename Scalar, typename MatrixType>
void rhs(const MatrixType& A, const DenseMatrix<Scalar>& H, DenseMatrix<Scalar>& B);

template<typename Resource, typename Scalar>
void nnls_batch(const DenseMatrix<Scalar>& G, DenseMatrix<Scalar>& B,
                DenseMatrix<Scalar>& X, int cd_maxit, Scalar cd_tol, ...);

template<typename Resource, typename Scalar, typename MatrixType>
Scalar trace_AtA(const MatrixType& A);
}
```

### CPU Specializations (`primitives/cpu/`)

| File | Operation | Notes |
|------|-----------|-------|
| `gram.hpp` | G = H·H^T | `selfadjointView::rankUpdate` for 2× speedup |
| `rhs.hpp` | B = H·A | OpenMP parallel over sparse columns |
| `nnls_batch.hpp` | CD NNLS | Per-column CD with warm start |
| `cholesky_clip.hpp` | Cholesky+clip | `LLT::solve` then clip to bounds |
| `nnls_batch_irls.hpp` | Weighted CD | IRLS weights for non-MSE losses |
| `fused_nnls.hpp` | Fused RHS+NNLS | One pass per column (better cache) |

### Fused Path

For sparse MSE without masking, the fused path computes RHS and solves NNLS per-column in a single parallel loop, eliminating one full O(nnz·k) pass:

```
Standard: rhs(A, W, B) → nnls_batch(G, B, H)  [2 passes over data]
Fused:    fused_rhs_nnls_sparse(A, W, G, H)    [1 pass over data]
```

---

## 11. Clustering

**Header:** `#include <FactorNet/clustering/gateway.hpp>`

### Spectral Bipartitioning

Rank-2 NMF-based column bipartitioning:

```cpp
auto result = clustering::bipartition_gateway(
    A, tol, max_iter, nonneg, sample_indices,
    seed, verbose, compute_distance, diag_scale, threads);
// result.assignments, result.cluster_centers, result.inter_cluster_distance
```

### Divisive Clustering

Hierarchical top-down clustering via recursive bipartitioning:

```cpp
auto tree = clustering::dclust_gateway(
    A, tol, max_iter, nonneg, sample_indices,
    seed, verbose, threads);
// tree.samples, tree.id
```

---

## 12. Streaming & I/O

### DataLoader Interface

**Header:** `#include <FactorNet/io/loader.hpp>`

Abstract interface for column-chunked data access:

```cpp
template<typename Scalar>
class DataLoader {
public:
    virtual uint32_t rows() const = 0;
    virtual uint32_t cols() const = 0;
    virtual bool next_forward(Chunk<Scalar>& out) = 0;
    virtual bool next_transpose(Chunk<Scalar>& out) = 0;
    virtual void reset_forward() = 0;
    virtual void reset_transpose() = 0;
};
```

### InMemoryLoader

Zero-copy chunked views of an in-memory sparse matrix:

```cpp
InMemoryLoader<double> loader(A, /*chunk_size=*/ 50000);
```

### SpzLoader

Streaming decompression of `.spz` files for out-of-core NMF:

```cpp
SpzLoader<double> loader("data.spz");
auto result = nmf::fit_streaming(loader, config);  // MSE only
```

---

## 13. GPU Support

GPU acceleration for all four primitives. Same NMF algorithm, dispatched through `primitives::GPU`.

### Requirements

- CUDA toolkit + nvcc
- `#define FACTORNET_HAS_GPU`
- cuBLAS + cuSPARSE

### GPU Kernels (`primitives/gpu/`)

| Kernel | Operation |
|--------|-----------|
| `gram.cuh` | cuBLAS SYRK |
| `rhs_sparse.cuh` | cuSPARSE SpMM |
| `rhs_dense.cuh` | cuBLAS GEMM |
| `nnls_batch.cuh` | Custom CUDA per-column CD |
| `loss.cuh` | Reduction kernel |
| `mixed_precision.cuh` | FP16 Tensor Core shadows |

### Runtime Bridge (CPU-only builds)

When compiled without CUDA, GPU ops dispatch via `dlsym` to a separate `.so`.

---

## 14. RNG

**Header:** `#include <FactorNet/rng/rng.hpp>`

```cpp
class SplitMix64 {
public:
    explicit SplitMix64(uint64_t seed) noexcept;
    uint64_t next() noexcept;

    template<typename T> T uniform() noexcept;       // [0, 1)
    template<typename T> void fill_uniform(T* ptr, int n) noexcept;

    // Thread-safe, GPU-compatible positional hash
    static uint64_t hash(uint64_t seed, int64_t i, int64_t j) noexcept;
    static bool is_holdout(uint64_t seed, int64_t i, int64_t j, double frac) noexcept;
};
```

`hash()` and `is_holdout()` are pure functions — deterministic, thread-safe, `__host__ __device__`. Enables lazy CV mask evaluation without materializing a mask matrix.

---

## 15. Building & Integration

### CMake

```cmake
find_package(Eigen3 REQUIRED)
add_executable(my_app main.cpp)
target_include_directories(my_app PRIVATE path/to/FactorNet/inst/include)
target_link_libraries(my_app Eigen3::Eigen)

# Optional: OpenMP, CUDA
```

### Dependencies

| Library | Required | Purpose |
|---------|----------|---------|
| Eigen 3.4+ | Yes | Dense/sparse linear algebra |
| OpenMP | Recommended | Parallel NNLS, RHS, loss |
| CUDA toolkit | Optional | GPU acceleration |

### Preprocessor Flags

| Flag | Effect |
|------|--------|
| `FACTORNET_HAS_GPU` | Compile-time GPU paths |

---

## Architecture Overview

```
FactorNet/
├── graph/         ← Network graph: nodes, compilation, execution
├── core/          ← Type aliases, configs, results, constants, traits
├── nmf/           ← NMF algorithms (standard, CV, streaming, GPU)
├── svd/           ← SVD/PCA (deflation, Lanczos, IRLBA, randomized, Krylov)
├── primitives/
│   ├── cpu/       ← OpenMP-parallel Gram, RHS, NNLS, loss
│   └── gpu/       ← CUDA kernels (same API via template dispatch)
├── features/      ← Regularization: L1, L2, L21, angular, graph, bounds
├── guides/        ← Semi-supervised targets (classifier, external)
├── clustering/    ← Spectral bipartitioning, divisive clustering
├── io/            ← DataLoader (in-memory, SPZ streaming)
├── gpu/           ← GPU context, bridge for CPU-only builds
├── math/          ← Loss functions, BLAS utilities
└── rng/           ← Unified SplitMix64 RNG (CPU + GPU)
```

**Design principles:**

1. **Graph-first** — The `FactorGraph` is the primary abstraction. Single-layer NMF is a degenerate graph.
2. **Single config type** — `NMFConfig<Scalar>` describes *what* to compute, never *how*.
3. **Backend abstraction** — Primitives are specialized per `Resource` tag (`CPU` or `GPU`).
4. **k×k feature layer** — Regularization modifies the small Gram/RHS matrices. O(k²) cost.
5. **Zero-copy sparse interface** — `MappedSparseMatrix` wraps raw CSC pointers.
6. **IRLS integration** — Non-MSE losses use per-element reweighting within the same NNLS solver.
7. **Lazy CV masking** — Holdout determined by `SplitMix64::is_holdout()` in O(1).

---

## Complete Example: Multi-Layer NMF with Guides

```cpp
#include <FactorNet/graph/graph_all.hpp>
#include <FactorNet/guides/classifier_guide.hpp>

using namespace FactorNet;
using namespace FactorNet::graph;

int main() {
    // Load sparse scRNA-seq data (genes × cells)
    SparseMatrix<float> A = load_counts();
    std::vector<int> labels = load_labels();

    // Build graph: input → encoder(64) → bottleneck(10)
    InputNode<float, SparseMatrix<float>> inp(A, "X");
    NMFLayerNode<float> encoder(&inp, 64, "encoder");
    NMFLayerNode<float> bottleneck(&encoder, 10, "bottleneck");

    // Regularize encoder: sparse H, group-sparse W
    encoder.H_config.L1 = 0.01f;
    encoder.W_config.L21 = 0.1f;

    // Guide bottleneck H toward cell-type centroids
    guides::ClassifierGuide<float> guide;
    guide.lambda = 0.5f;
    guide.set_labels(labels, 10);
    bottleneck.H_config.guides.push_back(&guide);

    // Compile and fit
    FactorGraph<float> net({&inp}, &bottleneck);
    net.max_iter = 200;
    net.tol      = 1e-5f;
    net.seed     = 42;
    net.threads  = 16;
    net.loss.type = LossType::GP;
    net.compile();

    auto result = fit(net, A);

    std::printf("Converged: %s in %d iterations, loss: %g\n",
        result.converged ? "yes" : "no",
        result.total_iterations,
        static_cast<double>(result.total_loss));

    // Access per-layer factors
    const auto& enc = result["encoder"];
    const auto& bot = result["bottleneck"];
    // enc.W: genes × 64, enc.H: 64 × cells
    // bot.W: 64 × 10,    bot.H: 10 × cells (final embedding)
}
```

## Complete Example: Multi-Modal + Conditioning

```cpp
#include <FactorNet/graph/graph_all.hpp>

using namespace FactorNet;
using namespace FactorNet::graph;

int main() {
    SparseMatrix<float> RNA  = load_rna();    // genes × cells
    SparseMatrix<float> ATAC = load_atac();   // peaks × cells
    DenseMatrix<float>  Z    = load_batch();  // 3 × cells (batch covariates)

    // Multi-modal shared embedding
    InputNode<float, SparseMatrix<float>> rna_in(RNA, "RNA");
    InputNode<float, SparseMatrix<float>> atac_in(ATAC, "ATAC");
    SharedNode<float> shared({&rna_in, &atac_in});
    NMFLayerNode<float> joint(&shared, 30, "joint");

    // Condition on batch before the bottleneck layer
    ConditionNode<float> conditioned(&joint, Z);
    NMFLayerNode<float> bottleneck(&conditioned, 10, "corrected");

    FactorGraph<float> net({&rna_in, &atac_in}, &bottleneck);
    net.max_iter = 150;
    net.compile();

    auto result = fit(net, RNA);
    // result["joint"].W_splits["RNA"], ["ATAC"]
    // result["corrected"].H → 10 × cells (batch-corrected embedding)
}
```

## Performance Guidelines

1. **Sparse CSC format** — FactorNet expects `Eigen::SparseMatrix<Scalar, ColMajor>`. Convert with `.makeCompressed()`.
2. **Thread count** — Set `config.threads` or `OMP_NUM_THREADS`. The fused path parallelizes over columns.
3. **Solver selection** — `solver_mode = 1` (Cholesky+clip) is faster for small k (< 20). `solver_mode = 0` (CD) is better for large k with L1 > 0.
4. **Warm start** — Pass `W_init` / `H_init` from a previous run. The fused NNLS path uses warm-start across iterations.
5. **GPU threshold** — GPU benefits emerge for n > 10000 or nnz > 10M. Smaller problems are faster on CPU.
6. **Memory** — Peak: O(m·k + k·n + nnz) for sparse, O(m·n + m·k + k·n) for dense. GPU adds device copies.
7. **Batch size for streaming** — `InMemoryLoader`'s `chunk_size` controls columns per chunk. Larger = better GPU utilization.
