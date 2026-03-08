# FactorNet C++ API Reference

This reference documents all public types, structs, enums, and entry-point functions in the FactorNet C++ library. For usage examples and architectural context, see [README.md](README.md) and [ARCHITECTURE.md](ARCHITECTURE.md). The canonical in-source reference is [GUIDE.md](../../inst/include/FactorNet/GUIDE.md).

---

## Table of Contents

1. [Entry Points](#1-entry-points)
2. [NMFConfig](#2-nmfconfig)
3. [NMFResult](#3-nmfresult)
4. [FactorConfig](#4-factorconfig)
5. [LossConfig](#5-lossconfig)
6. [SVDConfig](#6-svdconfig)
7. [SVDResult](#7-svdresult)
8. [Graph Types](#8-graph-types)
9. [Enumerations](#9-enumerations)
10. [Type Aliases](#10-type-aliases)
11. [Primitives API](#11-primitives-api)
12. [RNG](#12-rng)
13. [Clustering](#13-clustering)
14. [I/O](#14-io)

---

## 1. Entry Points

### `nmf::fit()` — Unified NMF

**Header:** `#include <FactorNet/nmf/fit.hpp>`

```cpp
namespace FactorNet::nmf {

template<typename Scalar, typename MatrixType>
NMFResult<Scalar> fit(
    const MatrixType& A,                              // m × n input (sparse or dense)
    const NMFConfig<Scalar>& config,                  // all parameters
    const DenseMatrix<Scalar>* W_init = nullptr,      // optional m × k initialization
    const DenseMatrix<Scalar>* H_init = nullptr);     // optional k × n initialization

} // namespace FactorNet::nmf
```

Dispatches to CPU or GPU, standard or cross-validation, based on config fields. GPU failures fall back to CPU transparently.

### `nmf::find_optimal_rank()` — Rank Auto-Selection

**Header:** `#include <FactorNet/nmf/rank_cv.hpp>`

```cpp
namespace FactorNet::nmf {

template<typename Scalar, typename MatrixType>
RankSearchResult<Scalar> find_optimal_rank(
    const MatrixType& A,
    int k_init,                     // starting rank
    int max_k,                      // maximum rank to try
    int tolerance,                  // rank search tolerance
    const NMFConfig<Scalar>& config,
    bool verbose = false);

} // namespace FactorNet::nmf
```

### `svd::svd_gateway()` — Unified SVD

**Header:** `#include <FactorNet/svd/gateway.hpp>`

```cpp
namespace FactorNet::svd {

template<typename Scalar, typename MatrixType>
SVDResult<Scalar> svd_gateway(
    const MatrixType& A,
    const SVDConfig<Scalar>& config,
    const std::string& method = "auto");  // "auto","deflation","lanczos","irlba","krylov","randomized"

} // namespace FactorNet::svd
```

### `graph::fit()` — Graph Execution

**Header:** `#include <FactorNet/graph/fit.hpp>`

```cpp
namespace FactorNet::graph {

template<typename Scalar, typename MatrixType>
GraphResult<Scalar> fit(
    const FactorGraph<Scalar>& graph,
    const MatrixType& data);

} // namespace FactorNet::graph
```

Single-layer graphs delegate directly to `nmf::fit()` (full GPU/streaming support). Multi-layer graphs use outer ALS.

### `clustering::bipartition_gateway()` — Spectral Bipartition

**Header:** `#include <FactorNet/clustering/gateway.hpp>`

```cpp
namespace FactorNet::clustering {

template<typename Scalar>
BipartitionResult bipartition_gateway(
    const SparseMatrix<Scalar>& A,
    Scalar tol, int max_iter, bool nonneg,
    const std::vector<int>& sample_indices,
    uint32_t seed, bool verbose,
    bool compute_distance, Scalar diag_scale,
    int threads);

} // namespace FactorNet::clustering
```

### `clustering::dclust_gateway()` — Divisive Clustering

```cpp
namespace FactorNet::clustering {

template<typename Scalar>
DclustResult dclust_gateway(
    const SparseMatrix<Scalar>& A,
    Scalar tol, int max_iter, bool nonneg,
    const std::vector<int>& sample_indices,
    uint32_t seed, bool verbose, int threads);

} // namespace FactorNet::clustering
```

---

## 2. NMFConfig

**Header:** `#include <FactorNet/core/config.hpp>`

```cpp
template<typename Scalar>
struct NMFConfig {
    // ── Core ──
    int      rank       = 10;
    int      max_iter   = NMF_MAXIT;     // default: 100
    Scalar   tol        = NMF_TOL;       // default: 1e-4
    int      patience   = NMF_PATIENCE;  // default: 3
    uint32_t seed       = 0;
    bool     verbose    = false;

    // ── Resource control ──
    int         threads           = 0;       // 0 = all available
    int         max_gpus          = 0;       // 0 = all available
    std::string resource_override = "auto";  // "auto", "cpu", "gpu"

    // ── Per-factor configuration ──
    FactorConfig<Scalar> W;
    FactorConfig<Scalar> H;

    // ── Algorithm variants ──
    bool projective = false;   // H = diag(d)·W^T·A (no H NNLS)
    bool symmetric  = false;   // A ≈ W^T·diag(d)·W (square A)

    // ── Initialization ──
    int init_mode = 0;         // 0 = random uniform, 1 = Lanczos SVD seed

    // ── NNLS solver ──
    int    solver_mode = 1;    // 0 = CD, 1 = Cholesky+clip
    int    cd_max_iter = CD_MAXIT;
    Scalar cd_tol      = CD_TOL;
    Scalar cd_abs_tol  = CD_ABS_TOL;

    // ── Loss function (IRLS) ──
    LossConfig<Scalar> loss;          // type, huber_delta, robust_delta, etc.
    int    irls_max_iter = 5;
    Scalar irls_tol      = 1e-4;

    // ── Dispersion parameters (GP/NB/Gamma/InvGauss) ──
    DispersionMode gp_dispersion   = DispersionMode::PER_ROW;
    Scalar         gp_theta_init   = 0.1;
    Scalar         gp_theta_max    = 5.0;
    Scalar         gp_theta_min    = 0.0;
    Scalar         nb_size_init    = 10.0;
    Scalar         nb_size_max     = 1e6;
    Scalar         nb_size_min     = 0.01;
    Scalar         gamma_phi_init  = 1.0;
    Scalar         gamma_phi_max   = 1e4;
    Scalar         gamma_phi_min   = 1e-6;

    // ── Zero inflation ──
    ZIMode zi_mode     = ZIMode::ZI_NONE;
    int    zi_em_iters = 1;

    // ── Cross-validation ──
    Scalar   holdout_fraction = 0.0;   // > 0 enables CV
    uint32_t cv_seed          = 0;
    bool     mask_zeros       = false; // true = sparse CV (nonzeros only)
    int      cv_patience      = NMF_PATIENCE;

    // ── User-supplied mask ──
    const SparseMatrix<Scalar>* mask = nullptr;

    // ── Loss tracking ──
    bool track_loss_history = false;
    bool track_train_loss   = true;
    int  loss_every         = 1;      // compute every N iterations

    // ── Post-processing ──
    NormType norm_type  = NormType::L1;
    bool     sort_model = false;

    // ── Streaming ──
    std::string spz_path;             // non-empty → out-of-core SPZ file NMF

    // ── Query methods ──
    bool is_cv() const;               // holdout_fraction > 0
    bool requires_irls() const;       // non-MSE loss or robust_delta > 0
    bool has_guides() const;          // W or H have guides
    bool has_mask() const;            // mask != nullptr
};
```

---

## 3. NMFResult

**Header:** `#include <FactorNet/core/result.hpp>`

```cpp
template<typename Scalar>
struct NMFResult {
    // ── Factorization: A ≈ W · diag(d) · H ──
    DenseMatrix<Scalar> W;     // m × k
    DenseVector<Scalar> d;     // k
    DenseMatrix<Scalar> H;     // k × n

    // ── Convergence ──
    int    iterations = 0;
    bool   converged  = false;
    Scalar final_tol  = 0;

    // ── Loss ──
    Scalar train_loss     = 0;
    Scalar test_loss      = 0;     // CV only
    Scalar best_test_loss = 0;     // CV only
    int    best_iter      = 0;     // CV only

    // ── History ──
    std::vector<Scalar> loss_history;       // per-check train loss
    std::vector<Scalar> test_loss_history;  // per-check test loss (CV)
    FitHistory<Scalar>  history;            // per-iteration diagnostics

    // ── Distribution-specific ──
    DenseVector<Scalar> theta;      // GP dispersion (per-row or global)
    DenseVector<Scalar> dispersion; // Gamma/InvGauss φ
    DenseVector<Scalar> pi_row;     // ZI row dropout probability
    DenseVector<Scalar> pi_col;     // ZI column dropout probability

    // ── Diagnostics ──
    double              wall_time_ms = 0;
    ResourceDiagnostics diagnostics;   // CPU cores, GPUs used, plan, reason

    // ── Methods ──
    DenseMatrix<Scalar> reconstruct() const;              // W · diag(d) · H
    void normalize(NormType type = NormType::L1);         // rescale W, d, H
    void sort_by_d(bool descending = true);               // reorder factors by d
};
```

---

## 4. FactorConfig

**Header:** `#include <FactorNet/core/factor_config.hpp>`

Per-factor (W or H) regularization and constraints:

```cpp
template<typename Scalar>
struct FactorConfig {
    // ── Tier 1: O(k) per column ──
    Scalar L1 = 0;             // lasso (element-wise sparsity)
    Scalar L2 = 0;             // ridge (shrinkage)

    // ── Tier 2: O(k²·n), Gram-level ──
    Scalar L21     = 0;        // group sparsity (drives entire rows to zero)
    Scalar angular = 0;        // orthogonality penalty

    // ── Constraints ──
    bool   nonneg      = true; // non-negativity (x ≥ 0)
    Scalar upper_bound = 0;    // box constraint (0 = no upper bound)

    // ── Graph Laplacian ──
    const SparseMatrix<Scalar>* graph = nullptr;
    Scalar graph_lambda = 0;

    // ── Semi-supervised guides ──
    std::vector<guides::Guide<Scalar>*> guides;

    // ── Query methods ──
    bool has_tier2_features() const;    // angular || graph || L21
    bool has_any_reg() const;           // any non-zero regularizer
    bool has_guides_active() const;     // non-empty guides vector
};
```

---

## 5. LossConfig

**Header:** `#include <FactorNet/math/loss.hpp>`

```cpp
template<typename Scalar>
struct LossConfig {
    LossType type          = LossType::MSE;
    Scalar   huber_delta   = 1.0;     // Huber transition point
    Scalar   kl_epsilon    = 1e-10;   // KL numerical stability
    Scalar   robust_delta  = 0;       // Huber on Pearson residuals (0 = off)
    Scalar   power_param   = 1.5;     // Tweedie power parameter p
    Scalar   gp_blend      = 1.0;     // KL↔GP weight blending
    Scalar   dispersion    = 1.0;     // runtime dispersion parameter

    DispersionMode gp_dispersion = DispersionMode::PER_ROW;
    Scalar         gp_theta_init = 0.1;

    bool requires_irls() const;       // true if non-MSE or robust_delta > 0
};
```

---

## 6. SVDConfig

**Header:** `#include <FactorNet/core/svd_config.hpp>`

```cpp
template<typename Scalar>
struct SVDConfig {
    int      k_max    = 10;
    Scalar   tol      = 1e-5;
    int      max_iter = 100;
    bool     center   = false;    // subtract column means (PCA)
    bool     scale    = false;    // divide by column std (correlation PCA)
    uint32_t seed     = 0;
    int      threads  = 0;
    bool     verbose  = false;

    // ── Per-component regularization ──
    Scalar L1_u = 0, L1_v = 0;
    Scalar L2_u = 0, L2_v = 0;
    bool   nonneg_u = false, nonneg_v = false;
    Scalar upper_bound_u = 0, upper_bound_v = 0;

    // ── Gram-level (deflation/Krylov only) ──
    Scalar L21_u = 0, L21_v = 0;
    Scalar angular_u = 0, angular_v = 0;
    const SparseMatrix<Scalar>* graph_u = nullptr;
    const SparseMatrix<Scalar>* graph_v = nullptr;
};
```

---

## 7. SVDResult

**Header:** `#include <FactorNet/core/svd_result.hpp>`

```cpp
template<typename Scalar>
struct SVDResult {
    DenseMatrix<Scalar>    U;       // m × k (scores / left singular vectors)
    DenseVector<Scalar>    d;       // k (singular values)
    DenseMatrix<Scalar>    V;       // n × k (loadings / right singular vectors)
    int                    k_selected = 0;

    // ── PCA centering/scaling ──
    bool centered = false, scaled = false;
    DenseRowVector<Scalar> row_means;   // column means (if centered)
    DenseRowVector<Scalar> row_sds;     // column stds (if scaled)

    // ── CV trajectory ──
    std::vector<Scalar> test_loss_trajectory;
    std::vector<Scalar> train_loss_trajectory;
    std::vector<int>    iters_per_factor;

    // ── Diagnostics ──
    double frobenius_norm_sq = 0;
    double wall_time_ms      = 0;

    // ── Methods ──
    DenseMatrix<Scalar> reconstruct() const;
};
```

---

## 8. Graph Types

**Header:** `#include <FactorNet/graph/graph_all.hpp>` (umbrella)

### Node Types

All nodes derive from `Node<Scalar>`:

```cpp
template<typename Scalar> struct Node {
    NodeType type;
    std::string name;
    bool is_layer() const;       // NMF_LAYER or SVD_LAYER
    bool is_combinator() const;  // SHARED, CONCAT, ADD
};
```

| Node | Constructor | Purpose |
|------|------------|---------|
| `InputNode<Scalar, MatrixType>` | `(const MatrixType& A, name)` | Data source |
| `NMFLayerNode<Scalar>` | `(Node* input, int k, name)` | NMF factorization layer |
| `SVDLayerNode<Scalar>` | `(Node* input, int k, name)` | SVD factorization layer |
| `SharedNode<Scalar>` | `(vector<Node*> inputs)` | Multi-modal shared H |
| `ConcatNode<Scalar>` | `(vector<Node*> inputs)` | Row-concatenate H matrices |
| `AddNode<Scalar>` | `(vector<Node*> inputs)` | Element-wise add H (same k) |
| `ConditionNode<Scalar>` | `(Node* input, DenseMatrix Z)` | Append covariates to H |

### FactorGraph

```cpp
template<typename Scalar>
class FactorGraph {
public:
    // ── Global config ──
    int      max_iter;
    Scalar   tol;
    bool     verbose;
    int      threads;
    uint32_t seed;
    LossConfig<Scalar> loss;
    NormType norm_type;
    int      solver_mode;
    std::string resource;    // "auto", "cpu", "gpu"

    // ── Construction ──
    FactorGraph(std::vector<Node<Scalar>*> inputs, Node<Scalar>* output);

    // ── Compilation ──
    void compile();
    bool is_compiled() const;
    int  n_layers() const;
    const std::vector<Node<Scalar>*>& layers() const;  // topological order

    NMFConfig<Scalar> build_layer_config(const Node<Scalar>* layer, int maxit_override = -1) const;
};
```

### GraphResult / LayerResult

```cpp
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

template<typename Scalar>
struct GraphResult {
    std::map<std::string, LayerResult<Scalar>> layers;
    int    total_iterations;
    Scalar total_loss;
    bool   converged;
    const LayerResult<Scalar>& operator[](const std::string& name) const;
};
```

---

## 9. Enumerations

**Header:** `#include <FactorNet/math/loss.hpp>` and `#include <FactorNet/core/config.hpp>`

```cpp
enum class LossType {
    MSE, MAE, HUBER, KL, GP, NB, GAMMA, INVGAUSS, TWEEDIE
};

enum class DispersionMode {
    NONE, GLOBAL, PER_ROW, PER_COL
};

enum class ZIMode {
    ZI_NONE, ZI_ROW, ZI_COL, ZI_TWOWAY
};

enum class NormType {
    L1, L2, NONE
};

enum class NodeType {
    INPUT, NMF_LAYER, SVD_LAYER, SHARED, CONCAT, ADD, CONDITION
};

enum class ResourcePlan {
    CPU, GPU
};
```

---

## 10. Type Aliases

**Header:** `#include <FactorNet/core/types.hpp>`

```cpp
template<typename Scalar>
using DenseMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

template<typename Scalar>
using DenseVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

template<typename Scalar>
using DenseRowVector = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;

template<typename Scalar>
using SparseMatrix = Eigen::SparseMatrix<Scalar, Eigen::ColMajor>;

template<typename Scalar>
using MappedSparseMatrix = Eigen::Map<const SparseMatrix<Scalar>>;
```

---

## 11. Primitives API

**Header:** `#include <FactorNet/primitives/primitives.hpp>`

Four operations abstracted over `Resource` (CPU or GPU):

```cpp
namespace FactorNet::primitives {

// Gram matrix: G = H · H^T  (k × k, symmetric)
template<typename Resource, typename Scalar>
void gram(const DenseMatrix<Scalar>& H, DenseMatrix<Scalar>& G);

// Right-hand side: B = H · A  (k × m or k × n)
template<typename Resource, typename Scalar, typename MatrixType>
void rhs(const MatrixType& A, const DenseMatrix<Scalar>& H, DenseMatrix<Scalar>& B);

// Batch NNLS: solve G·x = b for each column of B, with x ≥ 0
template<typename Resource, typename Scalar>
void nnls_batch(const DenseMatrix<Scalar>& G, DenseMatrix<Scalar>& B,
                DenseMatrix<Scalar>& X,
                int cd_maxit, Scalar cd_tol, Scalar cd_abs_tol,
                Scalar L1, bool nonneg, Scalar upper_bound);

// Trace: tr(A^T · A) — precomputed constant for Gram-trick loss
template<typename Resource, typename Scalar, typename MatrixType>
Scalar trace_AtA(const MatrixType& A);

} // namespace FactorNet::primitives
```

### CPU Specializations (`primitives/cpu/`)

| File | Notes |
|------|-------|
| `gram.hpp` | `selfadjointView::rankUpdate` for 2× BLAS speedup |
| `rhs.hpp` | OpenMP parallel over sparse columns |
| `nnls_batch.hpp` | Coordinate descent with warm start, L1 soft-thresholding |
| `nnls_batch_irls.hpp` | Weighted CD for non-MSE losses |
| `cholesky_clip.hpp` | Cholesky factorization + clip to bounds |
| `fused_nnls.hpp` | Single-pass RHS+NNLS (sparse MSE only) |
| `loss.hpp` | Gram-trick and explicit loss computation |
| `data_accessor.hpp` | CSC column iterator for sparse matrices |

### GPU Specializations (`primitives/gpu/`)

| File | Notes |
|------|-------|
| `gram.cuh` | cuBLAS SSYRK |
| `rhs.cuh` | cuSPARSE SpMM (sparse) or cuBLAS SGEMM (dense) |
| `nnls_batch.cuh` | Custom CUDA kernel, 1 thread-block per column |
| `nnls_batch_irls.cuh` | Weighted NNLS on GPU |
| `nnls_batch_zi_irls.cuh` | Zero-inflation aware IRLS on GPU |
| `cholesky_clip.cuh` | GPU Cholesky + clip |
| `loss.cuh` | Tree-reduction loss kernel |
| `context.cuh` | GPU context (streams, cuBLAS/cuSPARSE handles) |

---

## 12. RNG

**Header:** `#include <FactorNet/rng/rng.hpp>`

```cpp
class SplitMix64 {
public:
    explicit SplitMix64(uint64_t seed) noexcept;
    uint64_t next() noexcept;

    template<typename T> T uniform() noexcept;              // [0, 1)
    template<typename T> void fill_uniform(T* ptr, int n) noexcept;

    // Thread-safe, GPU-compatible positional hash
    static uint64_t hash(uint64_t seed, int64_t i, int64_t j) noexcept;
    static bool is_holdout(uint64_t seed, int64_t i, int64_t j, double frac) noexcept;
};
```

`hash()` and `is_holdout()` are `__host__ __device__` when compiled with CUDA. They enable lazy CV mask evaluation — O(1) per entry, no mask matrix materialized.

---

## 13. Clustering

**Header:** `#include <FactorNet/clustering/gateway.hpp>`

### BipartitionResult

```cpp
struct BipartitionResult {
    std::vector<int> assignments;    // 0/1 per sample
    DenseMatrix<float> cluster_centers;
    float inter_cluster_distance;
};
```

### DclustResult

```cpp
struct DclustResult {
    std::vector<int> samples;  // 0-indexed sample indices
    std::vector<int> id;       // numeric cluster IDs
};
```

**Note:** Both return 0-indexed values. The R bridge converts to 1-indexed.

---

## 14. I/O

**Header:** `#include <FactorNet/io/loader.hpp>`

### DataLoader Interface

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

### Implementations

| Class | Header | Purpose |
|-------|--------|---------|
| `InMemoryLoader<Scalar>` | `io/in_memory.hpp` | Zero-copy chunked views of in-memory sparse matrix |
| `SpzLoader<Scalar>` | `io/spz_loader.hpp` | Streaming decompression of `.spz` files |

### Chunk

```cpp
template<typename Scalar>
struct Chunk {
    SparseMatrix<Scalar> data;
    int col_offset;      // starting column in the full matrix
    int n_cols;          // number of columns in this chunk
};
```

### Usage with Streaming NMF

```cpp
SpzLoader<double> loader("data.spz");
NMFConfig<double> config;
config.rank = 20;

auto result = nmf::fit_streaming(loader, config);
```
