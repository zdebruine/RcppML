# FactorNet C++ Library

**FactorNet** is a header-only C++ library for building and executing factorization networks — directed acyclic graphs of NMF and SVD layers connected by combinators. It is the computational backbone of the RcppML R package.

## Features

- **Non-negative Matrix Factorization (NMF)** with ALS updates, W·diag(d)·H model
- **Singular Value Decomposition** via 5 methods (Deflation, Lanczos, IRLBA, Krylov, Randomized)
- **Graph composition** — wire NMF/SVD layers into multi-layer, multi-modal pipelines
- **9 loss functions** with IRLS reweighting (MSE, MAE, Huber, KL, GP, NB, Gamma, InvGauss, Tweedie)
- **Regularization** — L1, L2, L21, angular, graph Laplacian, box constraints
- **Semi-supervised guides** — classifier and external target guides
- **Cross-validation** with lazy speckled mask (O(1) holdout queries, no mask matrix)
- **Zero-inflation** (ZIGP/ZINB) via EM
- **GPU acceleration** (CUDA) with automatic CPU fallback
- **Streaming NMF** from SPZ compressed files (out-of-core)
- **Spectral clustering** — bipartitioning, divisive clustering

## Dependencies

| Library | Required | Purpose |
|---------|----------|---------|
| **Eigen 3.4+** | Yes | Dense and sparse linear algebra |
| **OpenMP** | Recommended | Parallel NNLS, RHS, loss computation |
| **CUDA toolkit** | Optional | GPU acceleration (cuBLAS, cuSPARSE) |

## Quick Start (Standalone C++)

```cpp
#include <FactorNet.h>
#include <Eigen/Sparse>

using namespace FactorNet;

int main() {
    // Load or construct a sparse matrix (CSC format)
    Eigen::SparseMatrix<float, Eigen::ColMajor> A = load_your_data();

    // Configure NMF
    NMFConfig<float> config;
    config.rank     = 20;
    config.max_iter = 100;
    config.tol      = 1e-5f;
    config.seed     = 42;
    config.H.L1     = 0.01f;  // sparse H

    // Run
    NMFResult<float> result = nmf::fit(A, config);

    // Access factors: A ≈ W · diag(d) · H
    auto& W = result.W;   // m × k
    auto& d = result.d;   // k
    auto& H = result.H;   // k × n

    std::cout << "Converged: " << result.converged
              << " in " << result.iterations << " iterations"
              << " (" << result.wall_time_ms << " ms)\n";
    return 0;
}
```

### Build with CMake

```cmake
find_package(Eigen3 REQUIRED)
add_executable(my_app main.cpp)
target_include_directories(my_app PRIVATE path/to/inst/include)
target_link_libraries(my_app Eigen3::Eigen)

# Optional GPU support
find_package(CUDA)
if(CUDA_FOUND)
    target_compile_definitions(my_app PRIVATE FACTORNET_HAS_GPU)
endif()
```

## Quick Start (Graph API)

```cpp
#include <FactorNet/graph/graph_all.hpp>

using namespace FactorNet;
using namespace FactorNet::graph;

// Single-layer NMF (equivalent to nmf::fit)
InputNode<float, SparseMatrix<float>> inp(A, "X");
NMFLayerNode<float> layer(&inp, 20, "factors");
layer.H_config.L1 = 0.01f;

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

### Multi-Modal Shared Factorization

```cpp
// Jointly factorize RNA + ATAC sharing the same cell embedding
InputNode<float, SparseMatrix<float>> rna_in(RNA, "RNA");
InputNode<float, SparseMatrix<float>> atac_in(ATAC, "ATAC");
SharedNode<float> shared({&rna_in, &atac_in});
NMFLayerNode<float> joint(&shared, 20, "joint");

FactorGraph<float> net({&rna_in, &atac_in}, &joint);
net.compile();

auto result = fit(net, RNA);
// result["joint"].H                → 20 × n (shared embedding)
// result["joint"].W_splits["RNA"]  → genes × 20
// result["joint"].W_splits["ATAC"] → peaks × 20
```

## Cross-Validation for Rank Selection

```cpp
NMFConfig<float> config;
config.rank              = 20;
config.holdout_fraction  = 0.1f;
config.cv_seed           = 42;
config.mask_zeros        = true;  // recommendation-style CV
config.max_iter          = 100;

auto result = nmf::fit(A, config);
// result.test_loss       — final test MSE
// result.best_test_loss  — best test MSE across iterations
// result.best_iter       — iteration that achieved best test loss
```

## Non-Gaussian Losses

```cpp
NMFConfig<float> config;
config.rank           = 15;
config.loss.type      = LossType::GP;       // Generalized Poisson
config.gp_dispersion  = DispersionMode::PER_ROW;
config.gp_theta_init  = 0.1f;
config.irls_max_iter  = 5;

// Zero-inflation
config.zi_mode     = ZIMode::ZI_ROW;
config.zi_em_iters = 3;

auto result = nmf::fit(A, config);
// result.theta   — per-row dispersion estimates
// result.pi_row  — per-row dropout probabilities
```

## Documentation

| Document | Contents |
|----------|----------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Dispatch flow, memory model, template hierarchy |
| [API_REFERENCE.md](API_REFERENCE.md) | All structs, enums, and function signatures |
| [gpu/README.md](gpu/README.md) | GPU kernel inventory, build instructions, performance |
| [io/README.md](io/README.md) | SPZ streaming API, DataLoader interface |
| [GUIDE.md](../../inst/include/FactorNet/GUIDE.md) | Comprehensive in-source API reference (canonical) |
| [algorithms/](algorithms/) | Mathematical documentation per algorithm family |
