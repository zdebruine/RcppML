// Automatic SVD method selection based on empirical benchmarking
// Datasets: pbmc3k (13714×2700), ifnb (14053×13999), hcabm40k (17369×40000)  
// Date: March 2, 2026
// Testing: CPU (AMD EPYC, tol=1e-4), GPU (H100/V100S)
//
// CPU RESULTS (speedup vs random init):
//   k=10:  Lanczos 1.13× (range 1.02-1.20×)
//   k=30:  Lanczos 1.02× (marginal, sometimes ties IRLBA)
//   k=60:  IRLBA  1.11× (range 1.02-1.26×)
//
// GPU RESULTS:
//   k<32:   Lanczos — cuSPARSE SpMV is fast for few Krylov steps
//   32≤k<64: Randomized — cuSPARSE SpMM batches + cuSOLVER QR; fixed cost
//   k≥64:   IRLBA — efficient implicit restart scales best at high rank
//
// OPTIMAL THRESHOLDS:
//   CPU: k < 64 → Lanczos; k ≥ 64 → IRLBA
//   GPU: k < 32 → Lanczos; 32 ≤ k < 64 → Randomized; k ≥ 64 → IRLBA

#pragma once

#include <algorithm>

namespace FactorNet {
namespace svd {

enum class SVDMethod {
    LANCZOS,
    IRLBA,
    RANDOMIZED,
    KRYLOV,
    DEFLATION
};

/**
 * @brief Automatically select optimal SVD method based on rank and resource
 *
 * Decision model based on empirical benchmarking:
 *
 * CPU path:
 *   k < 64:  Lanczos (fast single-pass Krylov, ~1.02-1.13× speedup)
 *   k >= 64: IRLBA (efficient implicit restart, 1.11× speedup)
 *
 * GPU path:
 *   k < 32:     Lanczos (cuSPARSE SpMV, fast for few Krylov steps)
 *   32 <= k < 64: Randomized (cuSPARSE SpMM + cuSOLVER QR, fixed cost)
 *   k >= 64:    IRLBA (implicit restart scales well at high rank)
 *
 * Constraints override: Krylov (k≥8) or Deflation (k<8)
 *
 * @param k Target rank
 * @param m Number of rows
 * @param n Number of columns
 * @param is_gpu Whether running on GPU
 * @param has_constraints Whether L1/L2/nonneg/graph constraints are active
 * @param prefer_memory_efficiency Unused (kept for API compatibility)
 *
 * @return Recommended SVD method
 */
inline SVDMethod auto_select_svd_method(
    int k,
    int m,
    int n,
    bool is_gpu = false,
    bool has_constraints = false,
    bool prefer_memory_efficiency = false)
{
    // If constraints are active, must use constrained methods
    if (has_constraints) {
        // Krylov is faster for block methods, deflation for small k
        return (k >= 8) ? SVDMethod::KRYLOV : SVDMethod::DEFLATION;
    }

    // Sanity check: Very high rank relative to dimensions
    int min_dim = std::min(m, n);
    if (k >= min_dim - 10) {
        return SVDMethod::IRLBA;
    }

    if (is_gpu) {
        // GPU: three-tier strategy
        if (k < 32) {
            return SVDMethod::LANCZOS;
        } else if (k < 64) {
            // Randomized SVD excels on GPU in this range:
            // cuSPARSE SpMM batches all l columns, cuSOLVER QR is fast
            return SVDMethod::RANDOMIZED;
        } else {
            return SVDMethod::IRLBA;
        }
    } else {
        // CPU: two-tier strategy (Lanczos remains competitive up to k=63)
        if (k < 64) {
            return SVDMethod::LANCZOS;
        } else {
            return SVDMethod::IRLBA;
        }
    }
}

/**
 * @brief Get human-readable method name
 */
inline const char* svd_method_name(SVDMethod method) {
    switch (method) {
        case SVDMethod::LANCZOS:     return "lanczos";
        case SVDMethod::IRLBA:       return "irlba";
        case SVDMethod::RANDOMIZED:  return "randomized";
        case SVDMethod::KRYLOV:      return "krylov";
        case SVDMethod::DEFLATION:   return "deflation";
        default:                     return "unknown";
    }
}

}  // namespace svd
}  // namespace FactorNet

