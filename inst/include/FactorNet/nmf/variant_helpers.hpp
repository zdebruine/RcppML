// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file nmf/variant_helpers.hpp
 * @brief Shared helpers for NMF algorithm variants and feature application
 *
 * This file centralizes:
 *   1. Algorithm variant classification (standard / projective / symmetric)
 *   2. Feature application sequences (H-side and W-side)
 *   3. Host-matrix operations for projective and symmetric NMF
 *
 * Every backend (CPU, GPU, CV, streaming) includes this file and uses
 * these helpers. Algorithm variant logic lives here — NOT duplicated across
 * backend iteration loops.
 *
 * Design principle: All helpers operate on HOST Eigen matrices. GPU backends
 * download device matrices to host, call these helpers, then upload back.
 * This ensures feature/variant logic is written exactly ONCE.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/core/config.hpp>
#include <FactorNet/core/factor_config.hpp>
#include <FactorNet/core/types.hpp>

// Feature headers
#include <FactorNet/features/sparsity.hpp>
#include <FactorNet/features/angular.hpp>
#include <FactorNet/features/L21.hpp>
#include <FactorNet/features/graph_reg.hpp>
#include <FactorNet/features/bounds.hpp>

#include <Eigen/Eigenvalues>  // SelfAdjointEigenSolver for PROJ_ADV

namespace FactorNet {
namespace nmf {
namespace variant {

// ========================================================================
// 1. Algorithm variant classification
// ========================================================================

/// H-update mode: determines what computation to perform for H
enum class HUpdateMode {
    STANDARD,        ///< Full Gram → RHS → features → NNLS → normalize
    PROJECTIVE,      ///< H = diag(d) · W^T · A, then normalize (no NNLS)
    SYMMETRIC_SKIP   ///< No-op; H will be set to W^T after W update
};

/// W-update mode: determines what computation to perform for W
enum class WUpdateMode {
    STANDARD,        ///< Full Gram(H) → RHS(H,A^T) → features → NNLS → normalize
    SYMMETRIC        ///< Gram(W) → RHS(W,A) → features → NNLS → normalize, then H = W^T
};

/// Classify the H-update mode from config
template<typename Scalar>
inline HUpdateMode h_update_mode(const NMFConfig<Scalar>& config) {
    if (config.projective)  return HUpdateMode::PROJECTIVE;
    if (config.symmetric)   return HUpdateMode::SYMMETRIC_SKIP;
    return HUpdateMode::STANDARD;
}

/// Classify the W-update mode from config
template<typename Scalar>
inline WUpdateMode w_update_mode(const NMFConfig<Scalar>& config) {
    if (config.symmetric)   return WUpdateMode::SYMMETRIC;
    return WUpdateMode::STANDARD;
}

// ========================================================================
// 2. Factor-agnostic feature application (unified versions)
// ========================================================================

/**
 * @brief Apply all features to Gram matrix G and RHS matrix B for any factor
 *
 * Factor-agnostic: works for W, H, or any future factor in a multi-layer
 * FactorNet.  Replaces the duplicated apply_h_features / apply_w_features.
 */
template<typename Scalar>
inline void apply_features(
    DenseMatrix<Scalar>& G,
    DenseMatrix<Scalar>& B,
    DenseMatrix<Scalar>& Factor,
    const FactorConfig<Scalar>& fc)
{
    features::apply_L1_L2(G, B, fc.L1, fc.L2);

    // Angular is now applied post-NNLS via apply_angular_posthoc,
    // not as a Gram modification. See angular.hpp.

    if (fc.graph)
        features::apply_graph_reg(G, *fc.graph, Factor, fc.graph_lambda);

    if (fc.L21 > 0)
        features::apply_L21(G, Factor, fc.L21);

    if (fc.target && fc.target_lambda != 0) {
        if (fc.target_lambda > 0) {
            // ENRICHMENT: ridge toward target.
            // G.diagonal() += λ;  B += λ * T
            G.diagonal().array() += fc.target_lambda;
            B.noalias() += fc.target_lambda * (*fc.target);
        } else {
            // PROJ_ADV batch removal: subtract trace-scaled target covariance
            // from Gram, then eigendecompose and clip negative eigenvalues.
            // B is NOT modified — we only project out target directions from
            // the system matrix, preserving the reconstruction objective.
            //
            // Trace-scaling: target_gram has entries ~O(1) while G has entries
            // ~O(||W||²).  Scale target_gram so its trace matches G's trace,
            // making λ interpretable as the fraction of target energy to remove.
            const Scalar abs_lambda = std::abs(fc.target_lambda);
            const Scalar trace_G  = G.trace();
            const Scalar trace_GT = fc.target_gram.trace();
            const Scalar scale = (trace_GT > static_cast<Scalar>(1e-10))
                               ? (trace_G / trace_GT) : static_cast<Scalar>(0);
            G.noalias() -= abs_lambda * scale * fc.target_gram;

            // Eigendecompose G (symmetric k×k) and clip negative eigenvalues
            Eigen::SelfAdjointEigenSolver<DenseMatrix<Scalar>> eig(G);
            auto evals = eig.eigenvalues();
            constexpr Scalar eps = static_cast<Scalar>(1e-8);
            bool has_neg = false;
            for (int i = 0; i < evals.size(); ++i) {
                if (evals(i) < eps) {
                    evals(i) = eps;
                    has_neg = true;
                }
            }
            if (has_neg) {
                G.noalias() = eig.eigenvectors() * evals.asDiagonal()
                            * eig.eigenvectors().transpose();
            }
        }
    }
}

// ========================================================================
// 2a. Per-column penalty terms for the CPU fit paths (fit_cpu.hpp)
// ========================================================================

/**
 * @brief Penalty terms that differ from column to column
 *
 * Built once per half-step by build_column_penalty() and consumed by every
 * CPU solve path (fused, standard, masked, IRLS), so that each path applies
 * the same penalties.
 */
template<typename Scalar>
struct ColumnPenalty {
    /// k × ncols term added to the right-hand side (target, graph neighbors).
    /// Empty when unused.
    DenseMatrix<Scalar> B_add;

    /// Per-column ridge from the graph Laplacian diagonal (λ·L_jj).
    /// Empty when unused.
    DenseVector<Scalar> ridge;

    /// ‖previous column‖₂ on the scale of the variable (for L21).
    /// Empty when unused.
    DenseVector<Scalar> group_norm;

    /// Ridge shared by all columns from the target penalty (λ_target > 0).
    Scalar shared_ridge = 0;

    bool has_B() const noexcept { return B_add.size() > 0; }
    bool has_ridge() const noexcept { return ridge.size() > 0; }
    bool has_group() const noexcept { return group_norm.size() > 0; }

    const DenseMatrix<Scalar>* B_ptr() const noexcept {
        return has_B() ? &B_add : nullptr;
    }
    const Scalar* ridge_ptr() const noexcept {
        return has_ridge() ? ridge.data() : nullptr;
    }
    const Scalar* group_ptr() const noexcept {
        return has_group() ? group_norm.data() : nullptr;
    }
};

/**
 * @brief PROJ_ADV batch removal (negative target_lambda) on a Gram matrix
 *
 * Subtracts the trace-scaled target covariance from G, then clips negative
 * eigenvalues.  Same operation as the negative branch of apply_features().
 */
template<typename Scalar>
inline void apply_proj_adv(DenseMatrix<Scalar>& G,
                           const FactorConfig<Scalar>& fc)
{
    const Scalar abs_lambda = std::abs(fc.target_lambda);
    const Scalar trace_G  = G.trace();
    const Scalar trace_GT = fc.target_gram.trace();
    const Scalar scale = (trace_GT > static_cast<Scalar>(1e-10))
                       ? (trace_G / trace_GT) : static_cast<Scalar>(0);
    G.noalias() -= abs_lambda * scale * fc.target_gram;

    Eigen::SelfAdjointEigenSolver<DenseMatrix<Scalar>> eig(G);
    DenseVector<Scalar> evals = eig.eigenvalues();
    const Scalar eps = static_cast<Scalar>(1e-8);
    bool has_neg = false;
    for (int i = 0; i < evals.size(); ++i) {
        if (evals(i) < eps) {
            evals(i) = eps;
            has_neg = true;
        }
    }
    if (has_neg) {
        const DenseMatrix<Scalar> V = eig.eigenvectors();
        G.noalias() = V * evals.asDiagonal() * V.transpose();
    }
}

/**
 * @brief Build all penalty terms for one half-step (any CPU path)
 *
 * Shared terms are applied to G; column-specific terms go into `cp`.
 * L1, bounds and non-negativity are handled inside the solvers.
 *
 *   L2      : G.diagonal() += L2                       (if add_L2)
 *   target  : λ > 0:  G.diagonal() += λ,  B_add += λ·T
 *             λ < 0:  PROJ_ADV on G                    (if allow_proj_adv)
 *   graph   : ridge_j = λ·L_jj,  B_add_j += λ·Σ_{i≠j}(−L_ij)·s_i
 *   L21     : group_norm_j = ‖s_j‖₂
 *
 * @param G       Gram matrix (k × k) to receive the shared terms
 * @param S       Previous iterate of the factor being solved, k × ncols, on
 *                the scale of the variable (i.e. with d applied)
 * @param fc      Factor configuration
 * @param cp      Output column penalty (fully overwritten)
 * @param add_L2  Add the L2 ridge to G (false when the solver adds it)
 * @param allow_proj_adv  Apply the negative-target branch to G
 */
template<typename Scalar>
inline void build_column_penalty(
    DenseMatrix<Scalar>& G,
    const DenseMatrix<Scalar>& S,
    const FactorConfig<Scalar>& fc,
    ColumnPenalty<Scalar>& cp,
    bool add_L2 = true,
    bool allow_proj_adv = true)
{
    cp.B_add.resize(0, 0);
    cp.ridge.resize(0);
    cp.group_norm.resize(0);
    cp.shared_ridge = 0;

    if (add_L2 && fc.L2 > 0) G.diagonal().array() += fc.L2;

    const bool pos_target = fc.target && fc.target_lambda > 0;
    const bool has_graph  = fc.graph && fc.graph_lambda > 0;

    if (pos_target || has_graph) {
        cp.B_add.resize(S.rows(), S.cols());
        cp.B_add.setZero();
    }

    if (pos_target) {
        G.diagonal().array() += fc.target_lambda;
        cp.shared_ridge = fc.target_lambda;
        cp.B_add.noalias() += fc.target_lambda * (*fc.target);
    } else if (fc.target && fc.target_lambda < 0 && allow_proj_adv) {
        apply_proj_adv(G, fc);
    }

    if (has_graph)
        features::graph_column_terms(*fc.graph, S, fc.graph_lambda,
                                     cp.ridge, cp.B_add);

    if (fc.L21 > 0)
        cp.group_norm = S.colwise().norm().transpose();
}

/**
 * @brief Apply Tier-2 features only (angular, graph_reg, L21)
 *
 * Factor-agnostic version for GPU backends that handle L1/L2 separately.
 */
template<typename Scalar>
inline void apply_tier2_features(
    DenseMatrix<Scalar>& G,
    DenseMatrix<Scalar>& Factor,
    const FactorConfig<Scalar>& fc)
{
    // Angular is now applied post-NNLS via apply_angular_posthoc.

    if (fc.graph)
        features::apply_graph_reg(G, *fc.graph, Factor, fc.graph_lambda);

    if (fc.L21 > 0)
        features::apply_L21(G, Factor, fc.L21);
}

/**
 * @brief Apply CV features (L2 + Tier 2, no L1)
 *
 * Factor-agnostic version. L1 is applied inside the per-column CD solver.
 */
template<typename Scalar>
inline void apply_cv_features(
    DenseMatrix<Scalar>& G,
    const DenseMatrix<Scalar>& Factor,
    const FactorConfig<Scalar>& fc)
{
    if (fc.L2 > 0)
        G.diagonal().array() += fc.L2;

    // Angular is now applied post-NNLS via apply_angular_posthoc.

    if (fc.graph)
        features::apply_graph_reg(G, *fc.graph, Factor, fc.graph_lambda);

    if (fc.L21 > 0)
        features::apply_L21(G, Factor, fc.L21);
}

// ========================================================================
// 2b. Legacy H/W-specific wrappers (delegate to unified versions)
// ========================================================================

template<typename Scalar>
inline void apply_h_features(
    DenseMatrix<Scalar>& G,
    DenseMatrix<Scalar>& B,
    DenseMatrix<Scalar>& H,
    const NMFConfig<Scalar>& config)
{
    apply_features(G, B, H, config.H);
}

template<typename Scalar>
inline void apply_w_features(
    DenseMatrix<Scalar>& G,
    DenseMatrix<Scalar>& B,
    DenseMatrix<Scalar>& W_T,
    const NMFConfig<Scalar>& config)
{
    apply_features(G, B, W_T, config.W);
}

template<typename Scalar>
inline bool has_h_tier2_features(const NMFConfig<Scalar>& config) {
    return config.H.has_tier2_features();
}

template<typename Scalar>
inline void apply_h_tier2_features(
    DenseMatrix<Scalar>& G,
    DenseMatrix<Scalar>& H,
    const NMFConfig<Scalar>& config)
{
    apply_tier2_features(G, H, config.H);
}

template<typename Scalar>
inline bool has_w_tier2_features(const NMFConfig<Scalar>& config) {
    return config.W.has_tier2_features();
}

template<typename Scalar>
inline void apply_w_tier2_features(
    DenseMatrix<Scalar>& G,
    DenseMatrix<Scalar>& W_T,
    const NMFConfig<Scalar>& config)
{
    apply_tier2_features(G, W_T, config.W);
}

template<typename Scalar>
inline void apply_h_cv_features(
    DenseMatrix<Scalar>& G,
    const DenseMatrix<Scalar>& H,
    const NMFConfig<Scalar>& config)
{
    apply_cv_features(G, H, config.H);
}

template<typename Scalar>
inline void apply_w_cv_features(
    DenseMatrix<Scalar>& G,
    const DenseMatrix<Scalar>& W_T,
    const NMFConfig<Scalar>& config)
{
    apply_cv_features(G, W_T, config.W);
}

// ========================================================================
// 3. Host-matrix operations for variant branches
// ========================================================================

/**
 * @brief Apply diagonal scaling to factor rows: factor[i,:] *= d[i]
 *
 * Used to form W·diag(d) before projective RHS computation.
 */
template<typename Scalar>
inline void apply_scaling(
    DenseMatrix<Scalar>& factor,
    const DenseVector<Scalar>& d)
{
    for (int i = 0; i < d.size(); ++i)
        factor.row(i) *= d(i);
}

/**
 * @brief Extract row norms into d and normalize rows
 *
 * After NNLS solve: d[i] = norm(X[i,:]), X[i,:] /= d[i].
 * Used in all backends after NNLS or projective computation.
 * Adds 1e-15 epsilon to prevent component death (matches legacy behavior).
 */
template<typename Scalar>
inline void extract_scaling(
    DenseMatrix<Scalar>& X,
    DenseVector<Scalar>& d,
    NormType norm_type)
{
    if (norm_type == NormType::None) {
        d.setOnes();
        return;
    }
    if (norm_type == NormType::L1) {
        d = X.array().abs().rowwise().sum().matrix();
    } else {
        d = X.rowwise().norm();
    }
    d.array() += static_cast<Scalar>(1e-15);
    for (int i = 0; i < d.size(); ++i) {
        X.row(i) /= d(i);
    }
}

/**
 * @brief Projective H-update on host matrices: H = diag(d) · W^T · A
 *
 * For projective NMF, H is deterministically derived from W and A.
 * No NNLS solve is needed. Non-negativity of H is automatic when W ≥ 0 and A ≥ 0.
 *
 * @tparam RhsFn  Callable: void(const DenseMatrix<Scalar>&, DenseMatrix<Scalar>&)
 *                that computes B = factor · A (the forward RHS operation)
 * @param W_T       Factor matrix (k × m), will be temporarily scaled
 * @param H         Output matrix (k × n), overwritten with projective result
 * @param d         Scaling vector (k), updated with new norms
 * @param rhs_fn    Backend-specific RHS computation
 * @param norm_type Normalization type for extract_scaling
 */
template<typename Scalar, typename RhsFn>
inline void projective_h_update(
    DenseMatrix<Scalar>& W_T,
    DenseMatrix<Scalar>& H,
    DenseVector<Scalar>& d,
    RhsFn rhs_fn,
    NormType norm_type)
{
    // Scale W_T by d: W_Td[i,:] = d[i] * W_T[i,:]
    DenseMatrix<Scalar> W_Td = W_T;
    apply_scaling(W_Td, d);

    // H = W_Td · A (via the backend-specific RHS function)
    rhs_fn(W_Td, H);

    // Normalize H.  The row norms of W_Td·A are NOT the scale of the model:
    // they carry an extra factor ‖w_iᵀA‖ and, written into d, would compound
    // every iteration (d is applied to W_T again above), overflowing within
    // a few iterations whenever something reads d before the W-update
    // resets it — the graph and L21 penalties do.  d therefore keeps the
    // scale left by the last W normalization, which is the scale of the
    // variable solved in the following W-update.
    if (norm_type == NormType::None) {
        d.setOnes();
    } else {
        DenseVector<Scalar> h_norms(d.size());
        extract_scaling(H, h_norms, norm_type);
    }
}

/**
 * @brief Symmetric W-update post-processing: enforce H = W^T
 *
 * After the W-update NNLS solve and normalization, copy W_T to H.
 * This enforces the symmetric constraint A ≈ W · diag(d) · W^T.
 */
template<typename Scalar>
inline void symmetric_enforce_h(
    DenseMatrix<Scalar>& H,
    const DenseMatrix<Scalar>& W_T)
{
    H = W_T;
}

// ========================================================================
// 4. Validation helpers
// ========================================================================

/**
 * @brief Validate algorithm variant compatibility with backend features
 *
 * Call at the start of each backend's fit function to catch unsupported
 * combinations early with clear error messages.
 */
template<typename Scalar>
inline void validate_variant_config(
    const NMFConfig<Scalar>& config,
    const char* backend_name)
{
    if (config.projective && config.symmetric) {
        throw std::invalid_argument(
            std::string(backend_name) +
            ": projective and symmetric cannot both be true");
    }
}

/**
 * @brief Check if projective/symmetric are compatible with CV
 *
 * Projective NMF in CV mode: the H-update doesn't use NNLS, so per-column
 * Gram correction is irrelevant for H. The W-update still uses standard CV.
 * This is supported.
 *
 * Symmetric NMF in CV mode: only W is updated. CV holds out entries from
 * both sides, but only the W update uses held-out data. This is supported
 * but the test loss semantics differ (reconstruction is W·d·W^T).
 */
template<typename Scalar>
inline void validate_variant_cv(
    const NMFConfig<Scalar>& config,
    const char* backend_name)
{
    validate_variant_config(config, backend_name);
    // Both projective and symmetric are compatible with CV — no error needed
}

}  // namespace variant
}  // namespace nmf
}  // namespace FactorNet

