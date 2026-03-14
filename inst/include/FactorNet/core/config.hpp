// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file config.hpp
 * @brief Unified NMF configuration — one struct for all backends
 *
 * Merges the following prior config types into a single NMFConfig<Scalar>:
 *   - nmf::NMFConfig<Scalar>       (nmf/types.hpp)          — 12 fields
 *   - NMFConfig<Scalar> (nmf/fit_cpu.hpp) — ~40 fields
 *   - nmf::CVConfig<Scalar>         (nmf/cv_types.hpp)       — ~20 fields
 *   - solvers::NNLSConfig           (solvers/nnls/config.hpp) — ~15 fields
 *
 * Design principles:
 *   - No backend field — the config describes WHAT to compute, never HOW.
 *   - All regularization parameters have per-factor (W/H) variants.
 *   - Feature dispatch is handled by runtime if-checks in the algorithm layer.
 *   - resource_override is internal API for debugging/benchmarking only.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <FactorNet/core/types.hpp>
#include <FactorNet/core/constants.hpp>
#include <FactorNet/core/exceptions.hpp>
#include <FactorNet/core/factor_config.hpp>
#include <FactorNet/math/loss.hpp>

#include <string>
#include <vector>
#include <cstdint>
#include <functional>
#include <memory>

namespace FactorNet {

/**
 * @brief Unified NMF configuration for all backends
 *
 * Describes the complete NMF problem specification including rank,
 * convergence criteria, regularization, cross-validation, and solver
 * parameters. Feature dispatch uses runtime if-checks — no feature
 * flags or backend tags.
 *
 * @tparam Scalar  Element type (float or double)
 */
template<typename Scalar>
struct NMFConfig {

    // ====================================================================
    // Core
    // ====================================================================

    /// Factorization rank
    int rank = 10;

    /// Maximum outer iterations
    int max_iter = NMF_MAXIT;

    /// Relative convergence tolerance (|Δloss| / |loss| < tol)
    Scalar tol = static_cast<Scalar>(NMF_TOL);

    /// Patience: tol must hold for this many consecutive checks
    int patience = NMF_PATIENCE;

    /// Random seed for initialization + CV mask
    uint32_t seed = 0;

    /// Print per-iteration diagnostics
    bool verbose = false;

    // ====================================================================
    // Resource constraints (NOT backend selection)
    // ====================================================================

    /// CPU thread limit (0 = all available cores)
    int threads = 0;

    /// GPU limit (0 = all available GPUs)
    int max_gpus = 0;

    // ====================================================================
    // Per-factor regularization, constraints, and targets
    // ====================================================================

    /// W-factor configuration (regularization, constraints, targets)
    FactorConfig<Scalar> W;

    /// H-factor configuration (regularization, constraints, targets)
    FactorConfig<Scalar> H;

    /// True if any target regularization is configured on either factor
    bool has_target() const noexcept {
        return W.has_target() || H.has_target();
    }

    // ====================================================================
    // Algorithm variants
    // ====================================================================

    /// Projective NMF: H = W·A (tied-weight, no NNLS for H)
    /// Eliminates one Gram + one NNLS batch per iteration → ~1.5-2× speedup.
    /// H-side regularization (L1_H, L2_H, angular_H, etc.) is ignored.
    bool projective = false;

    /// Symmetric NMF: A ≈ W^T·diag(d)·W, enforces H = W.
    /// Input A must be square and symmetric. Only W is updated;
    /// per-iteration cost is half of standard NMF.
    bool symmetric = false;

    // ====================================================================
    // Initialization
    // ====================================================================

    /// Initialization mode:
    ///   0 = Random (uniform, default)
    ///   1 = Lanczos SVD seed: W = |U|·√Σ, then solve for H
    int init_mode = 0;

    // ====================================================================
    // NNLS solver (coordinate descent)
    // ====================================================================

    /// Solver mode for the NNLS subproblem:
    ///   0 = Coordinate Descent NNLS (fixed-iteration batch CD)
    ///   1 = Cholesky + clip (unconstrained LS then max(0,x); default)
    int solver_mode = 1;

    /// Maximum inner CD iterations per column
    int cd_max_iter = CD_MAXIT;

    /// Inner CD convergence tolerance
    Scalar cd_tol = static_cast<Scalar>(CD_TOL);

    // cd_abs_tol removed from config — always uses constants::CD_ABS_TOL

    // ====================================================================
    // Loss function (IRLS for non-MSE)
    // ====================================================================

    /// Loss function configuration
    LossConfig<Scalar> loss{};

    /// Maximum IRLS iterations per column (outer loop wrapping CD)
    int irls_max_iter = 5;

    /// IRLS convergence tolerance (relative max change per column)
    Scalar irls_tol = static_cast<Scalar>(1e-4);

    /// True if loss function requires IRLS (non-MSE)
    bool requires_irls() const noexcept {
        return loss.requires_irls();
    }

    // ====================================================================
    // Generalized Poisson dispersion
    // ====================================================================

    /// Dispersion mode: NONE (theta=0), GLOBAL, or PER_ROW
    DispersionMode gp_dispersion = DispersionMode::PER_ROW;

    /// Initial theta value for GP dispersion (default 0.1, grows from below)
    Scalar gp_theta_init = static_cast<Scalar>(0.1);

    /// Maximum theta value to cap per-row dispersion
    Scalar gp_theta_max = static_cast<Scalar>(5.0);

    // nr_theta_refine removed — DEPRECATED. Overestimates theta by +62-110%
    // due to conditional MLE bias on imperfect W/H; zero effect on factor quality.

    // gp_warmup_iters and gp_max_blend removed — dead code (never read).

    // ====================================================================
    // Negative Binomial dispersion
    // ====================================================================

    /// NB inverse-dispersion ("size" parameter r): Var(Y) = μ + μ²/r
    /// Estimated via method of moments from residuals each iteration.
    /// Dispersion mode reuses gp_dispersion (NONE/GLOBAL/PER_ROW).
    Scalar nb_size_init = static_cast<Scalar>(10.0);

    /// Maximum NB size (cap to prevent numerical issues at Poisson limit)
    Scalar nb_size_max = static_cast<Scalar>(1e6);

    /// Minimum NB size floor
    Scalar nb_size_min = static_cast<Scalar>(0.01);

    // ====================================================================
    // Gamma / Inverse Gaussian dispersion
    // ====================================================================

    /// Initial dispersion φ for Gamma (V(μ)=φμ²) and InvGauss (V(μ)=φμ³).
    /// Estimated via method-of-moments Pearson estimator each iteration.
    /// Dispersion mode reuses gp_dispersion (NONE/GLOBAL/PER_ROW).
    Scalar gamma_phi_init = static_cast<Scalar>(1.0);

    /// Maximum dispersion cap
    Scalar gamma_phi_max = static_cast<Scalar>(1e4);

    /// Minimum dispersion floor
    Scalar gamma_phi_min = static_cast<Scalar>(1e-6);

    // ====================================================================
    // Zero-Inflated Generalized Poisson (ZIGP) / Zero-Inflated NB (ZINB)
    // ====================================================================

    /// Zero-inflation mode: NONE, ROW, COL, or TWOWAY
    ZIMode zi_mode = ZIMode::ZI_NONE;

    /// Number of ZIGP EM iterations (E-step for z_ij + M-step for π)
    /// Each EM iteration runs inside each ALS iteration after theta update.
    int zi_em_iters = 1;

    /// Minimum theta floor to prevent θ-collapse under zero-inflation.
    /// The EM tends to absorb all zero-excess into π, collapsing θ→0.
    /// A small floor preserves the overdispersion signal.
    Scalar gp_theta_min = static_cast<Scalar>(0.0);

    /// True if ZIGP zero-inflation is active
    bool has_zi() const noexcept { return zi_mode != ZIMode::ZI_NONE; }

    // ====================================================================
    // Cross-validation
    // ====================================================================

    /// Separate seed for CV mask (0 = derive from seed)
    uint32_t cv_seed = 0;

    /// Fraction of entries held out for test set
    Scalar holdout_fraction = static_cast<Scalar>(0.1);

    /// mask_zeros=true: test set is held-out nonzeros only (recommendation)
    /// mask_zeros=false: test set includes zeros (full reconstruction)
    bool mask_zeros = false;

    // ====================================================================
    // User-supplied mask — works on ALL backends
    // ====================================================================

    /// Sparse mask matrix (m × n): nonzero entries are MASKED (treated as
    /// missing/unobserved). The caller owns the storage; nullptr = no mask.
    /// When set, NMF uses per-column delta-G correction to exclude masked
    /// entries from the Gram matrix and RHS, producing a correct factorization
    /// that ignores the masked positions.
    const SparseMatrix<Scalar>* mask = nullptr;

    /// True if a user-supplied mask is active
    bool has_mask() const noexcept { return mask != nullptr && mask->nonZeros() > 0; }

    /// Early-stopping patience for CV (iterations w/o test improvement)
    int cv_patience = NMF_PATIENCE;

    // ====================================================================
    // Loss tracking
    // ====================================================================

    /// Store full loss history (train + optional test) in result
    bool track_loss_history = false;

    /// Compute and record loss every N iterations (1 = every iteration)
    int loss_every = 1;

    /// Whether to compute training loss during CV (if false, only test loss is computed)
    /// Setting to false skips the expensive Gram trick for total loss → faster CV
    bool track_train_loss = true;

    // ====================================================================
    // GPU-native alignment guide (NNLS regularization toward targets)
    // Adds λ·M to Gram and λ·M·t to RHS to steer H and/or W toward
    // externally- or internally-derived targets.
    // ====================================================================

    struct AlignmentGuideConfig {
        /// Alignment type (0 = disabled):
        ///  1  = A20 static per-cell target (M=I, t=H_guided)
        ///  2  = A21 Mahalanobis target (M=Σ⁻¹, t=H_guided)
        ///  3  = A22 batch nullspace repulsion (M=VV^T, no target)
        ///  4  = A23 combined target + nullspace
        ///  5  = A24 W-side target only
        ///  6  = A25 dual W+H targets
        ///  7  = A26 dynamic target update (recompute every N iters)
        ///  8  = A27 whitened centroid targets
        ///  9  = A28 annealed regularization (time-varying λ)
        int type = 0;

        Scalar lambda_H = 0;          ///< H-side regularization strength
        Scalar lambda_W = 0;          ///< W-side regularization strength
        Scalar lambda_batch = 0;      ///< Batch nullspace repulsion strength
        Scalar lambda_c = 0.3;        ///< Classifier guide λ (for target derivation)
        int update_interval = 10;     ///< Recompute targets every N iters (A26)
        int warmup_iters = 10;        ///< Iterations before enabling regularization

        /// Annealing schedule (A28): 0=none, 1=cosine, 2=linear, 3=cutoff_50, 4=cutoff_25
        int anneal_schedule = 0;

        /// Per-sample labels for target computation (0-indexed, -1 = unlabeled)
        std::vector<int> labels;
        int n_classes = 0;

        /// Per-sample batch labels for nullspace repulsion (0-indexed)
        std::vector<int> batch_labels;
        int n_batches = 0;

        /// Pre-computed H-side target matrix (k × n, column-major)
        /// If empty, computed internally from OAS-ZCA guided run
        std::vector<Scalar> h_target;

        /// Pre-computed W-side target matrix (k × m, column-major)
        std::vector<Scalar> w_target;

        /// Pre-computed precision matrix (k × k, column-major) for A21
        std::vector<Scalar> precision;

        /// Pre-computed batch subspace matrix (k × k, column-major) for A22/A23
        std::vector<Scalar> batch_subspace;
    };

    AlignmentGuideConfig alignment_guide;

    /// True if any alignment guide regularization is active
    bool has_alignment_guide() const noexcept {
        return alignment_guide.type > 0;
    }

    // ====================================================================
    // Classifier guides (H-side, multi-guide)
    // Adds λ·I to Gram and λ·centroid to RHS to steer each column of H
    // toward its class centroid. Supports multiple simultaneous guides.
    // ====================================================================

    struct ClassifierGuideEntry {
        std::vector<int> labels;   ///< Per-sample labels (0-indexed, length n)
        Scalar lambda = 0;        ///< Regularization strength
        int n_classes = 0;        ///< Number of classes
    };

    std::vector<ClassifierGuideEntry> classifier_guides_H;

    /// True if any classifier guide is active on H
    bool has_classifier_guide_H() const noexcept {
        return !classifier_guides_H.empty();
    }

    // ====================================================================
    // Post-processing
    // ====================================================================

    /// Normalization type for scaling diagonal (L1, L2, or None)
    NormType norm_type = NormType::L1;

    /// Sort factors by decreasing d after convergence
    bool sort_model = false;

    // ====================================================================
    // Resource override (internal API — see §4.4 of proposal)
    // ====================================================================

    /// "auto" (default), "cpu", "gpu"
    /// Also settable via FACTORNET_RESOURCE environment variable.
    std::string resource_override = "auto";

    // ====================================================================
    // Streaming / out-of-core NMF
    // ====================================================================

    /// Path to .spz v2 file for streaming NMF (empty = in-memory)
    std::string spz_path;

    // ====================================================================
    // Multi-rank / replicated CV
    // ====================================================================

    /// Vector of ranks to evaluate (for rank-CV or multi-rank fits)
    std::vector<int> k_vec;

    /// Vector of CV seeds for replicated cross-validation
    std::vector<uint32_t> cv_seeds;

    // ====================================================================
    // Iteration callback (optional)
    // ====================================================================

    /// Per-iteration callback: (iteration, train_loss, test_loss)
    /// Called after each outer iteration if set. Use for custom logging,
    /// early stopping, or progress reporting.
    std::function<void(int, Scalar, Scalar)> on_iteration;

    // ====================================================================
    // Profiling
    // ====================================================================

    /// Enable per-section wall-clock profiling (zero overhead when false)
    bool enable_profiling = false;

    // ====================================================================
    // Helpers
    // ====================================================================

    /// True if any regularization at all is active
    bool has_regularization() const noexcept {
        return W.has_any_reg() || H.has_any_reg();
    }

    /// True if cross-validation is enabled (holdout_fraction > 0)
    bool is_cv() const noexcept {
        return holdout_fraction > 0 && holdout_fraction < 1;
    }

    /// Effective CV seed (falls back to seed if cv_seed not set)
    uint32_t effective_cv_seed() const noexcept {
        return cv_seed != 0 ? cv_seed : seed;
    }

    /// Validate configuration (throws InvalidArgument on errors)
    void validate() const {
        if (rank <= 0)
            throw InvalidArgument("rank must be positive", rank);
        if (max_iter <= 0)
            throw InvalidArgument("max_iter must be positive", max_iter);
        if (tol < 0)
            throw InvalidArgument("tol must be non-negative");
        if (holdout_fraction < 0 || holdout_fraction >= 1)
            throw InvalidArgument("holdout_fraction must be in [0, 1)");
        if (cd_max_iter <= 0)
            throw InvalidArgument("cd_max_iter must be positive", cd_max_iter);
        if (projective && symmetric)
            throw InvalidArgument("projective and symmetric cannot both be true");

        // Zero-inflation only with GP or NB
        if (zi_mode != ZIMode::ZI_NONE) {
            if (zi_mode == ZIMode::ZI_TWOWAY)
                throw InvalidArgument(
                    "zi_mode=TWOWAY is disabled due to numerical instability "
                    "on high-sparsity data. Use ZI_ROW or ZI_COL instead.");
            if (loss.type != LossType::GP && loss.type != LossType::NB)
                throw InvalidArgument(
                    "Zero-inflation (zi_mode != NONE) requires GP or NB loss. "
                    "MSE, Gamma, InvGauss, and Tweedie do not support zero-inflation.");
        }

        // Cholesky solver does not support IRLS (non-MSE distributions or robust)
        if (solver_mode == 1 && requires_irls())
            throw InvalidArgument(
                "Cholesky solver (solver_mode=1) is not supported with IRLS-based "
                "distributions. Use CD solver (solver_mode=0) for GP, NB, Gamma, "
                "InvGauss, Tweedie, or robust estimation.");
    }
};

}  // namespace FactorNet

