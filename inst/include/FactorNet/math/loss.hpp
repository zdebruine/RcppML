// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file loss.hpp
 * @brief Loss function evaluation and IRLS weights
 * 
 * Implements multiple loss functions for NMF:
 * - MSE (Mean Squared Error / Frobenius norm)
 * - MAE (Mean Absolute Error / L1 loss)
 * - Huber (Robust hybrid loss)
 * - KL (Kullback-Leibler divergence)
 * 
 * Non-MSE losses use Iteratively Reweighted Least Squares (IRLS).
 * 
 * @author Zach DeBruine
 * @date 2026-01-24
 */

/*
 * LOSS DEFINITIONS (one per distribution, used on every CPU path)
 * ---------------------------------------------------------------
 * Model: A ~ mu = W diag(d) H, identity link. The loss reported by a fit is
 * the SUM over all m*n entries (stored zeros included; sparse and dense
 * input give the same number) of the per-entry value returned by
 * compute_loss():
 *
 *   mse               (y - mu)^2
 *   gp                exact generalized-Poisson negative log-likelihood in
 *                     nats, log(y!) included. Mean mu, variance
 *                     mu (1 + theta)^2 (Consul's form with
 *                     lambda = mu / (1 + theta)); at y = 0 it is
 *                     mu / (1 + theta). theta = 0 is the Poisson NLL.
 *   nb                exact negative-binomial NLL in nats, lgamma(y+1)
 *                     included; variance mu + mu^2 / r.
 *   gamma             unit deviance, V(mu) = mu^2
 *   inverse_gaussian  unit deviance, V(mu) = mu^3
 *   tweedie           unit deviance, V(mu) = mu^p, p >= 1
 *
 * Constants: likelihood losses (gp, nb) drop nothing, so they are
 * non-negative and comparable across dispersion values. Deviance losses
 * (gamma, inverse_gaussian, tweedie) are non-negative, zero at a perfect
 * fit and exclude the dispersion phi, which does not enter the W/H updates.
 *
 * With robust_delta > 0, compute_robust_loss() returns
 * 2 rho_delta(sqrt(d(y, mu))) with d the unit deviance of the working model
 * (unit_deviance(); Poisson deviance / (1 + theta)^2 for gp, NB deviance for
 * nb) and rho the Huber function.
 *
 * W/H updates minimise sum(unit_deviance) / 2 by IRLS with weights
 * 1 / V(mu) and step control (primitives/cpu/nnls_batch_irls.hpp). For gp the
 * working deviance is the Poisson quasi-deviance scaled by 1 / (1 + theta)^2,
 * so the reported gp loss is the exact NLL while the updates use the
 * quasi-likelihood with the same mean and variance.
 *
 * Floors and caps: every loss and deviance floors mu at 1e-10
 * (loss_detail::mu_floor). IRLS weights are capped for single-precision
 * conditioning (1e4 for Poisson-type weights, 1e6 for the others). The cap
 * alters only the search direction; the step control evaluates the deviance
 * with the 1e-10 floor, so an accepted step never increases it.
 */

#ifndef FactorNet_MATH_LOSS_HPP
#define FactorNet_MATH_LOSS_HPP

// Only include what we need — NOT core/core.hpp (umbrella).
// Including core.hpp creates a circular dependency:
//   config.hpp → math/loss.hpp → core/core.hpp → resources.hpp → config.hpp
// loss.hpp only needs tiny_num<> and kl_epsilon<> from constants.hpp.
#include <FactorNet/core/constants.hpp>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace FactorNet {

/**
 * @brief Loss function types
 */
enum class LossType {
    MSE = 0,    ///< Mean Squared Error (default, fastest)
    MAE = 1,    ///< Mean Absolute Error (robust to outliers) — DEPRECATED, use robust_delta
    HUBER = 2,  ///< Huber loss (MSE+MAE hybrid) — DEPRECATED, use robust_delta
    KL = 3,     ///< KL divergence — INTERNAL ONLY (used as IRLS weight mode inside GP)
                ///< User-facing KL is now loss="gp" with dispersion="none".
    GP = 4,     ///< Generalized Poisson (overdispersed count data)
    NB = 5,     ///< Negative Binomial (quadratic variance-mean, scRNA-seq standard)
    GAMMA = 6,  ///< Gamma distribution: V(μ) = μ². For positive continuous data.
    INVGAUSS = 7, ///< Inverse Gaussian: V(μ) = μ³. Heavy right skew.
    TWEEDIE = 8  ///< Tweedie: V(μ) = μ^p for continuous power p. Generalizes MSE/GP/Gamma/IG.
};

/**
 * @brief Dispersion mode for Generalized Poisson loss
 */
enum class DispersionMode {
    NONE = 0,     ///< No dispersion (θ=0, equivalent to KL/PMF)
    GLOBAL = 1,   ///< Single θ for entire matrix
    PER_ROW = 2,  ///< θ_i per row (recommended for heterogeneous data)
    PER_COL = 3   ///< θ_j per column
};

/**
 * @brief Zero-inflation mode for ZIGP-NMF
 *
 * Controls how the dropout probability π is parameterized:
 *   NONE:    No zero-inflation (standard GP-NMF)
 *   ROW:     Per-row π_i (gene-level dropout)
 *   COL:     Per-column π_j (cell/sample-level dropout)
 *   TWOWAY:  Independent row×col: π_ij = 1 - (1-π_i)(1-π_j)
 */
enum class ZIMode {
    ZI_NONE = 0,   ///< Standard GP-NMF (no zero-inflation)
    ZI_ROW = 1,    ///< Per-row π_i
    ZI_COL = 2,    ///< Per-column π_j
    ZI_TWOWAY = 3  ///< Independent row × column
};

/**
 * @brief Loss function configuration
 * 
 * @tparam Scalar Computation precision
 */
template<typename Scalar>
struct LossConfig {
    LossType type = LossType::MSE;
    Scalar huber_delta = static_cast<Scalar>(1.0);
    Scalar kl_epsilon = static_cast<Scalar>(1e-10);
    
    // Generalized Poisson dispersion parameters
    DispersionMode gp_dispersion = DispersionMode::PER_ROW;
    Scalar gp_theta_init = static_cast<Scalar>(0.1);

    // GP/KL weight blending: 0.0 = pure KL weight (1/s), 1.0 = full GP weight (1/s²+...)
    // Geometric interpolation: w = w_KL^(1-blend) * w_GP^blend
    // Ramped from 0 to 1 over post-warmup iterations for stability.
    Scalar gp_blend = static_cast<Scalar>(1.0);

    // Runtime dispersion parameter for distributions that need it (GP, NB, Gamma, IG).
    // For GP: theta (overdispersion); for NB: size parameter; for Gamma/IG: phi.
    // Updated during fitting; initial value comes from config (gp_theta_init etc.)
    Scalar dispersion = static_cast<Scalar>(1.0);

    // Robust outlier downweighting via Huber on Pearson residuals.
    // robust_delta <= 0 means no robustness.
    // robust_delta > 0 applies Huber trimming with that delta to the
    // deviance residual sqrt(d(y, μ)) of the working model (equal to |y - μ| for MSE).
    // Composes multiplicatively with the distribution-derived weight.
    // Small delta (~1e-4) ≈ MAE behavior; 1.345 = 95% efficiency; Inf = no effect.
    Scalar robust_delta = static_cast<Scalar>(0);

    // Tweedie continuous power parameter: V(μ) = μ^power_param.
    // Only used when type == TWEEDIE. Supported range: p >= 1.
    //   p=1 → Poisson, 1<p<2 → compound Poisson-gamma (zeros allowed),
    //   p=2 → Gamma, p=3 → Inverse Gaussian (p >= 2 needs strictly positive data).
    // Checked by validate_loss_config(); p < 1 is rejected.
    Scalar power_param = static_cast<Scalar>(1.5);

    /**
     * @brief Check if loss requires IRLS
     * @return true for any non-Gaussian distribution or if robustness is active
     */
    bool requires_irls() const {
        return type != LossType::MSE || robust_delta > static_cast<Scalar>(0);
    }
};

// ============================================================================
// Unit deviances (double precision, shared by weights, step control and loss)
// ============================================================================

namespace loss_detail {

/// Floor applied to the model value mu wherever a loss or deviance is evaluated.
inline double mu_floor() noexcept { return 1e-10; }

/// Huber rho: r^2/2 for |r| <= delta, delta*|r| - delta^2/2 otherwise.
inline double huber_rho(double r, double delta) noexcept {
    const double a = std::abs(r);
    if (a <= delta) return 0.5 * a * a;
    return delta * a - 0.5 * delta * delta;
}

/// Poisson unit deviance: 2[y log(y/mu) - (y - mu)]; 2*mu at y = 0.
inline double poisson_unit_deviance(double y, double mu) noexcept {
    mu = std::max(mu, mu_floor());
    if (!(y > 0.0)) return 2.0 * mu;
    const double d = 2.0 * (y * std::log(y / mu) - (y - mu));
    return d > 0.0 ? d : 0.0;
}

/// Negative binomial unit deviance at fixed size r; r <= 0 means "no size
/// supplied" and gives the Poisson limit.
inline double nb_unit_deviance(double y, double mu, double r) noexcept {
    if (!(r > 0.0)) return poisson_unit_deviance(y, mu);
    mu = std::max(mu, mu_floor());
    if (!(y > 0.0)) return 2.0 * r * std::log1p(mu / r);
    const double d = 2.0 * (y * std::log(y / mu)
                            - (y + r) * std::log((y + r) / (mu + r)));
    return d > 0.0 ? d : 0.0;
}

/// Power-variance (Tweedie) unit deviance for V(mu) = mu^p.
/// p = 1 is Poisson, p = 2 gamma and p = 3 inverse Gaussian, each evaluated
/// with its closed form so that Tweedie reproduces those losses exactly.
/// For p < 2, y = 0 is in the support and d(0, mu) = 2 mu^(2-p) / (2-p).
/// For p >= 2, y must be positive; y is floored at 1e-10 to stay finite.
inline double power_unit_deviance(double y, double mu, double p) noexcept {
    if (std::abs(p) < 1e-6) {
        const double r = y - mu;
        return r * r;
    }
    if (std::abs(p - 1.0) < 1e-6) return poisson_unit_deviance(y, mu);
    mu = std::max(mu, mu_floor());
    if (p < 2.0 - 1e-6) {
        if (!(y > 0.0)) return 2.0 * std::pow(mu, 2.0 - p) / (2.0 - p);
    } else {
        y = std::max(y, mu_floor());
    }
    double d;
    if (std::abs(p - 2.0) < 1e-6) {
        d = 2.0 * ((y - mu) / mu - std::log(y / mu));
    } else if (p == 3.0) {
        const double r = y - mu;
        d = r * r / (mu * mu * y);
    } else {
        const double omp = 1.0 - p;
        const double tmp = 2.0 - p;
        d = 2.0 * (std::pow(y, tmp) / (omp * tmp)
                   - y * std::pow(mu, omp) / omp
                   + std::pow(mu, tmp) / tmp);
    }
    return d > 0.0 ? d : 0.0;
}

}  // namespace loss_detail

/**
 * @brief Unit deviance of the working model used for the W/H updates.
 *
 * This is the function the IRLS weights are derived from: half of it has
 * derivative -(y - mu) * w(mu) in mu, where w is the IRLS weight.
 *
 *   MSE (and legacy MAE/HUBER types): (y - mu)^2
 *   KL:        Poisson deviance
 *   GP:        Poisson deviance / (1 + theta)^2   (quasi-likelihood, Var = mu (1+theta)^2)
 *   NB:        negative binomial deviance at size r = theta (r <= 0: Poisson)
 *   GAMMA:     power deviance, p = 2
 *   INVGAUSS:  power deviance, p = 3
 *   TWEEDIE:   power deviance, p = power_param
 */
template<typename Scalar>
inline double unit_deviance(Scalar observed, Scalar predicted,
                            const LossConfig<Scalar>& config,
                            Scalar theta = static_cast<Scalar>(0)) {
    const double y = static_cast<double>(observed);
    const double mu = static_cast<double>(predicted);
    switch (config.type) {
        case LossType::KL:
            return loss_detail::poisson_unit_deviance(y, mu);
        case LossType::GP: {
            const double c = 1.0 + std::max(static_cast<double>(theta), 0.0);
            return loss_detail::poisson_unit_deviance(y, mu) / (c * c);
        }
        case LossType::NB:
            return loss_detail::nb_unit_deviance(y, mu, static_cast<double>(theta));
        case LossType::GAMMA:
            return loss_detail::power_unit_deviance(y, mu, 2.0);
        case LossType::INVGAUSS:
            return loss_detail::power_unit_deviance(y, mu, 3.0);
        case LossType::TWEEDIE:
            return loss_detail::power_unit_deviance(
                y, mu, static_cast<double>(config.power_param));
        default: {
            const double r = y - mu;
            return r * r;
        }
    }
}

/**
 * @brief Validate the loss configuration (call from serial code only).
 *
 * Tweedie supports p = 1 (Poisson), 1 < p < 2 (compound Poisson-gamma, zeros
 * allowed) and p >= 2 (gamma at 2, inverse Gaussian at 3; data must be
 * strictly positive). No Tweedie distribution exists for 0 < p < 1, and
 * p <= 0 is not offered here; use loss = "mse" for p = 0.
 *
 * @throws std::invalid_argument if the Tweedie power is below 1 or not finite
 */
template<typename Scalar>
inline void validate_loss_config(const LossConfig<Scalar>& config) {
    if (config.type == LossType::TWEEDIE) {
        const double p = static_cast<double>(config.power_param);
        if (!(p >= 1.0) || !std::isfinite(p))
            throw std::invalid_argument(
                "Tweedie loss requires tweedie_power >= 1 "
                "(1 = Poisson, (1, 2) = compound Poisson-gamma, 2 = gamma, "
                "3 = inverse Gaussian); no Tweedie distribution exists for "
                "powers in (0, 1), and power 0 is loss = \"mse\".");
    }
}

// ============================================================================
// IRLS weights
// ============================================================================

/**
 * @brief Compute IRLS weight for MAE loss
 *
 * Weight w = 1 / (|residual| + eps)
 *
 * @tparam Scalar Precision type
 * @param residual Difference between observed and predicted
 * @param eps Numerical stability constant
 * @return IRLS weight
 */
template<typename Scalar>
inline Scalar irls_weight_mae(Scalar residual, Scalar eps = tiny_num<Scalar>()) {
    return static_cast<Scalar>(1) / (std::abs(residual) + eps);
}

/**
 * @brief Compute IRLS weight for Huber loss
 *
 * Quadratic for |r| < delta, linear for |r| >= delta:
 * - weight = 1 if |r| < delta
 * - weight = delta / |r| if |r| >= delta
 *
 * @tparam Scalar Precision type
 * @param residual Difference between observed and predicted
 * @param delta Huber transition threshold
 * @param eps Numerical stability constant
 * @return IRLS weight
 */
template<typename Scalar>
inline Scalar irls_weight_huber(Scalar residual, Scalar delta, Scalar eps = tiny_num<Scalar>()) {
    Scalar abs_r = std::abs(residual);
    if (abs_r < delta) {
        return static_cast<Scalar>(1);
    } else {
        return delta / (abs_r + eps);
    }
}

/**
 * @brief Compute IRLS weight for KL divergence (Poisson Fisher weight)
 *
 * Weight w = 1 / max(predicted, floor)
 * Floor prevents catastrophic weight explosion when predictions are near-zero,
 * which occurs on highly sparse count data (e.g., scRNA-seq with >90% sparsity).
 *
 * @tparam Scalar Precision type
 * @param predicted Predicted value (must be positive)
 * @param eps Floor to prevent division by zero / weight explosion
 * @return IRLS weight
 */
template<typename Scalar>
inline Scalar irls_weight_kl(Scalar predicted, Scalar eps = static_cast<Scalar>(1e-4)) {
    return static_cast<Scalar>(1) / std::max(predicted, eps);
}

/**
 * @brief Compute IRLS weight for Generalized Poisson loss
 *
 * Quasi-likelihood weight 1 / Var(y) for the mean-parameterised GP, whose
 * variance is s (1 + theta)^2:
 *   w = 1 / (max(s, 1e-4) * (1 + theta)^2)
 *
 * The weight depends on the model value only. (The observed-information
 * weight used previously depended on y; a weighted least-squares step whose
 * weights depend on the response solves a biased estimating equation.)
 * `observed`, `blend` and `eps` are kept for source compatibility and unused.
 *
 * @tparam Scalar Precision type
 * @param observed Observed count (unused)
 * @param predicted Predicted mean (s_ij)
 * @param theta Dispersion parameter (theta >= 0)
 * @param blend Unused
 * @param eps Unused
 * @return IRLS weight
 */
template<typename Scalar>
inline Scalar irls_weight_gp(Scalar observed, Scalar predicted, Scalar theta,
                             Scalar blend = static_cast<Scalar>(1),
                             Scalar eps = tiny_num<Scalar>()) {
    (void)observed;
    (void)blend;
    (void)eps;
    const double s = std::max(static_cast<double>(predicted), 1e-4);
    const double c = 1.0 + std::max(static_cast<double>(theta), 0.0);
    return static_cast<Scalar>(1.0 / (s * c * c));
}

/**
 * @brief Compute IRLS weight for Negative Binomial loss
 *
 * Based on the expected Fisher information of the NB negative log-likelihood
 * w.r.t. the mean parameter μ:
 *   E[-∂²ℓ/∂μ²] = r / (μ · (r + μ))
 *
 * where r is the "size" (inverse-dispersion) parameter. Var(Y) = μ + μ²/r.
 * - r → ∞ gives Poisson limit: w → 1/μ (same as KL weight)
 * - r <= 0 means no size was supplied: the Poisson weight is returned
 *
 * @tparam Scalar Precision type
 * @param predicted Predicted mean μ (must be positive)
 * @param nb_size Inverse-dispersion parameter r > 0
 * @param eps Stability floor
 * @return IRLS weight
 */
template<typename Scalar>
inline Scalar irls_weight_nb(Scalar predicted, Scalar nb_size,
                             Scalar eps = tiny_num<Scalar>()) {
    if (!(nb_size > static_cast<Scalar>(0))) return irls_weight_kl(predicted);
    double mu = std::max(static_cast<double>(predicted), static_cast<double>(eps));
    double r = static_cast<double>(nb_size);
    double w = r / (mu * (r + mu));
    w = std::min(w, 1e6);
    return static_cast<Scalar>(w);
}

/**
 * @brief Compute IRLS weight for power-variance family: V(μ) = μ^p
 *
 * Unifies Gaussian (p=0), Poisson (p=1), Gamma (p=2), InvGaussian (p=3).
 * Weight = 1/μ^p (Fisher information of the corresponding EDM).
 * p = 1, 2 and 3 are evaluated by multiplication, so Tweedie with those
 * powers gives the same weights as the gamma and inverse Gaussian losses.
 *
 * @tparam Scalar Precision type
 * @param predicted Predicted mean μ (must be positive for p>0)
 * @param power Variance power p
 * @param eps Stability floor
 * @return IRLS weight
 */
template<typename Scalar>
inline Scalar irls_weight_power(Scalar predicted, Scalar power,
                                Scalar eps = tiny_num<Scalar>()) {
    const double mu = std::max(static_cast<double>(predicted), static_cast<double>(eps));
    const double p = static_cast<double>(power);
    double v;
    if (std::abs(p - 1.0) < 1e-6) {
        v = mu;
    } else if (std::abs(p - 2.0) < 1e-6) {
        v = mu * mu;
    } else if (p == 3.0) {
        v = mu * mu * mu;
    } else {
        v = std::pow(mu, p);
    }
    double w = 1.0 / v;
    w = std::min(w, 1e6);
    return static_cast<Scalar>(w);
}

/**
 * @brief Compute Huber-style robust modifier on a standardized residual
 *
 * Given a standardized residual r (the deviance residual sqrt(d(y, μ)) in
 * the IRLS solver), apply Huber trimming:
 *   w_robust = min(1, δ / |r|)
 *
 * As δ → 0, this approaches 1/|r| (MAE-like behavior).
 * This modifier multiplies the distribution weight: w_final = w_dist × w_robust.
 *
 * @tparam Scalar Precision type
 * @param pearson_residual Standardized residual
 * @param delta Huber threshold (>0)
 * @param eps Numerical stability
 * @return Robust weight modifier ∈ (0, 1]
 */
template<typename Scalar>
inline Scalar robust_huber_modifier(Scalar pearson_residual, Scalar delta,
                                     Scalar eps = tiny_num<Scalar>()) {
    Scalar abs_r = std::abs(pearson_residual);
    if (abs_r <= delta) {
        return static_cast<Scalar>(1);
    } else {
        return delta / (abs_r + eps);
    }
}

// ============================================================================
// Reported loss, per entry
// ============================================================================

/**
 * @brief Compute loss contribution for single observation (MSE)
 *
 * @tparam Scalar Precision type
 * @param observed Observed value
 * @param predicted Predicted value
 * @return Squared error
 */
template<typename Scalar>
inline Scalar loss_contribution_mse(Scalar observed, Scalar predicted) {
    Scalar diff = observed - predicted;
    return diff * diff;
}

/**
 * @brief Compute loss contribution for single observation (MAE)
 *
 * @tparam Scalar Precision type
 * @param observed Observed value
 * @param predicted Predicted value
 * @return Absolute error
 */
template<typename Scalar>
inline Scalar loss_contribution_mae(Scalar observed, Scalar predicted) {
    return std::abs(observed - predicted);
}

/**
 * @brief Compute loss contribution for single observation (Huber)
 *
 * @tparam Scalar Precision type
 * @param observed Observed value
 * @param predicted Predicted value
 * @param delta Huber transition threshold
 * @return Huber loss
 */
template<typename Scalar>
inline Scalar loss_contribution_huber(Scalar observed, Scalar predicted, Scalar delta) {
    Scalar abs_diff = std::abs(observed - predicted);
    if (abs_diff <= delta) {
        return static_cast<Scalar>(0.5) * abs_diff * abs_diff;
    } else {
        return delta * (abs_diff - static_cast<Scalar>(0.5) * delta);
    }
}

/**
 * @brief Compute loss contribution for single observation (KL divergence)
 *
 * KL(observed || predicted) = observed × log(observed / predicted) - observed + predicted
 *
 * @tparam Scalar Precision type
 * @param observed Observed value (must be non-negative)
 * @param predicted Predicted value (must be positive)
 * @param eps Floor for log to avoid log(0)
 * @return KL divergence
 */
template<typename Scalar>
inline Scalar loss_contribution_kl(Scalar observed, Scalar predicted, Scalar eps = kl_epsilon<Scalar>()) {
    observed = std::max(observed, eps);
    predicted = std::max(predicted, eps);
    return observed * std::log(observed / predicted) - observed + predicted;
}

/**
 * @brief Compute loss contribution for single observation (Generalized Poisson)
 *
 * Exact GP negative log-likelihood, including log(y!):
 *   y > 0:  -log(s/c) - (y-1) log((s + θy)/c) + (s + θy)/c + lgamma(y+1)
 *   y = 0:  s/c
 * with c = 1 + θ. (At y = 0 the two log terms of the general expression
 * cancel; the previous code kept one of them.)
 *
 * @tparam Scalar Precision type
 * @param observed Observed count y_ij
 * @param predicted Predicted mean s_ij = Σ_k w_ik h_jk
 * @param theta Dispersion parameter θ >= 0
 * @param eps Unused (kept for source compatibility)
 * @return GP negative log-likelihood contribution
 */
template<typename Scalar>
inline Scalar loss_contribution_gp(Scalar observed, Scalar predicted, Scalar theta,
                                   Scalar eps = tiny_num<Scalar>()) {
    (void)eps;
    const double s = std::max(static_cast<double>(predicted), loss_detail::mu_floor());
    const double y = static_cast<double>(observed);
    const double th = std::max(static_cast<double>(theta), 0.0);
    const double c = 1.0 + th;
    if (!(y > 0.0)) return static_cast<Scalar>(s / c);
    const double inner = std::max((s + th * y) / c, loss_detail::mu_floor());
    const double loss = -std::log(s / c) - (y - 1.0) * std::log(inner)
                        + (s + th * y) / c + std::lgamma(y + 1.0);
    return static_cast<Scalar>(loss);
}

/**
 * @brief Compute loss contribution for single observation (Negative Binomial)
 *
 * Exact NB negative log-likelihood, including lgamma(y+1):
 *   y > 0:  -lgamma(y+r) + lgamma(r) + lgamma(y+1) - r·log(r/(r+μ)) - y·log(μ/(r+μ))
 *   y = 0:  r·log(1 + μ/r)
 *
 * where r is the inverse-dispersion parameter ("size"). r <= 0 means no size
 * was supplied and gives the Poisson negative log-likelihood
 * μ - y·log(μ) + lgamma(y+1).
 *
 * @tparam Scalar Precision type
 * @param observed Observed count y
 * @param predicted Predicted mean μ = (WH)_ij
 * @param nb_size Inverse-dispersion parameter r > 0
 * @return NB negative log-likelihood contribution
 */
template<typename Scalar>
inline Scalar loss_contribution_nb(Scalar observed, Scalar predicted, Scalar nb_size) {
    const double y = static_cast<double>(observed);
    const double mu = std::max(static_cast<double>(predicted), loss_detail::mu_floor());
    const double r = static_cast<double>(nb_size);
    if (!(r > 0.0)) {
        if (!(y > 0.0)) return static_cast<Scalar>(mu);
        return static_cast<Scalar>(mu - y * std::log(mu) + std::lgamma(y + 1.0));
    }
    if (!(y > 0.0)) return static_cast<Scalar>(r * std::log1p(mu / r));
    const double nll = -std::lgamma(y + r) + std::lgamma(r) + std::lgamma(y + 1.0)
                       - r * std::log(r / (r + mu))
                       - y * std::log(mu / (r + mu));
    return static_cast<Scalar>(nll);
}

/**
 * @brief Compute loss contribution for single observation (Gamma deviance)
 *
 * Unit deviance: d(y, μ) = 2[-log(y/μ) + (y − μ)/μ]
 * Valid for y > 0, μ > 0.
 *
 * @tparam Scalar Precision type
 * @param observed Observed value y (must be positive)
 * @param predicted Predicted value μ (must be positive)
 * @return Gamma deviance contribution
 */
template<typename Scalar>
inline Scalar loss_contribution_gamma(Scalar observed, Scalar predicted) {
    return static_cast<Scalar>(loss_detail::power_unit_deviance(
        static_cast<double>(observed), static_cast<double>(predicted), 2.0));
}

/**
 * @brief Compute loss contribution for single observation (Inverse Gaussian deviance)
 *
 * Unit deviance: d(y, μ) = (y − μ)² / (μ² · y)
 * Valid for y > 0, μ > 0.
 *
 * @tparam Scalar Precision type
 * @param observed Observed value y (must be positive)
 * @param predicted Predicted value μ (must be positive)
 * @return Inverse Gaussian deviance contribution
 */
template<typename Scalar>
inline Scalar loss_contribution_invgauss(Scalar observed, Scalar predicted) {
    return static_cast<Scalar>(loss_detail::power_unit_deviance(
        static_cast<double>(observed), static_cast<double>(predicted), 3.0));
}

/**
 * @brief Compute loss contribution for single observation (Tweedie deviance)
 *
 * Tweedie unit deviance for variance function V(μ) = μ^p.
 * p = 1 (Poisson), p = 2 (gamma) and p = 3 (inverse Gaussian) use their
 * closed forms. General formula for other p:
 *   d(y, μ) = 2 [ y^{2-p}/((1-p)(2-p)) - y·μ^{1-p}/(1-p) + μ^{2-p}/(2-p) ]
 * and d(0, μ) = 2 μ^{2-p}/(2-p) for p < 2.
 *
 * @tparam Scalar Precision type
 * @param observed Observed value y (must be non-negative)
 * @param predicted Predicted value μ (must be positive)
 * @param p Variance power parameter
 * @return Tweedie deviance contribution
 */
template<typename Scalar>
inline Scalar loss_contribution_tweedie(Scalar observed, Scalar predicted, Scalar p) {
    return static_cast<Scalar>(loss_detail::power_unit_deviance(
        static_cast<double>(observed), static_cast<double>(predicted),
        static_cast<double>(p)));
}

/**
 * @brief Compute the reported loss for one entry (see the table at the top
 *        of this file)
 *
 * @tparam Scalar Precision type
 * @param observed Observed value
 * @param predicted Predicted value
 * @param config Loss configuration
 * @param theta Dispersion for this entry (GP theta or NB size); ignored otherwise
 * @return Loss value
 */
template<typename Scalar>
inline Scalar compute_loss(Scalar observed, Scalar predicted, const LossConfig<Scalar>& config,
                           Scalar theta = static_cast<Scalar>(0)) {
    switch (config.type) {
        case LossType::MSE:
            return loss_contribution_mse(observed, predicted);
        case LossType::MAE:
            return loss_contribution_mae(observed, predicted);
        case LossType::HUBER:
            return loss_contribution_huber(observed, predicted, config.huber_delta);
        case LossType::KL:
            return loss_contribution_kl(observed, predicted, config.kl_epsilon);
        case LossType::GP:
            return loss_contribution_gp(observed, predicted, theta);
        case LossType::NB:
            return loss_contribution_nb(observed, predicted, theta);
        case LossType::GAMMA:
            return loss_contribution_gamma(observed, predicted);
        case LossType::INVGAUSS:
            return loss_contribution_invgauss(observed, predicted);
        case LossType::TWEEDIE:
            return loss_contribution_tweedie(observed, predicted, config.power_param);
        default:
            return loss_contribution_mse(observed, predicted);
    }
}

/**
 * @brief Compute the reported loss for one entry, with optional robustness
 *
 * When robust_delta <= 0, returns compute_loss().
 *
 * When robust_delta > 0, returns 2 ρ_δ(sqrt(d(y, μ))), where d is the unit
 * deviance of the working model (unit_deviance()) and ρ_δ is the Huber
 * function. It equals d wherever sqrt(d) <= δ and grows linearly in sqrt(d)
 * beyond, and it is the function whose gradient the robust IRLS weights
 * w_dist(μ) · min(1, δ / sqrt(d)) reproduce.
 */
template<typename Scalar>
inline Scalar compute_robust_loss(Scalar observed, Scalar predicted,
                                   const LossConfig<Scalar>& config,
                                   Scalar theta = static_cast<Scalar>(0)) {
    if (config.robust_delta <= static_cast<Scalar>(0)) {
        return compute_loss(observed, predicted, config, theta);
    }
    const double d = unit_deviance(observed, predicted, config, theta);
    return static_cast<Scalar>(2.0 * loss_detail::huber_rho(
        std::sqrt(d), static_cast<double>(config.robust_delta)));
}

}  // namespace FactorNet

#endif // FactorNet_MATH_LOSS_HPP
