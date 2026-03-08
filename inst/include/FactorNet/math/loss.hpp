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

#ifndef FactorNet_MATH_LOSS_HPP
#define FactorNet_MATH_LOSS_HPP

// Only include what we need — NOT core/core.hpp (umbrella).
// Including core.hpp creates a circular dependency:
//   config.hpp → math/loss.hpp → core/core.hpp → resources.hpp → config.hpp
// loss.hpp only needs tiny_num<> and kl_epsilon<> from constants.hpp.
#include <FactorNet/core/constants.hpp>
#include <cmath>
#include <algorithm>

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
    // standardized (Pearson) residual: r_P = (y - μ) / sqrt(V(μ)).
    // Composes multiplicatively with the distribution-derived weight.
    // Small delta (~1e-4) ≈ MAE behavior; 1.345 = 95% efficiency; Inf = no effect.
    Scalar robust_delta = static_cast<Scalar>(0);

    // Tweedie continuous power parameter: V(μ) = μ^power_param.
    // Only used when type == TWEEDIE. Subsumes:
    //   p=0 → Gaussian (MSE), p=1 → Poisson, p=2 → Gamma, p=3 → Inverse Gaussian.
    // Valid range: p ∈ [0, ∞), but p ∈ (0,1) is uncommon. Typical values: 1.0–3.0.
    Scalar power_param = static_cast<Scalar>(1.5);

    /**
     * @brief Check if loss requires IRLS
     * @return true for any non-Gaussian distribution or if robustness is active
     */
    bool requires_irls() const {
        return type != LossType::MSE || robust_delta > static_cast<Scalar>(0);
    }
};

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
 * @brief Compute IRLS weight for KL divergence
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
 * Based on the observed Fisher information of the GP negative log-likelihood
 * w.r.t. the mean parameter s_ij:
 *   ∂²ℓ/∂s² = 1/s² + (y-1)/(s + θy)²  for y >= 1
 *   ∂²ℓ/∂s² = 1/s²                      for y = 0
 *
 * @tparam Scalar Precision type
 * @param observed Observed count (y_ij)
 * @param predicted Predicted mean (s_ij)
 * @param theta Dispersion parameter (θ >= 0)
 * @param blend GP/KL blend factor (0=KL weight 1/s, 1=full GP weight 1/s²+...)
 * @param eps Stability floor
 * @return IRLS weight
 */
template<typename Scalar>
inline Scalar irls_weight_gp(Scalar observed, Scalar predicted, Scalar theta,
                             Scalar blend = static_cast<Scalar>(1),
                             Scalar eps = tiny_num<Scalar>()) {
    // Use fp64 internally to avoid fp32 overflow in 1/s² terms.
    double s = std::max(static_cast<double>(predicted), static_cast<double>(eps));
    double y = static_cast<double>(observed);
    double th = static_cast<double>(theta);
    double bl = static_cast<double>(blend);

    // Per-entry adaptive blend: reduce blend for small predictions
    // to prevent 1/s² from dominating the IRLS update.
    // For s < 1.0, scale blend linearly: eff_blend = bl * min(s, 1.0)
    double eff_blend = bl * std::min(s, 1.0);

    double w_gp = 1.0 / (s * s);
    if (y >= 1.0) {
        double denom = s + th * y;
        denom = std::max(denom, static_cast<double>(eps));
        w_gp += (y - 1.0) / (denom * denom);
    }
    // Geometric interpolation with KL weight: w = w_KL^(1-blend) * w_GP^blend
    // Using log-space: log(w) = (1-blend)*log(w_KL) + blend*log(w_GP)
    if (eff_blend < 0.999) {
        double log_w_kl = -std::log(s);
        double log_w_gp = std::log(std::max(w_gp, 1e-300));
        double weight = std::exp((1.0 - eff_blend) * log_w_kl + eff_blend * log_w_gp);
        weight = std::min(weight, 1e6);
        return static_cast<Scalar>(weight);
    }
    w_gp = std::min(w_gp, 1e6);
    return static_cast<Scalar>(w_gp);
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
 * - r → 0 gives heavy overdispersion: w → 0
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
    double mu = std::max(static_cast<double>(predicted), static_cast<double>(eps));
    double r = std::max(static_cast<double>(nb_size), 1e-10);
    double w = r / (mu * (r + mu));
    w = std::min(w, 1e6);
    return static_cast<Scalar>(w);
}

/**
 * @brief Compute IRLS weight for power-variance family: V(μ) = μ^p
 *
 * Unifies Gaussian (p=0), Poisson (p=1), Gamma (p=2), InvGaussian (p=3).
 * Weight = 1/μ^p (Fisher information of the corresponding EDM).
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
    double mu = std::max(static_cast<double>(predicted), static_cast<double>(eps));
    double w = 1.0 / std::pow(mu, static_cast<double>(power));
    w = std::min(w, 1e6);
    return static_cast<Scalar>(w);
}

/**
 * @brief Compute Huber-style robust modifier on Pearson residuals
 *
 * Given a Pearson residual r_P = (y − μ) / √V(μ), apply Huber trimming:
 *   w_robust = min(1, δ / |r_P|)
 *
 * As δ → 0, this approaches 1/|r_P| (MAE-like behavior).
 * This modifier multiplies the distribution weight: w_final = w_dist × w_robust.
 *
 * @tparam Scalar Precision type
 * @param pearson_residual Standardized residual r_P
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
 * GP negative log-likelihood (up to constant log(y!) which is data-dependent):
 *   -log(s/(1+θ)) - (y-1)*log((s+θy)/(1+θ)) + (s+θy)/(1+θ)
 *
 * @tparam Scalar Precision type
 * @param observed Observed count y_ij
 * @param predicted Predicted mean s_ij = Σ_k w_ik h_jk
 * @param theta Dispersion parameter θ >= 0
 * @param eps Stability floor
 * @return GP negative log-likelihood contribution
 */
template<typename Scalar>
inline Scalar loss_contribution_gp(Scalar observed, Scalar predicted, Scalar theta,
                                   Scalar eps = tiny_num<Scalar>()) {
    // Use fp64 internally for numerical stability in log/exp terms
    double s = std::max(static_cast<double>(predicted), 1e-10);
    double y = static_cast<double>(observed);
    double th = static_cast<double>(theta);
    double one_plus_theta = 1.0 + th;
    double loss = -std::log(s / one_plus_theta);
    if (y >= 1.0) {
        double inner = (s + th * y) / one_plus_theta;
        inner = std::max(inner, 1e-10);
        loss -= (y - 1.0) * std::log(inner);
    }
    loss += (s + th * y) / one_plus_theta;
    return static_cast<Scalar>(loss);
}

/**
 * @brief Compute loss contribution for single observation (Negative Binomial)
 *
 * NB negative log-likelihood (up to y-dependent constant lgamma(y+1)):
 *   -lgamma(y+r) + lgamma(r) - r·log(r/(r+μ)) - y·log(μ/(r+μ))
 *
 * where r is the inverse-dispersion parameter ("size").
 * When r→∞, this converges to the Poisson NLL: μ - y·log(μ).
 *
 * @tparam Scalar Precision type
 * @param observed Observed count y
 * @param predicted Predicted mean μ = (WH)_ij
 * @param nb_size Inverse-dispersion parameter r > 0
 * @return NB negative log-likelihood contribution
 */
template<typename Scalar>
inline Scalar loss_contribution_nb(Scalar observed, Scalar predicted, Scalar nb_size) {
    double y = static_cast<double>(observed);
    double mu = std::max(static_cast<double>(predicted), 1e-10);
    double r = std::max(static_cast<double>(nb_size), 1e-10);
    // NLL = -lgamma(y+r) + lgamma(r) + lgamma(y+1) - r*log(r/(r+mu)) - y*log(mu/(r+mu))
    // We omit lgamma(y+1) since it's constant w.r.t. model parameters
    double nll = -std::lgamma(y + r) + std::lgamma(r)
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
    double y = std::max(static_cast<double>(observed), 1e-10);
    double mu = std::max(static_cast<double>(predicted), 1e-10);
    return static_cast<Scalar>(2.0 * (-std::log(y / mu) + (y - mu) / mu));
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
    double y = std::max(static_cast<double>(observed), 1e-10);
    double mu = std::max(static_cast<double>(predicted), 1e-10);
    double diff = y - mu;
    return static_cast<Scalar>(diff * diff / (mu * mu * y));
}

/**
 * @brief Compute loss contribution for single observation (Tweedie deviance)
 *
 * Tweedie unit deviance for variance function V(μ) = μ^p.
 * Handles special cases p=1 (Poisson) and p=2 (Gamma) analytically.
 * General formula for p ∉ {1,2}:
 *   d(y, μ) = 2 [ y^{2-p}/((1-p)(2-p)) - y·μ^{1-p}/(1-p) + μ^{2-p}/(2-p) ]
 *
 * @tparam Scalar Precision type
 * @param observed Observed value y (must be non-negative)
 * @param predicted Predicted value μ (must be positive)
 * @param p Variance power parameter
 * @return Tweedie deviance contribution
 */
template<typename Scalar>
inline Scalar loss_contribution_tweedie(Scalar observed, Scalar predicted, Scalar p) {
    double y = std::max(static_cast<double>(observed), 1e-10);
    double mu = std::max(static_cast<double>(predicted), 1e-10);
    double pp = static_cast<double>(p);

    // Special case: p ≈ 1 (Poisson deviance)
    if (std::abs(pp - 1.0) < 1e-6) {
        return static_cast<Scalar>(2.0 * (y * std::log(y / mu) - (y - mu)));
    }
    // Special case: p ≈ 2 (Gamma deviance)
    if (std::abs(pp - 2.0) < 1e-6) {
        return static_cast<Scalar>(2.0 * (-std::log(y / mu) + (y - mu) / mu));
    }
    // General Tweedie deviance
    double omp = 1.0 - pp;   // 1 - p
    double tmp = 2.0 - pp;   // 2 - p
    double term1 = std::pow(y, tmp) / (omp * tmp);
    double term2 = y * std::pow(mu, omp) / omp;
    double term3 = std::pow(mu, tmp) / tmp;
    return static_cast<Scalar>(2.0 * (term1 - term2 + term3));
}

/**
 * @brief Compute total loss based on configuration
 * 
 * @tparam Scalar Precision type
 * @param observed Observed value
 * @param predicted Predicted value
 * @param config Loss configuration
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
 * @brief Compute robust loss for a single observation
 *
 * When robust_delta > 0, applies the Huber rho-function to the Pearson
 * residual r_P = (y - μ) / sqrt(V(μ)), where V(μ) is determined by the
 * distribution. The robust objective is: Σ ρ_H(r_P) where
 *   ρ_H(r) = r²/2              if |r| ≤ δ
 *           = δ|r| - δ²/2       if |r| > δ
 *
 * When robust_delta <= 0, returns the standard distribution deviance.
 */
template<typename Scalar>
inline Scalar compute_robust_loss(Scalar observed, Scalar predicted,
                                   const LossConfig<Scalar>& config,
                                   Scalar theta = static_cast<Scalar>(0)) {
    if (config.robust_delta <= static_cast<Scalar>(0)) {
        return compute_loss(observed, predicted, config, theta);
    }

    // Compute Pearson residual: r_P = (y - μ) / sqrt(V(μ))
    Scalar mu = std::max(predicted, static_cast<Scalar>(1e-10));
    Scalar residual = observed - mu;

    // Variance function V(μ) depends on distribution
    Scalar var_mu;
    switch (config.type) {
        case LossType::GP:
        case LossType::KL:
            var_mu = mu;  // V(μ) = μ (Poisson-like)
            break;
        case LossType::NB: {
            Scalar r = std::max(theta, static_cast<Scalar>(1e-10));
            var_mu = mu + mu * mu / r;  // V(μ) = μ + μ²/r
            break;
        }
        case LossType::GAMMA:
            var_mu = mu * mu;  // V(μ) = μ²
            break;
        case LossType::INVGAUSS:
            var_mu = mu * mu * mu;  // V(μ) = μ³
            break;
        case LossType::TWEEDIE:
            var_mu = std::pow(static_cast<double>(mu),
                              static_cast<double>(config.power_param));
            break;
        default:
            var_mu = static_cast<Scalar>(1);  // Gaussian: V(μ) = 1
            break;
    }

    Scalar sd = std::sqrt(std::max(var_mu, static_cast<Scalar>(1e-20)));
    Scalar pearson_r = residual / sd;
    Scalar abs_pr = std::abs(pearson_r);
    Scalar delta = config.robust_delta;

    // Huber rho-function
    if (abs_pr <= delta) {
        return static_cast<Scalar>(0.5) * pearson_r * pearson_r;
    } else {
        return delta * abs_pr - static_cast<Scalar>(0.5) * delta * delta;
    }
}

}  // namespace FactorNet

#endif // FactorNet_MATH_LOSS_HPP
