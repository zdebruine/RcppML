/**
 * @file gpu/bridge_nmf.hpp
 * @brief C++ bridge callers for GPU NMF (dlsym path)
 *
 * These functions are the C++ equivalent of R/gpu_backend.R's .gpu_nmf_unified()
 * and .gpu_nmf_cv_unified(). They:
 *   1. Pack NMFConfig + sparse/dense matrix into raw double arrays
 *   2. Call gpu_bridge.cu's extern "C" functions via dlsym()
 *   3. Unpack results into NMFResult<Scalar>
 *
 * Only compiled when FACTORNET_HAS_GPU is NOT defined (CPU-only build).
 * When compiled with nvcc, fit.hpp calls GPU functions directly.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#ifndef FACTORNET_HAS_GPU

#include <FactorNet/gpu/loader.hpp>
#include <FactorNet/core/config.hpp>
#include <FactorNet/core/result.hpp>
#include <FactorNet/core/traits.hpp>
#include <FactorNet/rng/rng.hpp>
#include <Eigen/Sparse>
#include <vector>
#include <stdexcept>

namespace FactorNet {
namespace gpu {

// ========================================================================
// Function pointer types for gpu_bridge.cu extern "C" functions
// ========================================================================

// rcppml_gpu_nmf_unified_double signature
using nmf_unified_double_fn_t = void(*)(
    const int*, const int*, const double*,    // CSC: col_ptr, row_idx, values
    int*, int*, int*, int*,                   // m, n, nnz, k
    double*, double*, double*,                // W, H, d (in/out)
    int*, double*,                            // max_iter, tol
    double*, double*, double*, double*,       // L1_H, L1_W, L2_H, L2_W
    double*, double*,                         // L21_H, L21_W
    double*, double*,                         // ortho_H, ortho_W
    double*, double*,                         // ub_H, ub_W
    int*, int*, int*,                         // cd_maxit, verbose, seed
    int*, int*,                               // loss_every, patience
    int*, int*,                               // nonneg_W, nonneg_H
    int*, double*,                            // loss_type, huber_delta
    int*, double*,                            // irls_max_iter, irls_tol
    int*,                                     // norm_type
    int*, int*,                               // projective, symmetric
    int*,                                     // solver_mode
    const int*, const int*, const double*,    // graph_W CSC
    int*, int*, double*,                      // graph_W_dim, graph_W_nnz, graph_W_lambda
    const int*, const int*, const double*,    // graph_H CSC
    int*, int*, double*,                      // graph_H_dim, graph_H_nnz, graph_H_lambda
    // Dispersion configuration
    int*,                                     // gp_dispersion_mode
    double*, double*, double*,                // gp_theta_init, gp_theta_max, gp_theta_min
    double*, double*, double*,                // nb_size_init, nb_size_max, nb_size_min
    double*, double*, double*,                // gamma_phi_init, gamma_phi_max, gamma_phi_min
    double*, double*,                         // robust_delta, tweedie_power
    // Dispersion output
    double*, int*,                            // out_theta, out_theta_len
    // Classifier guides (H-side, multi-guide)
    const int*, const int*,                   // guide_H_labels_flat, guide_H_ns
    const double*, const int*, int*,          // guide_H_lambdas, guide_H_ncs, guide_H_count
    // Standard outputs
    int*, int*, double*,                      // out_iter, out_converged, out_loss
    int*,                                     // out_status
    double*                                   // out_tol
);

// rcppml_gpu_nmf_cv_unified_double signature
using nmf_cv_unified_double_fn_t = void(*)(
    const int*, const int*, const double*,    // CSC
    int*, int*, int*, int*,                   // m, n, nnz, k
    double*, double*, double*,                // W, H, d
    int*, double*,                            // max_iter, tol
    double*, double*, double*, double*,       // L1_H, L1_W, L2_H, L2_W
    int*, int*, int*,                         // cd_maxit, verbose, seed
    double*, int*, int*,                      // holdout_frac, cv_seed, mask_zeros
    int*, int*,                               // nonneg_W, nonneg_H
    int*,                                     // norm_type
    int*, double*,                            // loss_type, huber_delta
    int*, double*,                            // irls_max_iter, irls_tol
    const int*, const int*, const double*,    // graph_W CSC
    int*, int*, double*,                      // graph_W_dim, graph_W_nnz, graph_W_lambda
    const int*, const int*, const double*,    // graph_H CSC
    int*, int*, double*,                      // graph_H_dim, graph_H_nnz, graph_H_lambda
    int*, int*, int*,                         // projective, symmetric, solver_mode
    int*, int*,                               // out_iter, out_converged
    double*, double*,                         // out_train_loss, out_test_loss
    double*, int*,                            // out_best_test, out_best_iter
    int*                                      // out_status
);

// rcppml_gpu_nmf_dense_unified_double signature
using nmf_dense_unified_double_fn_t = void(*)(
    const double*,                            // A_data (m×n column-major)
    int*, int*, int*,                         // m, n, k
    double*, double*, double*,                // W, H, d
    int*, double*,                            // max_iter, tol
    double*, double*, double*, double*,       // L1_H, L1_W, L2_H, L2_W
    double*, double*,                         // L21_H, L21_W
    double*, double*,                         // ortho_H, ortho_W
    double*, double*,                         // ub_H, ub_W
    int*, int*, int*,                         // cd_maxit, verbose, seed
    int*, int*,                               // loss_every, patience
    int*, int*,                               // nonneg_W, nonneg_H
    int*, double*,                            // loss_type, huber_delta
    int*, double*,                            // irls_max_iter, irls_tol
    int*,                                     // norm_type
    int*,                                     // gp_dispersion_mode
    double*, double*, double*,                // gp_theta_init, gp_theta_max, gp_theta_min
    double*, double*, double*,                // nb_size_init, nb_size_max, nb_size_min
    double*, double*,                         // robust_delta, tweedie_power
    int*, int*, int*,                         // projective, symmetric, solver_mode
    double*, int*,                            // out_theta, out_theta_len
    int*, int*, double*,                      // out_iter, out_converged, out_loss
    int*,                                     // out_status
    double*                                   // out_tol
);

// ========================================================================
// Helper: pack graph Laplacian into flat CSC arrays for bridge
// ========================================================================

struct GraphCSC {
    std::vector<int> p;
    std::vector<int> i;
    std::vector<double> x;
    int dim = 0;
    int nnz = 0;
    double lambda = 0.0;
};

template<typename Scalar>
GraphCSC pack_graph(const Eigen::SparseMatrix<Scalar>* graph, Scalar lambda) {
    GraphCSC g;
    if (!graph || graph->nonZeros() == 0) {
        g.p = {0};
        g.i = {0};
        g.x = {0.0};
        g.dim = 0;
        g.nnz = 0;
        g.lambda = static_cast<double>(lambda);
        return g;
    }
    int n = graph->cols();
    int nz = graph->nonZeros();
    g.dim = n;
    g.nnz = nz;
    g.lambda = static_cast<double>(lambda);
    g.p.assign(graph->outerIndexPtr(), graph->outerIndexPtr() + n + 1);
    g.i.assign(graph->innerIndexPtr(), graph->innerIndexPtr() + nz);
    g.x.resize(nz);
    for (int j = 0; j < nz; ++j)
        g.x[j] = static_cast<double>(graph->valuePtr()[j]);
    return g;
}

// ========================================================================
// Bridge caller: Standard NMF (sparse input)
// ========================================================================

/**
 * @brief Call GPU NMF via dlsym bridge (sparse input, double precision bridge)
 *
 * @tparam Scalar  float or double (result precision)
 * @param A        Sparse CSC matrix (m × n)
 * @param config   NMF configuration
 * @param W_init   Optional W init (m × k), nullptr = random
 * @param H_init   Optional H init (k × n), nullptr = random
 * @return         NMFResult<Scalar>
 */
template<typename Scalar, typename MatrixType>
NMFResult<Scalar> bridge_nmf_sparse(
    const MatrixType& A,
    const NMFConfig<Scalar>& config,
    const DenseMatrix<Scalar>* W_init,
    const DenseMatrix<Scalar>* H_init)
{
    auto fn = resolve<nmf_unified_double_fn_t>("rcppml_gpu_nmf_unified_float");
    if (!fn) {
        throw std::runtime_error(
            "GPU bridge function 'rcppml_gpu_nmf_unified_float' not found. "
            "Ensure RcppML_gpu.so is loaded (gpu_available() == TRUE).");
    }

    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int nnz = static_cast<int>(A.nonZeros());
    const int k = config.rank;

    // --- Convert sparse values to double ---
    std::vector<double> values_d(nnz);
    const auto* vptr = A.valuePtr();
    for (int j = 0; j < nnz; ++j)
        values_d[j] = static_cast<double>(vptr[j]);

    // --- Initialize W (flat k×m, column-major) and H (flat k×n) ---
    std::vector<double> W_flat(static_cast<size_t>(k) * m);
    std::vector<double> H_flat(static_cast<size_t>(k) * n);
    std::vector<double> d_vec(k, 1.0);

    if (W_init && W_init->rows() == m && W_init->cols() == k) {
        // W_init is m×k, pack as k×m (column-major: W[f + i*k])
        for (int i = 0; i < m; ++i)
            for (int f = 0; f < k; ++f)
                W_flat[f + static_cast<size_t>(i) * k] =
                    static_cast<double>((*W_init)(i, f));
    } else {
        rng::SplitMix64 rng(config.seed);
        rng.fill_uniform(W_flat.data(), k, m);
    }

    if (H_init && H_init->rows() == k && H_init->cols() == n) {
        for (int j = 0; j < n; ++j)
            for (int f = 0; f < k; ++f)
                H_flat[f + static_cast<size_t>(j) * k] =
                    static_cast<double>((*H_init)(f, j));
    } else {
        // Use a different seed offset for H to avoid correlation
        rng::SplitMix64 rng(config.seed + 0x9E3779B9u);
        rng.fill_uniform(H_flat.data(), k, n);
    }

    // --- Pack config scalars ---
    int max_iter = config.max_iter;
    double tol = static_cast<double>(config.tol);
    double L1_H = static_cast<double>(config.H.L1);
    double L1_W = static_cast<double>(config.W.L1);
    double L2_H = static_cast<double>(config.H.L2);
    double L2_W = static_cast<double>(config.W.L2);
    double L21_H = static_cast<double>(config.H.L21);
    double L21_W = static_cast<double>(config.W.L21);
    double ortho_H = static_cast<double>(config.H.angular);
    double ortho_W = static_cast<double>(config.W.angular);
    double ub_H = static_cast<double>(config.H.upper_bound);
    double ub_W = static_cast<double>(config.W.upper_bound);
    int cd_maxit = config.cd_max_iter;
    int verbose = config.verbose ? 1 : 0;
    int seed = static_cast<int>(config.seed);
    int loss_every = config.loss_every;
    int patience = config.patience;
    int nonneg_W = config.W.nonneg ? 1 : 0;
    int nonneg_H = config.H.nonneg ? 1 : 0;
    int loss_type = static_cast<int>(config.loss.type);
    double huber_delta = static_cast<double>(config.loss.huber_delta);
    int irls_max_iter = config.irls_max_iter;
    double irls_tol = static_cast<double>(config.irls_tol);
    int norm_type = static_cast<int>(config.norm_type);
    int projective = config.projective ? 1 : 0;
    int symmetric = config.symmetric ? 1 : 0;
    int solver_mode = config.solver_mode;

    // --- Pack dispersion config ---
    int gp_dispersion_mode = static_cast<int>(config.gp_dispersion);
    double gp_theta_init_d = static_cast<double>(config.gp_theta_init);
    double gp_theta_max_d = static_cast<double>(config.gp_theta_max);
    double gp_theta_min_d = static_cast<double>(config.gp_theta_min);
    double nb_size_init_d = static_cast<double>(config.nb_size_init);
    double nb_size_max_d = static_cast<double>(config.nb_size_max);
    double nb_size_min_d = static_cast<double>(config.nb_size_min);
    double gamma_phi_init_d = static_cast<double>(config.gamma_phi_init);
    double gamma_phi_max_d = static_cast<double>(config.gamma_phi_max);
    double gamma_phi_min_d = static_cast<double>(config.gamma_phi_min);
    double robust_delta_d = static_cast<double>(config.loss.robust_delta);
    double tweedie_power_d = static_cast<double>(config.loss.power_param);

    // --- Pack graph Laplacians ---
    auto gW = pack_graph(config.W.graph, config.W.graph_lambda);
    auto gH = pack_graph(config.H.graph, config.H.graph_lambda);

    // --- Output variables ---
    int out_iter = 0, out_converged = 0, out_status = 0;
    double out_loss = 0.0, out_tol = 0.0;

    // --- Theta/dispersion output buffer ---
    std::vector<double> theta_buf(m, 0.0);
    int out_theta_len = 0;

    // --- Pack classifier guides (H-side) ---
    std::vector<int> guide_labels_flat;
    std::vector<int> guide_ns;
    std::vector<double> guide_lambdas;
    std::vector<int> guide_ncs;
    int guide_count = static_cast<int>(config.classifier_guides_H.size());
    for (const auto& g : config.classifier_guides_H) {
        guide_labels_flat.insert(guide_labels_flat.end(),
                                 g.labels.begin(), g.labels.end());
        guide_ns.push_back(static_cast<int>(g.labels.size()));
        guide_lambdas.push_back(static_cast<double>(g.lambda));
        guide_ncs.push_back(g.n_classes);
    }
    // Ensure non-null pointers for the bridge call
    if (guide_labels_flat.empty()) guide_labels_flat.push_back(0);
    if (guide_ns.empty()) guide_ns.push_back(0);
    if (guide_lambdas.empty()) guide_lambdas.push_back(0.0);
    if (guide_ncs.empty()) guide_ncs.push_back(0);

    // --- Mutable copies of dimensions (bridge takes int*) ---
    int m_mut = m, n_mut = n, nnz_mut = nnz, k_mut = k;

    // --- Call the bridge function ---
    fn(
        A.outerIndexPtr(), A.innerIndexPtr(), values_d.data(),
        &m_mut, &n_mut, &nnz_mut, &k_mut,
        W_flat.data(), H_flat.data(), d_vec.data(),
        &max_iter, &tol,
        &L1_H, &L1_W, &L2_H, &L2_W,
        &L21_H, &L21_W,
        &ortho_H, &ortho_W,
        &ub_H, &ub_W,
        &cd_maxit, &verbose, &seed,
        &loss_every, &patience,
        &nonneg_W, &nonneg_H,
        &loss_type, &huber_delta,
        &irls_max_iter, &irls_tol,
        &norm_type,
        &projective, &symmetric,
        &solver_mode,
        gW.p.data(), gW.i.data(), gW.x.data(),
        &gW.dim, &gW.nnz, &gW.lambda,
        gH.p.data(), gH.i.data(), gH.x.data(),
        &gH.dim, &gH.nnz, &gH.lambda,
        &gp_dispersion_mode,
        &gp_theta_init_d, &gp_theta_max_d, &gp_theta_min_d,
        &nb_size_init_d, &nb_size_max_d, &nb_size_min_d,
        &gamma_phi_init_d, &gamma_phi_max_d, &gamma_phi_min_d,
        &robust_delta_d, &tweedie_power_d,
        theta_buf.data(), &out_theta_len,
        guide_labels_flat.data(), guide_ns.data(),
        guide_lambdas.data(), guide_ncs.data(), &guide_count,
        &out_iter, &out_converged, &out_loss,
        &out_status,
        &out_tol
    );

    if (out_status != 0) {
        throw std::runtime_error("GPU NMF bridge call failed (status != 0)");
    }

    // --- Unpack results into NMFResult<Scalar> ---
    NMFResult<Scalar> result;
    result.W.resize(m, k);
    result.H.resize(k, n);
    result.d.resize(k);

    // W: flat k×m → result.W (m×k)
    for (int i = 0; i < m; ++i)
        for (int f = 0; f < k; ++f)
            result.W(i, f) = static_cast<Scalar>(
                W_flat[f + static_cast<size_t>(i) * k]);

    // H: flat k×n → result.H (k×n)
    for (int j = 0; j < n; ++j)
        for (int f = 0; f < k; ++f)
            result.H(f, j) = static_cast<Scalar>(
                H_flat[f + static_cast<size_t>(j) * k]);

    // d
    for (int f = 0; f < k; ++f)
        result.d(f) = static_cast<Scalar>(d_vec[f]);

    result.iterations = out_iter;
    result.converged = (out_converged != 0);
    result.train_loss = static_cast<Scalar>(out_loss);
    result.final_tol = static_cast<Scalar>(out_tol);
    result.diagnostics.plan_chosen = "GPU (dlsym bridge)";
    result.diagnostics.plan_reason = "GPU via runtime-loaded bridge library";

    // Copy theta/dispersion if present
    if (out_theta_len > 0) {
        const auto lt = config.loss.type;
        if (lt == LossType::GP || lt == LossType::NB) {
            result.theta.resize(out_theta_len);
            for (int i = 0; i < out_theta_len; ++i)
                result.theta(i) = static_cast<Scalar>(theta_buf[i]);
        } else if (lt == LossType::GAMMA || lt == LossType::INVGAUSS ||
                   lt == LossType::TWEEDIE) {
            result.dispersion.resize(out_theta_len);
            for (int i = 0; i < out_theta_len; ++i)
                result.dispersion(i) = static_cast<Scalar>(theta_buf[i]);
        }
    }

    return result;
}

// ========================================================================
// Bridge caller: Cross-validation NMF (sparse input)
// ========================================================================

template<typename Scalar, typename MatrixType>
NMFResult<Scalar> bridge_nmf_cv_sparse(
    const MatrixType& A,
    const NMFConfig<Scalar>& config,
    const DenseMatrix<Scalar>* W_init,
    const DenseMatrix<Scalar>* H_init)
{
    auto fn = resolve<nmf_cv_unified_double_fn_t>(
        "rcppml_gpu_nmf_cv_unified_float");
    if (!fn) {
        throw std::runtime_error(
            "GPU bridge function 'rcppml_gpu_nmf_cv_unified_float' not found.");
    }

    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int nnz = static_cast<int>(A.nonZeros());
    const int k = config.rank;

    // Convert values to double
    std::vector<double> values_d(nnz);
    const auto* vptr = A.valuePtr();
    for (int j = 0; j < nnz; ++j)
        values_d[j] = static_cast<double>(vptr[j]);

    // Initialize W/H
    std::vector<double> W_flat(static_cast<size_t>(k) * m);
    std::vector<double> H_flat(static_cast<size_t>(k) * n);
    std::vector<double> d_vec(k, 1.0);

    if (W_init && W_init->rows() == m && W_init->cols() == k) {
        for (int i = 0; i < m; ++i)
            for (int f = 0; f < k; ++f)
                W_flat[f + static_cast<size_t>(i) * k] =
                    static_cast<double>((*W_init)(i, f));
    } else {
        rng::SplitMix64 rng(config.seed);
        rng.fill_uniform(W_flat.data(), k, m);
    }
    {
        rng::SplitMix64 rng(config.seed + 0x9E3779B9u);
        rng.fill_uniform(H_flat.data(), k, n);
    }

    // Pack scalars
    int max_iter = config.max_iter;
    double tol = static_cast<double>(config.tol);
    double L1_H = static_cast<double>(config.H.L1);
    double L1_W = static_cast<double>(config.W.L1);
    double L2_H = static_cast<double>(config.H.L2);
    double L2_W = static_cast<double>(config.W.L2);
    int cd_maxit = config.cd_max_iter;
    int verbose = config.verbose ? 1 : 0;
    int seed = static_cast<int>(config.seed);
    double holdout_frac = static_cast<double>(config.holdout_fraction);
    int cv_seed = static_cast<int>(config.cv_seed);
    int mask_zeros = config.mask_zeros ? 1 : 0;
    int nonneg_W = config.W.nonneg ? 1 : 0;
    int nonneg_H = config.H.nonneg ? 1 : 0;
    int norm_type = static_cast<int>(config.norm_type);
    int loss_type = static_cast<int>(config.loss.type);
    double huber_delta = static_cast<double>(config.loss.huber_delta);
    int irls_max_iter = config.irls_max_iter;
    double irls_tol = static_cast<double>(config.irls_tol);

    auto gW = pack_graph(config.W.graph, config.W.graph_lambda);
    auto gH = pack_graph(config.H.graph, config.H.graph_lambda);

    int projective_int = config.projective ? 1 : 0;
    int symmetric_int = config.symmetric ? 1 : 0;
    int solver_mode_int = config.solver_mode;

    // Output variables
    int out_iter = 0, out_converged = 0, out_status = 0;
    double out_train = 0, out_test = 0, out_best_test = 0;
    int out_best_iter = 0;

    int m_mut = m, n_mut = n, nnz_mut = nnz, k_mut = k;

    fn(
        A.outerIndexPtr(), A.innerIndexPtr(), values_d.data(),
        &m_mut, &n_mut, &nnz_mut, &k_mut,
        W_flat.data(), H_flat.data(), d_vec.data(),
        &max_iter, &tol,
        &L1_H, &L1_W, &L2_H, &L2_W,
        &cd_maxit, &verbose, &seed,
        &holdout_frac, &cv_seed, &mask_zeros,
        &nonneg_W, &nonneg_H,
        &norm_type,
        &loss_type, &huber_delta,
        &irls_max_iter, &irls_tol,
        gW.p.data(), gW.i.data(), gW.x.data(),
        &gW.dim, &gW.nnz, &gW.lambda,
        gH.p.data(), gH.i.data(), gH.x.data(),
        &gH.dim, &gH.nnz, &gH.lambda,
        &projective_int, &symmetric_int, &solver_mode_int,
        &out_iter, &out_converged,
        &out_train, &out_test,
        &out_best_test, &out_best_iter,
        &out_status
    );

    if (out_status != 0) {
        throw std::runtime_error("GPU CV NMF bridge call failed (status != 0)");
    }

    NMFResult<Scalar> result;
    result.W.resize(m, k);
    result.H.resize(k, n);
    result.d.resize(k);

    for (int i = 0; i < m; ++i)
        for (int f = 0; f < k; ++f)
            result.W(i, f) = static_cast<Scalar>(
                W_flat[f + static_cast<size_t>(i) * k]);
    for (int j = 0; j < n; ++j)
        for (int f = 0; f < k; ++f)
            result.H(f, j) = static_cast<Scalar>(
                H_flat[f + static_cast<size_t>(j) * k]);
    for (int f = 0; f < k; ++f)
        result.d(f) = static_cast<Scalar>(d_vec[f]);

    result.iterations = out_iter;
    result.converged = (out_converged != 0);
    result.train_loss = static_cast<Scalar>(out_train);
    result.test_loss = static_cast<Scalar>(out_test);
    result.best_test_loss = static_cast<Scalar>(out_best_test);
    result.best_iter = out_best_iter;
    result.diagnostics.plan_chosen = "GPU (dlsym bridge, CV)";
    result.diagnostics.plan_reason = "GPU CV via runtime-loaded bridge library";

    return result;
}

// ========================================================================
// Bridge caller: Dense NMF
// ========================================================================

template<typename Scalar>
NMFResult<Scalar> bridge_nmf_dense(
    const DenseMatrix<Scalar>& A,
    const NMFConfig<Scalar>& config,
    const DenseMatrix<Scalar>* W_init,
    const DenseMatrix<Scalar>* H_init)
{
    auto fn = resolve<nmf_dense_unified_double_fn_t>(
        "rcppml_gpu_nmf_dense_unified_float");
    if (!fn) {
        throw std::runtime_error(
            "GPU bridge function 'rcppml_gpu_nmf_dense_unified_float' not found.");
    }

    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int k = config.rank;

    // Convert A to double column-major
    std::vector<double> A_data(static_cast<size_t>(m) * n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            A_data[i + static_cast<size_t>(j) * m] = static_cast<double>(A(i, j));

    // Initialize W/H
    std::vector<double> W_flat(static_cast<size_t>(k) * m);
    std::vector<double> H_flat(static_cast<size_t>(k) * n);
    std::vector<double> d_vec(k, 1.0);

    if (W_init && W_init->rows() == m && W_init->cols() == k) {
        for (int i = 0; i < m; ++i)
            for (int f = 0; f < k; ++f)
                W_flat[f + static_cast<size_t>(i) * k] =
                    static_cast<double>((*W_init)(i, f));
    } else {
        rng::SplitMix64 rng(config.seed);
        rng.fill_uniform(W_flat.data(), k, m);
    }
    {
        rng::SplitMix64 rng(config.seed + 0x9E3779B9u);
        rng.fill_uniform(H_flat.data(), k, n);
    }

    int max_iter = config.max_iter;
    double tol = static_cast<double>(config.tol);
    double L1_H = static_cast<double>(config.H.L1);
    double L1_W = static_cast<double>(config.W.L1);
    double L2_H = static_cast<double>(config.H.L2);
    double L2_W = static_cast<double>(config.W.L2);
    double L21_H = static_cast<double>(config.H.L21);
    double L21_W = static_cast<double>(config.W.L21);
    double ortho_H = static_cast<double>(config.H.angular);
    double ortho_W = static_cast<double>(config.W.angular);
    double ub_H = static_cast<double>(config.H.upper_bound);
    double ub_W = static_cast<double>(config.W.upper_bound);
    int cd_maxit = config.cd_max_iter;
    int verbose = config.verbose ? 1 : 0;
    int seed = static_cast<int>(config.seed);
    int loss_every = config.loss_every;
    int patience = config.patience;
    int nonneg_W = config.W.nonneg ? 1 : 0;
    int nonneg_H = config.H.nonneg ? 1 : 0;
    int loss_type = static_cast<int>(config.loss.type);
    double huber_delta = static_cast<double>(config.loss.huber_delta);
    int irls_max_iter = config.irls_max_iter;
    double irls_tol = static_cast<double>(config.irls_tol);
    int norm_type = static_cast<int>(config.norm_type);
    int gp_dispersion_mode = static_cast<int>(config.gp_dispersion);
    double gp_theta_init_d = static_cast<double>(config.gp_theta_init);
    double gp_theta_max_d = static_cast<double>(config.gp_theta_max);
    double gp_theta_min_d = static_cast<double>(config.gp_theta_min);
    double nb_size_init_d = static_cast<double>(config.nb_size_init);
    double nb_size_max_d = static_cast<double>(config.nb_size_max);
    double nb_size_min_d = static_cast<double>(config.nb_size_min);
    double robust_delta_d = static_cast<double>(config.loss.robust_delta);
    double tweedie_power_d = static_cast<double>(config.loss.power_param);
    int projective_int = config.projective ? 1 : 0;
    int symmetric_int = config.symmetric ? 1 : 0;
    int solver_mode_int = config.solver_mode;

    int out_iter = 0, out_converged = 0, out_status = 0;
    double out_loss = 0.0, out_tol = 0.0;
    int m_mut = m, n_mut = n, k_mut = k;

    // Theta output buffer (max size is max(m,n))
    std::vector<double> theta_buf(std::max(m, n), 0.0);
    int out_theta_len = 0;

    fn(
        A_data.data(),
        &m_mut, &n_mut, &k_mut,
        W_flat.data(), H_flat.data(), d_vec.data(),
        &max_iter, &tol,
        &L1_H, &L1_W, &L2_H, &L2_W,
        &L21_H, &L21_W,
        &ortho_H, &ortho_W,
        &ub_H, &ub_W,
        &cd_maxit, &verbose, &seed,
        &loss_every, &patience,
        &nonneg_W, &nonneg_H,
        &loss_type, &huber_delta,
        &irls_max_iter, &irls_tol,
        &norm_type,
        &gp_dispersion_mode,
        &gp_theta_init_d, &gp_theta_max_d, &gp_theta_min_d,
        &nb_size_init_d, &nb_size_max_d, &nb_size_min_d,
        &robust_delta_d, &tweedie_power_d,
        &projective_int, &symmetric_int, &solver_mode_int,
        theta_buf.data(), &out_theta_len,
        &out_iter, &out_converged, &out_loss,
        &out_status,
        &out_tol
    );

    if (out_status != 0) {
        throw std::runtime_error("GPU dense NMF bridge call failed (status != 0)");
    }

    NMFResult<Scalar> result;
    result.W.resize(m, k);
    result.H.resize(k, n);
    result.d.resize(k);

    for (int i = 0; i < m; ++i)
        for (int f = 0; f < k; ++f)
            result.W(i, f) = static_cast<Scalar>(
                W_flat[f + static_cast<size_t>(i) * k]);
    for (int j = 0; j < n; ++j)
        for (int f = 0; f < k; ++f)
            result.H(f, j) = static_cast<Scalar>(
                H_flat[f + static_cast<size_t>(j) * k]);
    for (int f = 0; f < k; ++f)
        result.d(f) = static_cast<Scalar>(d_vec[f]);

    result.iterations = out_iter;
    result.converged = (out_converged != 0);
    result.train_loss = static_cast<Scalar>(out_loss);
    result.final_tol = static_cast<Scalar>(out_tol);
    result.diagnostics.plan_chosen = "GPU (dlsym bridge, dense)";
    result.diagnostics.plan_reason = "GPU dense via runtime-loaded bridge library";

    // Copy theta (dispersion) if present
    if (out_theta_len > 0) {
        result.theta.resize(out_theta_len);
        for (int i = 0; i < out_theta_len; ++i)
            result.theta(i) = static_cast<Scalar>(theta_buf[i]);
    }

    return result;
}

}  // namespace gpu
}  // namespace FactorNet

#endif  // !FACTORNET_HAS_GPU
