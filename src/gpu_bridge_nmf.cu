/**
 * @file gpu_bridge_nmf.cu
 * @brief GPU NMF entry points (unified double/float, CV, dense, zerocopy)
 *
 * Part of RcppML GPU bridge. Compiled by nvcc into RcppML_gpu.so.
 * Extracted from gpu_bridge.cu (2026).
 *
 * Build: make -f Makefile.gpu
 */

#include "gpu_bridge_common.cuh"

extern "C" {


/**
 * @brief Unified GPU NMF with all features (fp64).
 *
 * Calls FactorNet::nmf::nmf_fit_gpu<double> which:
 *   - Manages device-resident data (upload once, compute on GPU)
 *   - Downloads G (k×k) to host for feature application
 *   - Supports all regularization: L1, L2, L21, ortho, bounds
 *   - Uses adaptive dense GEMM or mixed precision RHS
 *
 * @param col_ptr, row_idx, values  CSC sparse matrix (host)
 * @param m, n, nnz                 Matrix dimensions
 * @param k                         Factorization rank
 * @param W                         k × m factor (in: init, out: result)
 * @param H                         k × n factor (in: init, out: result)
 * @param d                         k diagonal (in: init, out: result)
 * @param max_iter, tol             Convergence parameters
 * @param L1_H, L1_W, L2_H, L2_W   L1/L2 regularization
 * @param L21_H, L21_W              L2,1 group sparsity
 * @param ortho_H, ortho_W          Orthogonality penalty
 * @param ub_H, ub_W                Upper bound constraints
 * @param cd_maxit                  NNLS CD iterations
 * @param verbose, seed             Misc parameters
 * @param loss_every                Compute loss every N iterations
 * @param patience                  Convergence patience
 * @param out_iter, out_converged, out_loss, out_status  Outputs
 */
void rcppml_gpu_nmf_unified_double(
    const int* col_ptr, const int* row_idx, const double* values,
    int* m, int* n, int* nnz, int* k,
    double* W, double* H, double* d,
    int* max_iter, double* tol,
    double* L1_H, double* L1_W, double* L2_H, double* L2_W,
    double* L21_H, double* L21_W,
    double* ortho_H, double* ortho_W,
    double* ub_H, double* ub_W,
    int* cd_maxit, int* verbose, int* seed,
    int* loss_every, int* patience,
    int* nonneg_W, int* nonneg_H,
    int* loss_type, double* huber_delta,
    int* irls_max_iter, double* irls_tol,
    int* norm_type,
    int* projective, int* symmetric,
    int* solver_mode,
    // Graph W regularization (CSC arrays; nnz=0 means not provided)
    const int* graph_W_p, const int* graph_W_i, const double* graph_W_x,
    int* graph_W_dim, int* graph_W_nnz, double* graph_W_lambda,
    // Graph H regularization (CSC arrays; nnz=0 means not provided)
    const int* graph_H_p, const int* graph_H_i, const double* graph_H_x,
    int* graph_H_dim, int* graph_H_nnz, double* graph_H_lambda,
    // Dispersion configuration
    int* gp_dispersion_mode,
    double* gp_theta_init, double* gp_theta_max, double* gp_theta_min,
    double* nb_size_init, double* nb_size_max, double* nb_size_min,
    double* gamma_phi_init, double* gamma_phi_max, double* gamma_phi_min,
    double* robust_delta, double* tweedie_power,
    // Dispersion output
    double* out_theta, int* out_theta_len,
    // Standard outputs
    int* out_iter, int* out_converged, double* out_loss,
    int* out_status,
    double* out_tol)
{
    try {
        // Build unified config
        FactorNet::NMFConfig<double> config;
        config.rank = *k;
        config.max_iter = *max_iter;
        config.tol = *tol;
        config.H.L1 = *L1_H;
        config.W.L1 = *L1_W;
        config.H.L2 = *L2_H;
        config.W.L2 = *L2_W;
        config.H.L21 = *L21_H;
        config.W.L21 = *L21_W;
        config.H.angular = *ortho_H;
        config.W.angular = *ortho_W;
        config.H.upper_bound = *ub_H;
        config.W.upper_bound = *ub_W;
        config.cd_max_iter = *cd_maxit;
        config.verbose = (*verbose != 0);
        config.seed = static_cast<uint32_t>(*seed);
        config.loss_every = *loss_every;
        config.patience = *patience;
        config.W.nonneg = (*nonneg_W != 0);
        config.H.nonneg = (*nonneg_H != 0);
        config.loss.type = static_cast<FactorNet::LossType>(*loss_type);
        config.loss.huber_delta = *huber_delta;
        config.irls_max_iter = *irls_max_iter;
        config.irls_tol = *irls_tol;
        config.norm_type = static_cast<FactorNet::NormType>(*norm_type);
        config.projective = (*projective != 0);
        config.symmetric = (*symmetric != 0);
        config.solver_mode = *solver_mode;
        config.track_loss_history = false;

        // Dispersion configuration
        config.gp_dispersion = static_cast<FactorNet::DispersionMode>(*gp_dispersion_mode);
        config.gp_theta_init = *gp_theta_init;
        config.gp_theta_max = *gp_theta_max;
        config.gp_theta_min = *gp_theta_min;
        config.nb_size_init = *nb_size_init;
        config.nb_size_max = *nb_size_max;
        config.nb_size_min = *nb_size_min;
        config.gamma_phi_init = *gamma_phi_init;
        config.gamma_phi_max = *gamma_phi_max;
        config.gamma_phi_min = *gamma_phi_min;
        config.loss.robust_delta = *robust_delta;
        config.loss.power_param = *tweedie_power;

        // Graph regularization from raw CSC arrays
        Eigen::SparseMatrix<double> graph_W_mat, graph_H_mat;
        if (*graph_W_nnz > 0) {
            graph_W_mat = sparse_from_csc<double>(
                *graph_W_dim, *graph_W_dim, *graph_W_nnz,
                graph_W_p, graph_W_i, graph_W_x);
            config.W.graph = &graph_W_mat;
            config.W.graph_lambda = *graph_W_lambda;
        }
        if (*graph_H_nnz > 0) {
            graph_H_mat = sparse_from_csc<double>(
                *graph_H_dim, *graph_H_dim, *graph_H_nnz,
                graph_H_p, graph_H_i, graph_H_x);
            config.H.graph = &graph_H_mat;
            config.H.graph_lambda = *graph_H_lambda;
        }

        // Initialize W (k×m) and H (k×n) from input arrays
        using DenseMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
        DenseMatrix W_init = Eigen::Map<DenseMatrix>(W, *k, *m).transpose(); // k×m → m×k
        DenseMatrix H_init = Eigen::Map<DenseMatrix>(H, *k, *n);

        auto result = FactorNet::nmf::nmf_fit_gpu<double>(
            col_ptr, row_idx, values, *m, *n, *nnz,
            config, &W_init, &H_init);

        // Write results back to in/out arrays
        // W: m×k → k×m (transposed, column-major)
        for (int i = 0; i < *m; ++i)
            for (int f = 0; f < *k; ++f)
                W[f + static_cast<size_t>(i) * (*k)] = result.W(i, f);

        // H: k×n
        for (int j = 0; j < *n; ++j)
            for (int f = 0; f < *k; ++f)
                H[f + static_cast<size_t>(j) * (*k)] = result.H(f, j);

        // d
        for (int f = 0; f < *k; ++f)
            d[f] = result.d(f);

        // Return theta/dispersion
        const int tlen_theta = static_cast<int>(result.theta.size());
        const int tlen_disp = static_cast<int>(result.dispersion.size());
        if (tlen_theta > 0) {
            *out_theta_len = tlen_theta;
            for (int i = 0; i < tlen_theta; ++i)
                out_theta[i] = static_cast<double>(result.theta(i));
        } else if (tlen_disp > 0) {
            *out_theta_len = tlen_disp;
            for (int i = 0; i < tlen_disp; ++i)
                out_theta[i] = static_cast<double>(result.dispersion(i));
        } else {
            *out_theta_len = 0;
        }

        *out_iter = result.iterations;
        *out_converged = result.converged ? 1 : 0;
        *out_loss = static_cast<double>(result.train_loss);
        *out_tol = static_cast<double>(result.final_tol);
        *out_status = 0;

    } catch (const std::exception& e) {
        FACTORNET_GPU_WARN("GPU unified NMF error: %s\n", e.what());
        *out_status = -1;
    }
}

/**
 * @brief Unified GPU CV NMF with all features (fp64).
 *
 * Cross-validation NMF on GPU using the fused delta+NNLS pipeline.
 * Returns both train and test loss for rank selection.
 *
 * @param col_ptr, row_idx, values  CSC sparse matrix (host)
 * @param m, n, nnz                 Matrix dimensions
 * @param k                         Factorization rank
 * @param W, H, d                   Factors (in: init, out: result)
 * @param max_iter, tol             Convergence
 * @param L1_H, L1_W, L2_H, L2_W   Regularization
 * @param cd_maxit, verbose, seed   NNLS/misc
 * @param holdout_frac              CV holdout fraction
 * @param cv_seed                   CV mask seed
 * @param mask_zeros                Test set semantics (0/1)
 * @param out_iter, out_converged   Convergence output
 * @param out_train_loss, out_test_loss  Loss outputs
 * @param out_best_test, out_best_iter   Best test loss tracking
 * @param out_status                0=success, -1=error
 */
void rcppml_gpu_nmf_cv_unified_double(
    const int* col_ptr, const int* row_idx, const double* values,
    int* m, int* n, int* nnz, int* k,
    double* W, double* H, double* d,
    int* max_iter, double* tol,
    double* L1_H, double* L1_W, double* L2_H, double* L2_W,
    int* cd_maxit, int* verbose, int* seed,
    double* holdout_frac, int* cv_seed, int* mask_zeros,
    int* nonneg_W, int* nonneg_H,
    int* norm_type,
    int* loss_type, double* huber_delta,
    int* irls_max_iter, double* irls_tol,
    // Graph W regularization (CSC arrays; nnz=0 means not provided)
    const int* graph_W_p, const int* graph_W_i, const double* graph_W_x,
    int* graph_W_dim, int* graph_W_nnz, double* graph_W_lambda,
    // Graph H regularization (CSC arrays; nnz=0 means not provided)
    const int* graph_H_p, const int* graph_H_i, const double* graph_H_x,
    int* graph_H_dim, int* graph_H_nnz, double* graph_H_lambda,
    int* projective, int* symmetric, int* solver_mode,
    int* out_iter, int* out_converged,
    double* out_train_loss, double* out_test_loss,
    double* out_best_test, int* out_best_iter,
    int* out_status)
{
    try {
        FactorNet::NMFConfig<double> config;
        config.rank = *k;
        config.projective = (*projective != 0);
        config.symmetric = (*symmetric != 0);
        config.solver_mode = *solver_mode;
        config.max_iter = *max_iter;
        config.tol = *tol;
        config.H.L1 = *L1_H;
        config.W.L1 = *L1_W;
        config.H.L2 = *L2_H;
        config.W.L2 = *L2_W;
        config.cd_max_iter = *cd_maxit;
        config.verbose = (*verbose != 0);
        config.seed = static_cast<uint32_t>(*seed);
        config.holdout_fraction = *holdout_frac;
        config.cv_seed = static_cast<uint32_t>(*cv_seed);
        config.mask_zeros = (*mask_zeros != 0);
        config.W.nonneg = (*nonneg_W != 0);
        config.H.nonneg = (*nonneg_H != 0);
        config.norm_type = static_cast<FactorNet::NormType>(*norm_type);
        config.loss.type = static_cast<FactorNet::LossType>(*loss_type);
        config.loss.huber_delta = *huber_delta;
        config.irls_max_iter = *irls_max_iter;
        config.irls_tol = *irls_tol;
        config.track_loss_history = false;

        // Graph regularization from raw CSC arrays
        Eigen::SparseMatrix<double> graph_W_mat, graph_H_mat;
        if (*graph_W_nnz > 0) {
            graph_W_mat = sparse_from_csc<double>(
                *graph_W_dim, *graph_W_dim, *graph_W_nnz,
                graph_W_p, graph_W_i, graph_W_x);
            config.W.graph = &graph_W_mat;
            config.W.graph_lambda = *graph_W_lambda;
        }
        if (*graph_H_nnz > 0) {
            graph_H_mat = sparse_from_csc<double>(
                *graph_H_dim, *graph_H_dim, *graph_H_nnz,
                graph_H_p, graph_H_i, graph_H_x);
            config.H.graph = &graph_H_mat;
            config.H.graph_lambda = *graph_H_lambda;
        }

        using DenseMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
        DenseMatrix W_init = Eigen::Map<DenseMatrix>(W, *k, *m).transpose();
        DenseMatrix H_init = Eigen::Map<DenseMatrix>(H, *k, *n);

        auto result = FactorNet::nmf::nmf_cv_fit_gpu<double>(
            col_ptr, row_idx, values, *m, *n, *nnz,
            config, &W_init, &H_init);

        // Write factors back
        for (int i = 0; i < *m; ++i)
            for (int f = 0; f < *k; ++f)
                W[f + static_cast<size_t>(i) * (*k)] = result.W(i, f);
        for (int j = 0; j < *n; ++j)
            for (int f = 0; f < *k; ++f)
                H[f + static_cast<size_t>(j) * (*k)] = result.H(f, j);
        for (int f = 0; f < *k; ++f)
            d[f] = result.d(f);

        *out_iter = result.iterations;
        *out_converged = result.converged ? 1 : 0;
        *out_train_loss = static_cast<double>(result.train_loss);
        *out_test_loss = static_cast<double>(result.test_loss);
        *out_best_test = static_cast<double>(result.best_test_loss);
        *out_best_iter = result.best_iter;
        *out_status = 0;

    } catch (const std::exception& e) {
        FACTORNET_GPU_WARN("GPU unified CV NMF error: %s\n", e.what());
        *out_status = -1;
    }
}

/**
 * @brief Unified GPU cross-validation NMF (fp32).
 *
 * Same as rcppml_gpu_nmf_cv_unified_double but uses float precision.
 * Input/output arrays are still double (R's native type) — conversion
 * happens at the boundary.
 */
void rcppml_gpu_nmf_cv_unified_float(
    const int* col_ptr, const int* row_idx, const double* values,
    int* m, int* n, int* nnz, int* k,
    double* W, double* H, double* d,
    int* max_iter, double* tol,
    double* L1_H, double* L1_W, double* L2_H, double* L2_W,
    int* cd_maxit, int* verbose, int* seed,
    double* holdout_frac, int* cv_seed, int* mask_zeros,
    int* nonneg_W, int* nonneg_H,
    int* norm_type,
    int* loss_type, double* huber_delta,
    int* irls_max_iter, double* irls_tol,
    // Graph W regularization (CSC arrays; nnz=0 means not provided)
    const int* graph_W_p, const int* graph_W_i, const double* graph_W_x,
    int* graph_W_dim, int* graph_W_nnz, double* graph_W_lambda,
    // Graph H regularization (CSC arrays; nnz=0 means not provided)
    const int* graph_H_p, const int* graph_H_i, const double* graph_H_x,
    int* graph_H_dim, int* graph_H_nnz, double* graph_H_lambda,
    int* projective, int* symmetric, int* solver_mode,
    int* out_iter, int* out_converged,
    double* out_train_loss, double* out_test_loss,
    double* out_best_test, int* out_best_iter,
    int* out_status)
{
    try {
        FactorNet::NMFConfig<float> config;
        config.rank = *k;
        config.projective = (*projective != 0);
        config.symmetric = (*symmetric != 0);
        config.solver_mode = *solver_mode;
        config.max_iter = *max_iter;
        config.tol = static_cast<float>(*tol);
        config.H.L1 = static_cast<float>(*L1_H);
        config.W.L1 = static_cast<float>(*L1_W);
        config.H.L2 = static_cast<float>(*L2_H);
        config.W.L2 = static_cast<float>(*L2_W);
        config.cd_max_iter = *cd_maxit;
        config.verbose = (*verbose != 0);
        config.seed = static_cast<uint32_t>(*seed);
        config.holdout_fraction = static_cast<float>(*holdout_frac);
        config.cv_seed = static_cast<uint32_t>(*cv_seed);
        config.mask_zeros = (*mask_zeros != 0);
        config.W.nonneg = (*nonneg_W != 0);
        config.H.nonneg = (*nonneg_H != 0);
        config.norm_type = static_cast<FactorNet::NormType>(*norm_type);
        config.loss.type = static_cast<FactorNet::LossType>(*loss_type);
        config.loss.huber_delta = static_cast<float>(*huber_delta);
        config.irls_max_iter = *irls_max_iter;
        config.irls_tol = static_cast<float>(*irls_tol);
        config.track_loss_history = false;

        // Graph regularization from raw CSC arrays (cast to float)
        Eigen::SparseMatrix<float> graph_W_mat, graph_H_mat;
        if (*graph_W_nnz > 0) {
            graph_W_mat = sparse_from_csc<float>(
                *graph_W_dim, *graph_W_dim, *graph_W_nnz,
                graph_W_p, graph_W_i, graph_W_x);
            config.W.graph = &graph_W_mat;
            config.W.graph_lambda = static_cast<float>(*graph_W_lambda);
        }
        if (*graph_H_nnz > 0) {
            graph_H_mat = sparse_from_csc<float>(
                *graph_H_dim, *graph_H_dim, *graph_H_nnz,
                graph_H_p, graph_H_i, graph_H_x);
            config.H.graph = &graph_H_mat;
            config.H.graph_lambda = static_cast<float>(*graph_H_lambda);
        }

        // Convert input values from double to float
        const int kk = *k, mm = *m, nn = *n;
        std::vector<float> values_f(*nnz);
        for (int i = 0; i < *nnz; ++i) values_f[i] = static_cast<float>(values[i]);

        using DenseMatrixF = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
        DenseMatrixF W_init(mm, kk), H_init(kk, nn);

        for (int i = 0; i < mm; ++i)
            for (int f = 0; f < kk; ++f)
                W_init(i, f) = static_cast<float>(W[f + static_cast<size_t>(i) * kk]);
        for (int j = 0; j < nn; ++j)
            for (int f = 0; f < kk; ++f)
                H_init(f, j) = static_cast<float>(H[f + static_cast<size_t>(j) * kk]);

        auto result = FactorNet::nmf::nmf_cv_fit_gpu<float>(
            col_ptr, row_idx, values_f.data(), mm, nn, *nnz,
            config, &W_init, &H_init);

        // Convert results back to double
        for (int i = 0; i < mm; ++i)
            for (int f = 0; f < kk; ++f)
                W[f + static_cast<size_t>(i) * kk] = static_cast<double>(result.W(i, f));
        for (int j = 0; j < nn; ++j)
            for (int f = 0; f < kk; ++f)
                H[f + static_cast<size_t>(j) * kk] = static_cast<double>(result.H(f, j));
        for (int f = 0; f < kk; ++f)
            d[f] = static_cast<double>(result.d(f));

        *out_iter = result.iterations;
        *out_converged = result.converged ? 1 : 0;
        *out_train_loss = static_cast<double>(result.train_loss);
        *out_test_loss = static_cast<double>(result.test_loss);
        *out_best_test = static_cast<double>(result.best_test_loss);
        *out_best_iter = result.best_iter;
        *out_status = 0;

    } catch (const std::exception& e) {
        FACTORNET_GPU_WARN("GPU unified CV NMF (fp32) error: %s\n", e.what());
        *out_status = -1;
    }
}


/**
 * @brief Unified GPU NMF with all features (fp32).
 *
 * Same as rcppml_gpu_nmf_unified_double but uses float precision.
 * Input/output arrays are still double (R's native type) — conversion
 * happens at the boundary.
 * Signature matches _double exactly for ABI compatibility with bridge.
 */
void rcppml_gpu_nmf_unified_float(
    const int* col_ptr, const int* row_idx, const double* values,
    int* m, int* n, int* nnz, int* k,
    double* W, double* H, double* d,
    int* max_iter, double* tol,
    double* L1_H, double* L1_W, double* L2_H, double* L2_W,
    double* L21_H, double* L21_W,
    double* ortho_H, double* ortho_W,
    double* ub_H, double* ub_W,
    int* cd_maxit, int* verbose, int* seed,
    int* loss_every, int* patience,
    int* nonneg_W, int* nonneg_H,
    int* loss_type, double* huber_delta,
    int* irls_max_iter, double* irls_tol,
    int* norm_type,
    int* projective, int* symmetric,
    int* solver_mode,
    // Graph W regularization (CSC arrays; nnz=0 means not provided)
    const int* graph_W_p, const int* graph_W_i, const double* graph_W_x,
    int* graph_W_dim, int* graph_W_nnz, double* graph_W_lambda,
    // Graph H regularization (CSC arrays; nnz=0 means not provided)
    const int* graph_H_p, const int* graph_H_i, const double* graph_H_x,
    int* graph_H_dim, int* graph_H_nnz, double* graph_H_lambda,
    // Dispersion configuration (matches _double signature exactly)
    int* gp_dispersion_mode,
    double* gp_theta_init, double* gp_theta_max, double* gp_theta_min,
    double* nb_size_init, double* nb_size_max, double* nb_size_min,
    double* gamma_phi_init, double* gamma_phi_max, double* gamma_phi_min,
    double* robust_delta, double* tweedie_power,
    // Dispersion output
    double* out_theta, int* out_theta_len,
    // Standard outputs
    int* out_iter, int* out_converged, double* out_loss,
    int* out_status,
    double* out_tol)
{
    try {
        // Build config in float precision
        FactorNet::NMFConfig<float> config;
        config.rank = *k;
        config.max_iter = *max_iter;
        config.tol = static_cast<float>(*tol);
        config.H.L1 = static_cast<float>(*L1_H);
        config.W.L1 = static_cast<float>(*L1_W);
        config.H.L2 = static_cast<float>(*L2_H);
        config.W.L2 = static_cast<float>(*L2_W);
        config.H.L21 = static_cast<float>(*L21_H);
        config.W.L21 = static_cast<float>(*L21_W);
        config.H.angular = static_cast<float>(*ortho_H);
        config.W.angular = static_cast<float>(*ortho_W);
        config.H.upper_bound = static_cast<float>(*ub_H);
        config.W.upper_bound = static_cast<float>(*ub_W);
        config.cd_max_iter = *cd_maxit;
        config.verbose = (*verbose != 0);
        config.seed = static_cast<uint32_t>(*seed);
        config.loss_every = *loss_every;
        config.patience = *patience;
        config.W.nonneg = (*nonneg_W != 0);
        config.H.nonneg = (*nonneg_H != 0);
        config.loss.type = static_cast<FactorNet::LossType>(*loss_type);
        config.loss.huber_delta = static_cast<float>(*huber_delta);
        config.irls_max_iter = *irls_max_iter;
        config.irls_tol = static_cast<float>(*irls_tol);
        config.norm_type = static_cast<FactorNet::NormType>(*norm_type);
        config.projective = (*projective != 0);
        config.symmetric = (*symmetric != 0);
        config.solver_mode = *solver_mode;
        config.track_loss_history = false;
        // Dispersion parameters
        config.gp_dispersion = static_cast<FactorNet::DispersionMode>(*gp_dispersion_mode);
        config.gp_theta_init = static_cast<float>(*gp_theta_init);
        config.gp_theta_max = static_cast<float>(*gp_theta_max);
        config.gp_theta_min = static_cast<float>(*gp_theta_min);
        config.nb_size_init = static_cast<float>(*nb_size_init);
        config.nb_size_max = static_cast<float>(*nb_size_max);
        config.nb_size_min = static_cast<float>(*nb_size_min);
        config.gamma_phi_init = static_cast<float>(*gamma_phi_init);
        config.gamma_phi_max = static_cast<float>(*gamma_phi_max);
        config.gamma_phi_min = static_cast<float>(*gamma_phi_min);
        config.loss.robust_delta = static_cast<float>(*robust_delta);
        config.loss.power_param = static_cast<float>(*tweedie_power);

        // Graph regularization from raw CSC arrays (cast to float)
        Eigen::SparseMatrix<float> graph_W_mat, graph_H_mat;
        if (*graph_W_nnz > 0) {
            graph_W_mat = sparse_from_csc<float>(
                *graph_W_dim, *graph_W_dim, *graph_W_nnz,
                graph_W_p, graph_W_i, graph_W_x);
            config.W.graph = &graph_W_mat;
            config.W.graph_lambda = static_cast<float>(*graph_W_lambda);
        }
        if (*graph_H_nnz > 0) {
            graph_H_mat = sparse_from_csc<float>(
                *graph_H_dim, *graph_H_dim, *graph_H_nnz,
                graph_H_p, graph_H_i, graph_H_x);
            config.H.graph = &graph_H_mat;
            config.H.graph_lambda = static_cast<float>(*graph_H_lambda);
        }

        // Convert double input to float for initialization
        const int kk = *k, mm = *m, nn = *n;
        std::vector<float> values_f(*nnz);
        for (int i = 0; i < *nnz; ++i) values_f[i] = static_cast<float>(values[i]);

        using DenseMatrixF = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
        DenseMatrixF W_init_f(mm, kk), H_init_f(kk, nn);

        // W input is k×m (column-major), transpose to m×k
        for (int i = 0; i < mm; ++i)
            for (int f = 0; f < kk; ++f)
                W_init_f(i, f) = static_cast<float>(W[f + static_cast<size_t>(i) * kk]);
        for (int j = 0; j < nn; ++j)
            for (int f = 0; f < kk; ++f)
                H_init_f(f, j) = static_cast<float>(H[f + static_cast<size_t>(j) * kk]);

        auto result = FactorNet::nmf::nmf_fit_gpu<float>(
            col_ptr, row_idx, values_f.data(), mm, nn, *nnz,
            config, &W_init_f, &H_init_f);

        // Convert results back to double
        for (int i = 0; i < mm; ++i)
            for (int f = 0; f < kk; ++f)
                W[f + static_cast<size_t>(i) * kk] = static_cast<double>(result.W(i, f));
        for (int j = 0; j < nn; ++j)
            for (int f = 0; f < kk; ++f)
                H[f + static_cast<size_t>(j) * kk] = static_cast<double>(result.H(f, j));
        for (int f = 0; f < kk; ++f)
            d[f] = static_cast<double>(result.d(f));

        // Write theta (dispersion) back
        const int tlen = static_cast<int>(result.theta.size());
        *out_theta_len = tlen;
        for (int i = 0; i < tlen; ++i)
            out_theta[i] = static_cast<double>(result.theta(i));

        *out_iter = result.iterations;
        *out_converged = result.converged ? 1 : 0;
        *out_loss = static_cast<double>(result.train_loss);
        *out_tol = static_cast<double>(result.final_tol);
        *out_status = 0;

    } catch (const std::exception& e) {
        FACTORNET_GPU_WARN("GPU unified NMF (fp32) error: %s\n", e.what());
        *out_status = -1;
    }
}

/**
 * @brief Unified GPU NMF for dense input (fp64).
 *
 * Accepts a dense matrix instead of CSC sparse. The matrix data is
 * passed as a flat column-major array (R's native storage).
 *
 * @param A_data   Dense matrix data (m × n, column-major)
 * @param m, n     Matrix dimensions
 * @param k        Factorization rank
 * @param W, H, d  Factors (in: init, out: result)
 * @param ... remaining parameters same as sparse version
 */
void rcppml_gpu_nmf_dense_unified_double(
    const double* A_data,
    int* m, int* n, int* k,
    double* W, double* H, double* d,
    int* max_iter, double* tol,
    double* L1_H, double* L1_W, double* L2_H, double* L2_W,
    double* L21_H, double* L21_W,
    double* ortho_H, double* ortho_W,
    double* ub_H, double* ub_W,
    int* cd_maxit, int* verbose, int* seed,
    int* loss_every, int* patience,
    int* nonneg_W, int* nonneg_H,
    int* loss_type, double* huber_delta,
    int* irls_max_iter, double* irls_tol,
    int* norm_type,
    int* gp_dispersion_mode,
    double* gp_theta_init, double* gp_theta_max, double* gp_theta_min,
    double* nb_size_init, double* nb_size_max, double* nb_size_min,
    double* robust_delta, double* tweedie_power,
    int* projective, int* symmetric, int* solver_mode,
    double* out_theta, int* out_theta_len,
    int* out_iter, int* out_converged, double* out_loss,
    int* out_status,
    double* out_tol)
{
    try {
        FactorNet::NMFConfig<double> config;
        config.rank = *k;
        config.max_iter = *max_iter;
        config.tol = *tol;
        config.H.L1 = *L1_H;
        config.W.L1 = *L1_W;
        config.H.L2 = *L2_H;
        config.W.L2 = *L2_W;
        config.H.L21 = *L21_H;
        config.W.L21 = *L21_W;
        config.H.angular = *ortho_H;
        config.W.angular = *ortho_W;
        config.H.upper_bound = *ub_H;
        config.W.upper_bound = *ub_W;
        config.cd_max_iter = *cd_maxit;
        config.verbose = (*verbose != 0);
        config.seed = static_cast<uint32_t>(*seed);
        config.loss_every = *loss_every;
        config.patience = *patience;
        config.W.nonneg = (*nonneg_W != 0);
        config.H.nonneg = (*nonneg_H != 0);
        config.loss.type = static_cast<FactorNet::LossType>(*loss_type);
        config.loss.huber_delta = *huber_delta;
        config.irls_max_iter = *irls_max_iter;
        config.irls_tol = *irls_tol;
        config.norm_type = static_cast<FactorNet::NormType>(*norm_type);
        config.track_loss_history = false;
        config.gp_dispersion = static_cast<FactorNet::DispersionMode>(*gp_dispersion_mode);
        config.gp_theta_init = *gp_theta_init;
        config.gp_theta_max = *gp_theta_max;
        config.gp_theta_min = *gp_theta_min;
        config.nb_size_init = *nb_size_init;
        config.nb_size_max = *nb_size_max;
        config.nb_size_min = *nb_size_min;
        config.loss.robust_delta = *robust_delta;
        config.loss.power_param = *tweedie_power;
        config.projective = (*projective != 0);
        config.symmetric = (*symmetric != 0);
        config.solver_mode = *solver_mode;

        using DenseMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
        DenseMatrix W_init = Eigen::Map<DenseMatrix>(W, *k, *m).transpose();
        DenseMatrix H_init = Eigen::Map<DenseMatrix>(H, *k, *n);

        auto result = FactorNet::nmf::nmf_fit_gpu_dense<double>(
            A_data, *m, *n,
            config, &W_init, &H_init);

        // Write factors back
        for (int i = 0; i < *m; ++i)
            for (int f = 0; f < *k; ++f)
                W[f + static_cast<size_t>(i) * (*k)] = result.W(i, f);
        for (int j = 0; j < *n; ++j)
            for (int f = 0; f < *k; ++f)
                H[f + static_cast<size_t>(j) * (*k)] = result.H(f, j);
        for (int f = 0; f < *k; ++f)
            d[f] = result.d(f);

        // Write theta (dispersion) back
        const int tlen = static_cast<int>(result.theta.size());
        *out_theta_len = tlen;
        for (int i = 0; i < tlen; ++i)
            out_theta[i] = static_cast<double>(result.theta(i));

        *out_iter = result.iterations;
        *out_converged = result.converged ? 1 : 0;
        *out_loss = static_cast<double>(result.train_loss);
        *out_tol = static_cast<double>(result.final_tol);
        *out_status = 0;

    } catch (const std::exception& e) {
        FACTORNET_GPU_WARN("GPU dense unified NMF error: %s\n", e.what());
        *out_status = -1;
    }
}

// ============================================================================
// Dense GPU NMF (fp32)
// ============================================================================

/**
 * @brief Unified GPU NMF for dense input (fp32).
 *
 * Same as rcppml_gpu_nmf_dense_unified_double but uses float precision.
 * Input/output arrays are still double (R's native type) — conversion
 * happens at the boundary.
 */
void rcppml_gpu_nmf_dense_unified_float(
    const double* A_data,
    int* m, int* n, int* k,
    double* W, double* H, double* d,
    int* max_iter, double* tol,
    double* L1_H, double* L1_W, double* L2_H, double* L2_W,
    double* L21_H, double* L21_W,
    double* ortho_H, double* ortho_W,
    double* ub_H, double* ub_W,
    int* cd_maxit, int* verbose, int* seed,
    int* loss_every, int* patience,
    int* nonneg_W, int* nonneg_H,
    int* loss_type, double* huber_delta,
    int* irls_max_iter, double* irls_tol,
    int* norm_type,
    int* gp_dispersion_mode,
    double* gp_theta_init, double* gp_theta_max, double* gp_theta_min,
    double* nb_size_init, double* nb_size_max, double* nb_size_min,
    double* robust_delta, double* tweedie_power,
    int* projective, int* symmetric, int* solver_mode,
    double* out_theta, int* out_theta_len,
    int* out_iter, int* out_converged, double* out_loss,
    int* out_status,
    double* out_tol)
{
    try {
        FactorNet::NMFConfig<float> config;
        config.rank = *k;
        config.max_iter = *max_iter;
        config.tol = static_cast<float>(*tol);
        config.H.L1 = static_cast<float>(*L1_H);
        config.W.L1 = static_cast<float>(*L1_W);
        config.H.L2 = static_cast<float>(*L2_H);
        config.W.L2 = static_cast<float>(*L2_W);
        config.H.L21 = static_cast<float>(*L21_H);
        config.W.L21 = static_cast<float>(*L21_W);
        config.H.angular = static_cast<float>(*ortho_H);
        config.W.angular = static_cast<float>(*ortho_W);
        config.H.upper_bound = static_cast<float>(*ub_H);
        config.W.upper_bound = static_cast<float>(*ub_W);
        config.cd_max_iter = *cd_maxit;
        config.verbose = (*verbose != 0);
        config.seed = static_cast<uint32_t>(*seed);
        config.loss_every = *loss_every;
        config.patience = *patience;
        config.W.nonneg = (*nonneg_W != 0);
        config.H.nonneg = (*nonneg_H != 0);
        config.loss.type = static_cast<FactorNet::LossType>(*loss_type);
        config.loss.huber_delta = static_cast<float>(*huber_delta);
        config.irls_max_iter = *irls_max_iter;
        config.irls_tol = static_cast<float>(*irls_tol);
        config.norm_type = static_cast<FactorNet::NormType>(*norm_type);
        config.track_loss_history = false;
        config.gp_dispersion = static_cast<FactorNet::DispersionMode>(*gp_dispersion_mode);
        config.gp_theta_init = static_cast<float>(*gp_theta_init);
        config.gp_theta_max = static_cast<float>(*gp_theta_max);
        config.gp_theta_min = static_cast<float>(*gp_theta_min);
        config.nb_size_init = static_cast<float>(*nb_size_init);
        config.nb_size_max = static_cast<float>(*nb_size_max);
        config.nb_size_min = static_cast<float>(*nb_size_min);
        config.loss.robust_delta = static_cast<float>(*robust_delta);
        config.loss.power_param = static_cast<float>(*tweedie_power);
        config.projective = (*projective != 0);
        config.symmetric = (*symmetric != 0);
        config.solver_mode = *solver_mode;

        // Convert dense matrix from double to float
        const int kk = *k, mm = *m, nn = *n;
        const size_t mn = static_cast<size_t>(mm) * nn;
        std::vector<float> A_f(mn);
        for (size_t i = 0; i < mn; ++i) A_f[i] = static_cast<float>(A_data[i]);

        using DenseMatrixF = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
        DenseMatrixF W_init(mm, kk), H_init(kk, nn);

        for (int i = 0; i < mm; ++i)
            for (int f = 0; f < kk; ++f)
                W_init(i, f) = static_cast<float>(W[f + static_cast<size_t>(i) * kk]);
        for (int j = 0; j < nn; ++j)
            for (int f = 0; f < kk; ++f)
                H_init(f, j) = static_cast<float>(H[f + static_cast<size_t>(j) * kk]);

        auto result = FactorNet::nmf::nmf_fit_gpu_dense<float>(
            A_f.data(), mm, nn,
            config, &W_init, &H_init);

        // Convert results back to double
        for (int i = 0; i < mm; ++i)
            for (int f = 0; f < kk; ++f)
                W[f + static_cast<size_t>(i) * kk] = static_cast<double>(result.W(i, f));
        for (int j = 0; j < nn; ++j)
            for (int f = 0; f < kk; ++f)
                H[f + static_cast<size_t>(j) * kk] = static_cast<double>(result.H(f, j));
        for (int f = 0; f < kk; ++f)
            d[f] = static_cast<double>(result.d(f));

        // Write theta (dispersion) back
        const int tlen = static_cast<int>(result.theta.size());
        *out_theta_len = tlen;
        for (int i = 0; i < tlen; ++i)
            out_theta[i] = static_cast<double>(result.theta(i));

        *out_iter = result.iterations;
        *out_converged = result.converged ? 1 : 0;
        *out_loss = static_cast<double>(result.train_loss);
        *out_tol = static_cast<double>(result.final_tol);
        *out_status = 0;

    } catch (const std::exception& e) {
        FACTORNET_GPU_WARN("GPU dense unified NMF (fp32) error: %s\n", e.what());
        *out_status = -1;
    }
}

// ============================================================================
// Zero-copy GPU NMF (sp_read_gpu → nmf directly)
// ============================================================================

/**
 * @brief Zero-copy GPU NMF: accepts device pointers from sp_read_gpu().
 *
 * The CSC arrays (col_ptr, row_idx, values) are already on GPU device
 * memory. This avoids the host→device upload that the standard bridge
 * performs. Only double precision is supported (sp_read_gpu returns double).
 *
 * Device pointer addresses are passed as doubles since R has no int64.
 * They are reinterpreted as raw device pointers.
 */
void rcppml_gpu_nmf_zerocopy_double(
    double* d_col_ptr_addr, double* d_row_idx_addr, double* d_values_addr,
    int* m, int* n, double* nnz_d, int* k,
    double* W, double* H, double* d,
    int* max_iter, double* tol,
    double* L1_H, double* L1_W, double* L2_H, double* L2_W,
    double* L21_H, double* L21_W,
    double* ortho_H, double* ortho_W,
    double* ub_H, double* ub_W,
    int* cd_maxit, int* verbose, int* seed,
    int* loss_every, int* patience,
    int* nonneg_W, int* nonneg_H,
    int* loss_type, double* huber_delta,
    int* irls_max_iter, double* irls_tol,
    int* norm_type,
    int* out_iter, int* out_converged, double* out_loss,
    int* out_status,
    double* out_tol)
{
    try {
        // Reconstruct device pointers from double-encoded addresses
        auto to_ptr = [](double addr) -> void* {
            return reinterpret_cast<void*>(static_cast<uintptr_t>(addr));
        };
        int* dev_col_ptr = static_cast<int*>(to_ptr(*d_col_ptr_addr));
        int* dev_row_idx = static_cast<int*>(to_ptr(*d_row_idx_addr));
        double* dev_values = static_cast<double*>(to_ptr(*d_values_addr));

        int nnz = static_cast<int>(*nnz_d);

        // Build unified config
        FactorNet::NMFConfig<double> config;
        config.rank = *k;
        config.max_iter = *max_iter;
        config.tol = *tol;
        config.H.L1 = *L1_H;
        config.W.L1 = *L1_W;
        config.H.L2 = *L2_H;
        config.W.L2 = *L2_W;
        config.H.L21 = *L21_H;
        config.W.L21 = *L21_W;
        config.H.angular = *ortho_H;
        config.W.angular = *ortho_W;
        config.H.upper_bound = *ub_H;
        config.W.upper_bound = *ub_W;
        config.cd_max_iter = *cd_maxit;
        config.verbose = (*verbose != 0);
        config.seed = static_cast<uint32_t>(*seed);
        config.loss_every = *loss_every;
        config.patience = *patience;
        config.W.nonneg = (*nonneg_W != 0);
        config.H.nonneg = (*nonneg_H != 0);
        config.loss.type = static_cast<FactorNet::LossType>(*loss_type);
        config.loss.huber_delta = *huber_delta;
        config.irls_max_iter = *irls_max_iter;
        config.irls_tol = *irls_tol;
        config.norm_type = static_cast<FactorNet::NormType>(*norm_type);
        config.track_loss_history = false;

        // Initialize W (k×m) and H (k×n) from input arrays
        using DenseMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
        DenseMatrix W_init = Eigen::Map<DenseMatrix>(W, *k, *m).transpose();
        DenseMatrix H_init = Eigen::Map<DenseMatrix>(H, *k, *n);

        auto result = FactorNet::nmf::nmf_fit_gpu_zerocopy<double>(
            dev_col_ptr, dev_row_idx, dev_values,
            *m, *n, nnz, config, &W_init, &H_init);

        // Write results back
        for (int i = 0; i < *m; ++i)
            for (int f = 0; f < *k; ++f)
                W[f + static_cast<size_t>(i) * (*k)] = result.W(i, f);
        for (int j = 0; j < *n; ++j)
            for (int f = 0; f < *k; ++f)
                H[f + static_cast<size_t>(j) * (*k)] = result.H(f, j);
        for (int f = 0; f < *k; ++f)
            d[f] = result.d(f);

        *out_iter = result.iterations;
        *out_converged = result.converged ? 1 : 0;
        *out_loss = static_cast<double>(result.train_loss);
        *out_tol = static_cast<double>(result.final_tol);
        *out_status = 0;

    } catch (const std::exception& e) {
        FACTORNET_GPU_WARN("GPU zero-copy NMF error: %s\n", e.what());
        *out_status = -1;
    }
}

} // extern "C"
