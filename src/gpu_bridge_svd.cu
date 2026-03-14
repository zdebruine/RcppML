/**
 * @file gpu_bridge_svd.cu
 * @brief GPU SVD/PCA entry points (Lanczos, deflation; double/float/dense)
 *
 * Part of RcppML GPU bridge. Compiled by nvcc into RcppML_gpu.so.
 * Extracted from gpu_bridge.cu (2026).
 *
 * Build: make -f Makefile.gpu
 */

#include "gpu_bridge_common.cuh"

extern "C" {

// ============================================================================
// GPU SVD/PCA (deflation-based)
// ============================================================================

/**
 * @brief GPU truncated SVD/PCA with algorithm selection (fp64).
 *
 * All scalars are pointers per R .C() convention.
 *
 * @param col_ptr, row_idx, values  CSC sparse matrix (host)
 * @param m, n, nnz                 Matrix dimensions
 * @param k_max                     Maximum rank
 * @param U                         Output: m × k_max factor (column-major)
 * @param d                         Output: k_max singular values
 * @param V                         Output: n × k_max factor (column-major)
 * @param tol                       Convergence tolerance per factor
 * @param max_iter                  Max ALS iterations per factor
 * @param center                    PCA centering (0/1)
 * @param verbose                   Verbose output (0/1)
 * @param seed                      Random seed
 * @param threads                   (unused for GPU, reserved)
 * @param L1_u, L1_v               L1 regularization
 * @param L2_u, L2_v               L2 regularization
 * @param nonneg_u, nonneg_v       Non-negativity constraints (0/1)
 * @param ub_u, ub_v               Upper bound constraints
 * @param test_fraction             CV holdout fraction
 * @param cv_seed                   CV mask seed
 * @param patience                  Auto-rank patience
 * @param mask_zeros                Mask zeros in CV (0/1)
 * @param algorithm                 SVD algorithm: 0=deflation, 1=irlba, 2=lanczos, 3=randomized
 * @param out_k_selected            Output: selected rank
 * @param out_wall_time_ms          Output: wall time in ms
 * @param out_test_loss             Output: test loss trajectory (pre-allocated k_max)
 * @param out_iters_per_factor      Output: iterations per factor (pre-allocated k_max)
 * @param out_frobenius_norm_sq     Output: ||A||^2_F
 * @param out_row_means             Output: row means if centered (pre-allocated m)
 * @param out_status                Output: 0=success, -1=error
 */
void rcppml_gpu_svd_pca_double(
    const int* col_ptr, const int* row_idx, const double* values,
    int* m, int* n, int* nnz, int* k_max,
    double* U, double* d, double* V,
    double* tol, int* max_iter,
    int* center, int* verbose, int* seed, int* threads,
    double* L1_u, double* L1_v, double* L2_u, double* L2_v,
    int* nonneg_u, int* nonneg_v,
    double* ub_u, double* ub_v,
    double* L21_u, double* L21_v,
    double* angular_u, double* angular_v,
    double* test_fraction, int* cv_seed, int* patience, int* mask_zeros,
    int* algorithm,
    // Graph U regularization (CSC arrays; nnz=0 means not provided)
    const int* graph_u_p, const int* graph_u_i, const double* graph_u_x,
    int* graph_u_dim, int* graph_u_nnz, double* graph_u_lambda_val,
    // Graph V regularization (CSC arrays; nnz=0 means not provided)
    const int* graph_v_p, const int* graph_v_i, const double* graph_v_x,
    int* graph_v_dim, int* graph_v_nnz, double* graph_v_lambda_val,
    // Observation mask (CSC arrays; nnz=0 means not provided)
    const int* obs_mask_p, const int* obs_mask_i, const double* obs_mask_x,
    int* obs_mask_rows, int* obs_mask_cols, int* obs_mask_nnz,
    int* out_k_selected, double* out_wall_time_ms,
    double* out_test_loss, int* out_iters_per_factor,
    double* out_frobenius_norm_sq, double* out_row_means,
    double* robust_delta, int* irls_max_iter, double* irls_tol,
    int* out_status)
{
    try {
        FactorNet::SVDConfig<double> config;
        config.k_max = *k_max;
        config.tol = *tol;
        config.max_iter = *max_iter;
        config.center = (*center != 0);
        config.verbose = (*verbose != 0);
        config.seed = static_cast<uint32_t>(*seed);
        config.threads = *threads;
        config.L1_u = *L1_u;
        config.L1_v = *L1_v;
        config.L2_u = *L2_u;
        config.L2_v = *L2_v;
        config.nonneg_u = (*nonneg_u != 0);
        config.nonneg_v = (*nonneg_v != 0);
        config.upper_bound_u = *ub_u;
        config.upper_bound_v = *ub_v;
        config.L21_u = L21_u ? *L21_u : 0.0;
        config.L21_v = L21_v ? *L21_v : 0.0;
        config.angular_u = angular_u ? *angular_u : 0.0;
        config.angular_v = angular_v ? *angular_v : 0.0;
        config.robust_delta = robust_delta ? *robust_delta : 0.0;
        config.irls_max_iter = irls_max_iter ? *irls_max_iter : 5;
        config.irls_tol = irls_tol ? *irls_tol : 1e-4;
        config.test_fraction = *test_fraction;
        config.cv_seed = static_cast<uint32_t>(*cv_seed);
        config.patience = *patience;
        config.mask_zeros = (*mask_zeros != 0);

        // Graph Laplacians from raw CSC arrays
        Eigen::SparseMatrix<double> graph_u_mat, graph_v_mat;
        if (*graph_u_nnz > 0) {
            graph_u_mat = sparse_from_csc<double>(
                *graph_u_dim, *graph_u_dim, *graph_u_nnz,
                graph_u_p, graph_u_i, graph_u_x);
            config.graph_u = &graph_u_mat;
            config.graph_u_lambda = *graph_u_lambda_val;
        }
        if (*graph_v_nnz > 0) {
            graph_v_mat = sparse_from_csc<double>(
                *graph_v_dim, *graph_v_dim, *graph_v_nnz,
                graph_v_p, graph_v_i, graph_v_x);
            config.graph_v = &graph_v_mat;
            config.graph_v_lambda = *graph_v_lambda_val;
        }

        // Observation mask from raw CSC arrays
        Eigen::SparseMatrix<double> obs_mask_mat;
        if (*obs_mask_nnz > 0) {
            obs_mask_mat = sparse_from_csc<double>(
                *obs_mask_rows, *obs_mask_cols, *obs_mask_nnz,
                obs_mask_p, obs_mask_i, obs_mask_x);
            config.obs_mask = &obs_mask_mat;
        }

        int algo = algorithm ? *algorithm : 0;

        FactorNet::SVDResult<double> result;

        switch (algo) {
            case 1:  // irlba
                result = FactorNet::svd::irlba_svd_gpu<double>(
                    col_ptr, row_idx, values,
                    *m, *n, *nnz, config);
                break;
            case 2:  // lanczos
                result = FactorNet::svd::lanczos_svd_gpu<double>(
                    col_ptr, row_idx, values,
                    *m, *n, *nnz, config);
                break;
            case 3:  // randomized
                result = FactorNet::svd::randomized_svd_gpu<double>(
                    col_ptr, row_idx, values,
                    *m, *n, *nnz, config);
                break;
            case 4:  // krylov
                result = FactorNet::svd::krylov_svd_gpu<double>(
                    col_ptr, row_idx, values,
                    *m, *n, *nnz, config);
                break;
            default:  // 0 = deflation (default, supports regularization + CV)
                result = FactorNet::svd::deflation_svd_gpu<double>(
                    col_ptr, row_idx, values,
                    *m, *n, *nnz, config);
                break;
        }

        int ksel = result.k_selected;
        *out_k_selected = ksel;
        *out_wall_time_ms = result.wall_time_ms;
        *out_frobenius_norm_sq = result.frobenius_norm_sq;

        // Write U (m × ksel, column-major)
        for (int j = 0; j < ksel; ++j)
            for (int i = 0; i < *m; ++i)
                U[i + static_cast<size_t>(j) * (*m)] = result.U(i, j);

        // Write d (ksel)
        for (int j = 0; j < ksel; ++j)
            d[j] = result.d(j);

        // Write V (n × ksel, column-major)
        for (int j = 0; j < ksel; ++j)
            for (int i = 0; i < *n; ++i)
                V[i + static_cast<size_t>(j) * (*n)] = result.V(i, j);

        // Write test loss trajectory
        for (int j = 0; j < static_cast<int>(result.test_loss_trajectory.size()); ++j)
            out_test_loss[j] = result.test_loss_trajectory[j];

        // Write iters per factor
        for (int j = 0; j < static_cast<int>(result.iters_per_factor.size()); ++j)
            out_iters_per_factor[j] = result.iters_per_factor[j];

        // Write row means if centered
        if (result.centered && result.row_means.size() > 0) {
            for (int i = 0; i < *m; ++i)
                out_row_means[i] = result.row_means(i);
        }

        *out_status = 0;

    } catch (const std::exception& e) {
        FACTORNET_GPU_WARN("GPU SVD/PCA error: %s\n", e.what());
        *out_status = -1;
    }
}

/**
 * @brief Float-precision GPU SVD/PCA entry point.
 *
 * R only has double-precision numerics, so all buffers come in as double*.
 * This function converts sparse matrix values to float, runs the SVD in
 * single precision, then converts results back to double for R.
 *
 * On GPUs with FP32-optimized throughput (V100S, RTX 8000), this can give
 * ~2× bandwidth improvement over FP64. For scRNA-seq PCA, float precision
 * is more than sufficient (data itself is integer counts).
 */
void rcppml_gpu_svd_pca_float(
    const int* col_ptr, const int* row_idx, const double* values,
    int* m, int* n, int* nnz, int* k_max,
    double* U, double* d, double* V,
    double* tol, int* max_iter,
    int* center, int* verbose, int* seed, int* threads,
    double* L1_u, double* L1_v, double* L2_u, double* L2_v,
    int* nonneg_u, int* nonneg_v,
    double* ub_u, double* ub_v,
    double* L21_u, double* L21_v,
    double* angular_u, double* angular_v,
    double* test_fraction, int* cv_seed, int* patience, int* mask_zeros,
    int* algorithm,
    // Graph U regularization (CSC arrays; nnz=0 means not provided)
    const int* graph_u_p, const int* graph_u_i, const double* graph_u_x,
    int* graph_u_dim, int* graph_u_nnz, double* graph_u_lambda_val,
    // Graph V regularization (CSC arrays; nnz=0 means not provided)
    const int* graph_v_p, const int* graph_v_i, const double* graph_v_x,
    int* graph_v_dim, int* graph_v_nnz, double* graph_v_lambda_val,
    // Observation mask (CSC arrays; nnz=0 means not provided)
    const int* obs_mask_p, const int* obs_mask_i, const double* obs_mask_x,
    int* obs_mask_rows, int* obs_mask_cols, int* obs_mask_nnz,
    int* out_k_selected, double* out_wall_time_ms,
    double* out_test_loss, int* out_iters_per_factor,
    double* out_frobenius_norm_sq, double* out_row_means,
    double* robust_delta, int* irls_max_iter, double* irls_tol,
    int* out_status)
{
    try {
        // Convert sparse values from double to float
        int nz = *nnz;
        std::vector<float> values_f(nz);
        for (int i = 0; i < nz; ++i)
            values_f[i] = static_cast<float>(values[i]);

        FactorNet::SVDConfig<float> config;
        config.k_max = *k_max;
        config.tol = static_cast<float>(*tol);
        config.max_iter = *max_iter;
        config.center = (*center != 0);
        config.verbose = (*verbose != 0);
        config.seed = static_cast<uint32_t>(*seed);
        config.threads = *threads;
        config.L1_u = static_cast<float>(*L1_u);
        config.L1_v = static_cast<float>(*L1_v);
        config.L2_u = static_cast<float>(*L2_u);
        config.L2_v = static_cast<float>(*L2_v);
        config.nonneg_u = (*nonneg_u != 0);
        config.nonneg_v = (*nonneg_v != 0);
        config.upper_bound_u = static_cast<float>(*ub_u);
        config.upper_bound_v = static_cast<float>(*ub_v);
        config.L21_u = L21_u ? static_cast<float>(*L21_u) : 0.0f;
        config.L21_v = L21_v ? static_cast<float>(*L21_v) : 0.0f;
        config.angular_u = angular_u ? static_cast<float>(*angular_u) : 0.0f;
        config.angular_v = angular_v ? static_cast<float>(*angular_v) : 0.0f;
        config.robust_delta = robust_delta ? static_cast<float>(*robust_delta) : 0.0f;
        config.irls_max_iter = irls_max_iter ? *irls_max_iter : 5;
        config.irls_tol = irls_tol ? static_cast<float>(*irls_tol) : 1e-4f;
        config.test_fraction = static_cast<float>(*test_fraction);
        config.cv_seed = static_cast<uint32_t>(*cv_seed);
        config.patience = *patience;
        config.mask_zeros = (*mask_zeros != 0);

        // Graph Laplacians from raw CSC arrays (cast to float)
        Eigen::SparseMatrix<float> graph_u_mat, graph_v_mat;
        if (*graph_u_nnz > 0) {
            graph_u_mat = sparse_from_csc<float>(
                *graph_u_dim, *graph_u_dim, *graph_u_nnz,
                graph_u_p, graph_u_i, graph_u_x);
            config.graph_u = &graph_u_mat;
            config.graph_u_lambda = static_cast<float>(*graph_u_lambda_val);
        }
        if (*graph_v_nnz > 0) {
            graph_v_mat = sparse_from_csc<float>(
                *graph_v_dim, *graph_v_dim, *graph_v_nnz,
                graph_v_p, graph_v_i, graph_v_x);
            config.graph_v = &graph_v_mat;
            config.graph_v_lambda = static_cast<float>(*graph_v_lambda_val);
        }

        // Observation mask from raw CSC arrays (cast to float)
        Eigen::SparseMatrix<float> obs_mask_mat;
        if (*obs_mask_nnz > 0) {
            obs_mask_mat = sparse_from_csc<float>(
                *obs_mask_rows, *obs_mask_cols, *obs_mask_nnz,
                obs_mask_p, obs_mask_i, obs_mask_x);
            config.obs_mask = &obs_mask_mat;
        }

        int algo = algorithm ? *algorithm : 0;

        FactorNet::SVDResult<float> result;

        switch (algo) {
            case 1:  // irlba
                result = FactorNet::svd::irlba_svd_gpu<float>(
                    col_ptr, row_idx, values_f.data(),
                    *m, *n, *nnz, config);
                break;
            case 2:  // lanczos
                result = FactorNet::svd::lanczos_svd_gpu<float>(
                    col_ptr, row_idx, values_f.data(),
                    *m, *n, *nnz, config);
                break;
            case 3:  // randomized
                result = FactorNet::svd::randomized_svd_gpu<float>(
                    col_ptr, row_idx, values_f.data(),
                    *m, *n, *nnz, config);
                break;
            case 4:  // krylov
                result = FactorNet::svd::krylov_svd_gpu<float>(
                    col_ptr, row_idx, values_f.data(),
                    *m, *n, *nnz, config);
                break;
            default:  // 0 = deflation
                result = FactorNet::svd::deflation_svd_gpu<float>(
                    col_ptr, row_idx, values_f.data(),
                    *m, *n, *nnz, config);
                break;
        }

        int ksel = result.k_selected;
        *out_k_selected = ksel;
        *out_wall_time_ms = static_cast<double>(result.wall_time_ms);
        *out_frobenius_norm_sq = static_cast<double>(result.frobenius_norm_sq);

        // Convert U (m × ksel) from float → double
        for (int j = 0; j < ksel; ++j)
            for (int i = 0; i < *m; ++i)
                U[i + static_cast<size_t>(j) * (*m)] = static_cast<double>(result.U(i, j));

        // Convert d (ksel) from float → double
        for (int j = 0; j < ksel; ++j)
            d[j] = static_cast<double>(result.d(j));

        // Convert V (n × ksel) from float → double
        for (int j = 0; j < ksel; ++j)
            for (int i = 0; i < *n; ++i)
                V[i + static_cast<size_t>(j) * (*n)] = static_cast<double>(result.V(i, j));

        // Write test loss trajectory (float → double)
        for (int j = 0; j < static_cast<int>(result.test_loss_trajectory.size()); ++j)
            out_test_loss[j] = static_cast<double>(result.test_loss_trajectory[j]);

        // Write iters per factor
        for (int j = 0; j < static_cast<int>(result.iters_per_factor.size()); ++j)
            out_iters_per_factor[j] = result.iters_per_factor[j];

        // Write row means if centered (float → double)
        if (result.centered && result.row_means.size() > 0) {
            for (int i = 0; i < *m; ++i)
                out_row_means[i] = static_cast<double>(result.row_means(i));
        }

        *out_status = 0;

    } catch (const std::exception& e) {
        FACTORNET_GPU_WARN("GPU SVD/PCA (float) error: %s\n", e.what());
        *out_status = -1;
    }
}

/**
 * @brief Dense-matrix GPU SVD/PCA (double precision).
 *
 * Accepts a column-major dense matrix A (m×n) instead of CSC format.
 * For deflation: uses cuBLAS GEMV instead of cuSPARSE SpMV.
 * For all other algorithms: converts to CSC and uses sparse path.
 *
 * Called from R via .gpu_svd_pca_dense() in gpu_backend.R when the input
 * @p A is a base R matrix (is.matrix(A) == TRUE).
 */
void rcppml_gpu_svd_pca_dense_double(
    const double* A_data, int* m, int* n, int* k_max,
    double* U, double* d, double* V,
    double* tol, int* max_iter,
    int* center, int* verbose, int* seed, int* threads,
    double* L1_u, double* L1_v, double* L2_u, double* L2_v,
    int* nonneg_u, int* nonneg_v,
    double* ub_u, double* ub_v,
    double* L21_u, double* L21_v,
    double* angular_u, double* angular_v,
    double* test_fraction, int* cv_seed, int* patience, int* mask_zeros,
    int* algorithm,
    // Graph U regularization (CSC arrays; nnz=0 means not provided)
    const int* graph_u_p, const int* graph_u_i, const double* graph_u_x,
    int* graph_u_dim, int* graph_u_nnz, double* graph_u_lambda_val,
    // Graph V regularization (CSC arrays; nnz=0 means not provided)
    const int* graph_v_p, const int* graph_v_i, const double* graph_v_x,
    int* graph_v_dim, int* graph_v_nnz, double* graph_v_lambda_val,
    // Observation mask (CSC arrays; nnz=0 means not provided)
    const int* obs_mask_p, const int* obs_mask_i, const double* obs_mask_x,
    int* obs_mask_rows, int* obs_mask_cols, int* obs_mask_nnz,
    int* out_k_selected, double* out_wall_time_ms,
    double* out_test_loss, int* out_iters_per_factor,
    double* out_frobenius_norm_sq, double* out_row_means,
    double* robust_delta, int* irls_max_iter, double* irls_tol,
    int* out_status)
{
    try {
        FactorNet::SVDConfig<double> config;
        config.k_max    = *k_max;
        config.tol      = *tol;
        config.max_iter = *max_iter;
        config.center   = (*center != 0);
        config.verbose  = (*verbose != 0);
        config.seed     = static_cast<uint32_t>(*seed);
        config.threads  = *threads;
        config.L1_u     = *L1_u;   config.L1_v     = *L1_v;
        config.L2_u     = *L2_u;   config.L2_v     = *L2_v;
        config.nonneg_u = (*nonneg_u != 0);
        config.nonneg_v = (*nonneg_v != 0);
        config.upper_bound_u = *ub_u;
        config.upper_bound_v = *ub_v;
        config.L21_u    = L21_u ? *L21_u : 0.0;
        config.L21_v    = L21_v ? *L21_v : 0.0;
        config.angular_u = angular_u ? *angular_u : 0.0;
        config.angular_v = angular_v ? *angular_v : 0.0;
        config.robust_delta = robust_delta ? *robust_delta : 0.0;
        config.irls_max_iter = irls_max_iter ? *irls_max_iter : 5;
        config.irls_tol = irls_tol ? *irls_tol : 1e-4;
        config.test_fraction = *test_fraction;
        config.cv_seed  = static_cast<uint32_t>(*cv_seed);
        config.patience = *patience;
        config.mask_zeros = (*mask_zeros != 0);

        // Graph Laplacians from raw CSC arrays
        Eigen::SparseMatrix<double> graph_u_mat, graph_v_mat;
        if (*graph_u_nnz > 0) {
            graph_u_mat = sparse_from_csc<double>(
                *graph_u_dim, *graph_u_dim, *graph_u_nnz,
                graph_u_p, graph_u_i, graph_u_x);
            config.graph_u = &graph_u_mat;
            config.graph_u_lambda = *graph_u_lambda_val;
        }
        if (*graph_v_nnz > 0) {
            graph_v_mat = sparse_from_csc<double>(
                *graph_v_dim, *graph_v_dim, *graph_v_nnz,
                graph_v_p, graph_v_i, graph_v_x);
            config.graph_v = &graph_v_mat;
            config.graph_v_lambda = *graph_v_lambda_val;
        }

        // Observation mask from raw CSC arrays
        Eigen::SparseMatrix<double> obs_mask_mat;
        if (*obs_mask_nnz > 0) {
            obs_mask_mat = sparse_from_csc<double>(
                *obs_mask_rows, *obs_mask_cols, *obs_mask_nnz,
                obs_mask_p, obs_mask_i, obs_mask_x);
            config.obs_mask = &obs_mask_mat;
        }

        int algo = algorithm ? *algorithm : 0;

        FactorNet::SVDResult<double> result;

        if (algo == 0) {
            // Deflation: native cuBLAS GEMV dense path
            result = FactorNet::svd::deflation_svd_gpu_dense<double>(
                A_data, *m, *n, config);
        } else if (algo == 2) {
            // Lanczos: native cuBLAS GEMV dense path
            result = FactorNet::svd::lanczos_svd_gpu_dense<double>(
                A_data, *m, *n, config);
        } else if (algo == 3) {
            // Randomized: native cuBLAS GEMM dense path
            result = FactorNet::svd::randomized_svd_gpu_dense<double>(
                A_data, *m, *n, config);
        } else {
            // Other algorithms: fast dense→CSC (no zero-scan, direct enumeration)
            int mm = *m, nn = *n;
            size_t mn = static_cast<size_t>(mm) * nn;
            std::vector<int> col_ptr(nn + 1);
            std::vector<int> row_idx(mn);
            for (int j = 0; j <= nn; ++j) col_ptr[j] = j * mm;
            for (int j = 0; j < nn; ++j)
                for (int i = 0; i < mm; ++i)
                    row_idx[static_cast<size_t>(j) * mm + i] = i;
            // Values = dense matrix in column-major order (no copy needed)
            int nnz_actual = static_cast<int>(mn);
            switch (algo) {
                case 1:
                    result = FactorNet::svd::irlba_svd_gpu<double>(
                        col_ptr.data(), row_idx.data(), A_data, mm, nn, nnz_actual, config);
                    break;
                case 3:
                    result = FactorNet::svd::randomized_svd_gpu<double>(
                        col_ptr.data(), row_idx.data(), A_data, mm, nn, nnz_actual, config);
                    break;
                case 4:
                    result = FactorNet::svd::krylov_svd_gpu<double>(
                        col_ptr.data(), row_idx.data(), A_data, mm, nn, nnz_actual, config);
                    break;
                default:
                    result = FactorNet::svd::deflation_svd_gpu_dense<double>(
                        A_data, *m, *n, config);
                    break;
            }
        }

        int ksel = result.k_selected;
        *out_k_selected   = ksel;
        *out_wall_time_ms = result.wall_time_ms;
        *out_frobenius_norm_sq = result.frobenius_norm_sq;

        for (int j = 0; j < ksel; ++j)
            for (int i = 0; i < *m; ++i)
                U[i + static_cast<size_t>(j) * (*m)] = result.U(i, j);
        for (int j = 0; j < ksel; ++j)
            d[j] = result.d(j);
        for (int j = 0; j < ksel; ++j)
            for (int i = 0; i < *n; ++i)
                V[i + static_cast<size_t>(j) * (*n)] = result.V(i, j);
        for (int j = 0; j < (int)result.test_loss_trajectory.size(); ++j)
            out_test_loss[j] = result.test_loss_trajectory[j];
        for (int j = 0; j < (int)result.iters_per_factor.size(); ++j)
            out_iters_per_factor[j] = result.iters_per_factor[j];
        if (result.centered && result.row_means.size() > 0)
            for (int i = 0; i < *m; ++i)
                out_row_means[i] = result.row_means(i);

        *out_status = 0;
    } catch (const std::exception& e) {
        FACTORNET_GPU_WARN("GPU SVD/PCA (dense) error: %s\n", e.what());
        *out_status = -1;
    }
}

/**
 * @brief Dense-matrix GPU SVD/PCA (float precision).
 *
 * Same as rcppml_gpu_svd_pca_dense_double but uses float precision.
 * Input/output arrays are still double (R's native type) — conversion
 * happens at the boundary.
 */
void rcppml_gpu_svd_pca_dense_float(
    const double* A_data, int* m, int* n, int* k_max,
    double* U, double* d, double* V,
    double* tol, int* max_iter,
    int* center, int* verbose, int* seed, int* threads,
    double* L1_u, double* L1_v, double* L2_u, double* L2_v,
    int* nonneg_u, int* nonneg_v,
    double* ub_u, double* ub_v,
    double* L21_u, double* L21_v,
    double* angular_u, double* angular_v,
    double* test_fraction, int* cv_seed, int* patience, int* mask_zeros,
    int* algorithm,
    // Graph U regularization (CSC arrays; nnz=0 means not provided)
    const int* graph_u_p, const int* graph_u_i, const double* graph_u_x,
    int* graph_u_dim, int* graph_u_nnz, double* graph_u_lambda_val,
    // Graph V regularization (CSC arrays; nnz=0 means not provided)
    const int* graph_v_p, const int* graph_v_i, const double* graph_v_x,
    int* graph_v_dim, int* graph_v_nnz, double* graph_v_lambda_val,
    // Observation mask (CSC arrays; nnz=0 means not provided)
    const int* obs_mask_p, const int* obs_mask_i, const double* obs_mask_x,
    int* obs_mask_rows, int* obs_mask_cols, int* obs_mask_nnz,
    int* out_k_selected, double* out_wall_time_ms,
    double* out_test_loss, int* out_iters_per_factor,
    double* out_frobenius_norm_sq, double* out_row_means,
    double* robust_delta, int* irls_max_iter, double* irls_tol,
    int* out_status)
{
    try {
        // Convert dense matrix from double to float
        const size_t mn = static_cast<size_t>(*m) * (*n);
        std::vector<float> A_f(mn);
        for (size_t i = 0; i < mn; ++i)
            A_f[i] = static_cast<float>(A_data[i]);

        FactorNet::SVDConfig<float> config;
        config.k_max    = *k_max;
        config.tol      = static_cast<float>(*tol);
        config.max_iter = *max_iter;
        config.center   = (*center != 0);
        config.verbose  = (*verbose != 0);
        config.seed     = static_cast<uint32_t>(*seed);
        config.threads  = *threads;
        config.L1_u     = static_cast<float>(*L1_u);
        config.L1_v     = static_cast<float>(*L1_v);
        config.L2_u     = static_cast<float>(*L2_u);
        config.L2_v     = static_cast<float>(*L2_v);
        config.nonneg_u = (*nonneg_u != 0);
        config.nonneg_v = (*nonneg_v != 0);
        config.upper_bound_u = static_cast<float>(*ub_u);
        config.upper_bound_v = static_cast<float>(*ub_v);
        config.L21_u    = L21_u ? static_cast<float>(*L21_u) : 0.0f;
        config.L21_v    = L21_v ? static_cast<float>(*L21_v) : 0.0f;
        config.angular_u = angular_u ? static_cast<float>(*angular_u) : 0.0f;
        config.angular_v = angular_v ? static_cast<float>(*angular_v) : 0.0f;
        config.robust_delta = robust_delta ? static_cast<float>(*robust_delta) : 0.0f;
        config.irls_max_iter = irls_max_iter ? *irls_max_iter : 5;
        config.irls_tol = irls_tol ? static_cast<float>(*irls_tol) : 1e-4f;
        config.test_fraction = static_cast<float>(*test_fraction);
        config.cv_seed  = static_cast<uint32_t>(*cv_seed);
        config.patience = *patience;
        config.mask_zeros = (*mask_zeros != 0);

        // Graph Laplacians from raw CSC arrays (cast to float)
        Eigen::SparseMatrix<float> graph_u_mat, graph_v_mat;
        if (*graph_u_nnz > 0) {
            graph_u_mat = sparse_from_csc<float>(
                *graph_u_dim, *graph_u_dim, *graph_u_nnz,
                graph_u_p, graph_u_i, graph_u_x);
            config.graph_u = &graph_u_mat;
            config.graph_u_lambda = static_cast<float>(*graph_u_lambda_val);
        }
        if (*graph_v_nnz > 0) {
            graph_v_mat = sparse_from_csc<float>(
                *graph_v_dim, *graph_v_dim, *graph_v_nnz,
                graph_v_p, graph_v_i, graph_v_x);
            config.graph_v = &graph_v_mat;
            config.graph_v_lambda = static_cast<float>(*graph_v_lambda_val);
        }

        // Observation mask from raw CSC arrays (cast to float)
        Eigen::SparseMatrix<float> obs_mask_mat;
        if (*obs_mask_nnz > 0) {
            obs_mask_mat = sparse_from_csc<float>(
                *obs_mask_rows, *obs_mask_cols, *obs_mask_nnz,
                obs_mask_p, obs_mask_i, obs_mask_x);
            config.obs_mask = &obs_mask_mat;
        }

        int algo = algorithm ? *algorithm : 0;

        FactorNet::SVDResult<float> result;

        if (algo == 0) {
            // Deflation: native cuBLAS GEMV dense path
            result = FactorNet::svd::deflation_svd_gpu_dense<float>(
                A_f.data(), *m, *n, config);
        } else if (algo == 2) {
            // Lanczos: native cuBLAS GEMV dense path
            result = FactorNet::svd::lanczos_svd_gpu_dense<float>(
                A_f.data(), *m, *n, config);
        } else if (algo == 3) {
            // Randomized: native cuBLAS GEMM dense path
            result = FactorNet::svd::randomized_svd_gpu_dense<float>(
                A_f.data(), *m, *n, config);
        } else {
            // Other algorithms: fast dense→CSC (no zero-scan, direct enumeration)
            int mm = *m, nn = *n;
            size_t mn = static_cast<size_t>(mm) * nn;
            std::vector<int> col_ptr(nn + 1);
            std::vector<int> row_idx(mn);
            for (int j = 0; j <= nn; ++j) col_ptr[j] = j * mm;
            for (int j = 0; j < nn; ++j)
                for (int i = 0; i < mm; ++i)
                    row_idx[static_cast<size_t>(j) * mm + i] = i;
            // Values = float vector (already converted from double above)
            int nnz_actual = static_cast<int>(mn);
            switch (algo) {
                case 1:
                    result = FactorNet::svd::irlba_svd_gpu<float>(
                        col_ptr.data(), row_idx.data(), A_f.data(), mm, nn, nnz_actual, config);
                    break;
                case 3:
                    result = FactorNet::svd::randomized_svd_gpu<float>(
                        col_ptr.data(), row_idx.data(), A_f.data(), mm, nn, nnz_actual, config);
                    break;
                case 4:
                    result = FactorNet::svd::krylov_svd_gpu<float>(
                        col_ptr.data(), row_idx.data(), A_f.data(), mm, nn, nnz_actual, config);
                    break;
                default:
                    result = FactorNet::svd::deflation_svd_gpu_dense<float>(
                        A_f.data(), *m, *n, config);
                    break;
            }
        }

        int ksel = result.k_selected;
        *out_k_selected   = ksel;
        *out_wall_time_ms = static_cast<double>(result.wall_time_ms);
        *out_frobenius_norm_sq = static_cast<double>(result.frobenius_norm_sq);

        for (int j = 0; j < ksel; ++j)
            for (int i = 0; i < *m; ++i)
                U[i + static_cast<size_t>(j) * (*m)] = static_cast<double>(result.U(i, j));
        for (int j = 0; j < ksel; ++j)
            d[j] = static_cast<double>(result.d(j));
        for (int j = 0; j < ksel; ++j)
            for (int i = 0; i < *n; ++i)
                V[i + static_cast<size_t>(j) * (*n)] = static_cast<double>(result.V(i, j));
        for (int j = 0; j < (int)result.test_loss_trajectory.size(); ++j)
            out_test_loss[j] = static_cast<double>(result.test_loss_trajectory[j]);
        for (int j = 0; j < (int)result.iters_per_factor.size(); ++j)
            out_iters_per_factor[j] = result.iters_per_factor[j];
        if (result.centered && result.row_means.size() > 0)
            for (int i = 0; i < *m; ++i)
                out_row_means[i] = static_cast<double>(result.row_means(i));

        *out_status = 0;
    } catch (const std::exception& e) {
        FACTORNET_GPU_WARN("GPU SVD/PCA (dense, fp32) error: %s\n", e.what());
        *out_status = -1;
    }
}

} // extern "C"
