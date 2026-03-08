// test_nmf.cpp — NMFResult and NMFConfig tests (R-independent)
//
// Note: nmf_fit() depends on R (interrupt checking, logging).
// Full NMF fit tests run through R's testthat framework instead.

#include "test_framework.hpp"
#include <FactorNet/core/types.hpp>
#include <FactorNet/core/config.hpp>
#include <FactorNet/core/result.hpp>
#include <FactorNet/rng/rng.hpp>

using namespace FactorNet;

TEST_CASE("nmf_result_reconstruct") {
    NMFResult<double> result;
    result.W = DenseMatrix<double>::Ones(10, 3);
    result.d.resize(3);
    result.d << 2.0, 3.0, 1.0;
    result.H = DenseMatrix<double>::Ones(3, 20);

    DenseMatrix<double> recon = result.reconstruct();
    CHECK_EQ(recon.rows(), 10);
    CHECK_EQ(recon.cols(), 20);
    // Each entry = sum_k(W_ik * d_k * H_kj) = 1*2*1 + 1*3*1 + 1*1*1 = 6
    CHECK_NEAR(recon(0, 0), 6.0, 1e-10);
    CHECK_NEAR(recon(9, 19), 6.0, 1e-10);
}

TEST_CASE("nmf_result_normalize_preserves_reconstruction") {
    NMFResult<double> result;
    rng::SplitMix64 rng(42);
    result.W.resize(10, 3);
    rng.fill_uniform(result.W);
    result.d = DenseVector<double>::Ones(3);
    result.H.resize(3, 20);
    rng.fill_uniform(result.H);

    DenseMatrix<double> recon_before = result.reconstruct();
    result.normalize();
    DenseMatrix<double> recon_after = result.reconstruct();

    double err = (recon_before - recon_after).squaredNorm();
    CHECK_LT(err, 1e-10);
}

TEST_CASE("nmf_result_sort_orders_by_d") {
    NMFResult<double> result;
    rng::SplitMix64 rng(42);
    result.W.resize(10, 3);
    rng.fill_uniform(result.W);
    result.d.resize(3);
    result.d << 1.0, 5.0, 3.0;  // Unsorted
    result.H.resize(3, 20);
    rng.fill_uniform(result.H);

    DenseMatrix<double> recon_before = result.reconstruct();
    result.sort();
    DenseMatrix<double> recon_after = result.reconstruct();

    // d should be sorted decreasing
    for (int i = 0; i < result.d.size() - 1; ++i) {
        CHECK_GE(result.d(i), result.d(i + 1));
    }

    // Reconstruction preserved
    double err = (recon_before - recon_after).squaredNorm();
    CHECK_LT(err, 1e-10);
}
