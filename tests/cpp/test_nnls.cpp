// test_nnls.cpp — NNLS batch solver correctness tests

#include "test_framework.hpp"
#include <FactorNet/primitives/primitives.hpp>
#include <FactorNet/primitives/cpu/gram.hpp>
#include <FactorNet/primitives/cpu/nnls_batch.hpp>

using namespace FactorNet;
using namespace FactorNet::primitives;

TEST_CASE("nnls_identity_gram") {
    // G = I, B = positive vector → X should equal B
    const int k = 3, n = 5;
    DenseMatrix<double> G = DenseMatrix<double>::Identity(k, k);
    G.diagonal().array() += 1e-10;  // Match gram's tiny_num addition

    DenseMatrix<double> B = DenseMatrix<double>::Random(k, n).cwiseAbs();
    DenseMatrix<double> B_copy = B;
    DenseMatrix<double> X(k, n);
    X.setZero();

    nnls_batch<CPU, double>(G, B, X, 100, 1e-8, 0, 0, true, 1);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < k; ++i) {
            CHECK_NEAR(X(i, j), B_copy(i, j), 1e-4);
        }
    }
}

TEST_CASE("nnls_nonnegativity") {
    // Solution must be non-negative
    const int k = 4, n = 10;
    DenseMatrix<double> H = DenseMatrix<double>::Random(k, 20);
    DenseMatrix<double> G(k, k);
    gram<CPU, double>(H, G);

    DenseMatrix<double> B = DenseMatrix<double>::Random(k, n);  // Some negative RHS
    DenseMatrix<double> X(k, n);
    X.setZero();

    nnls_batch<CPU, double>(G, B, X, 100, 1e-8, 0, 0, true, 1);

    CHECK_GE(X.minCoeff(), 0.0);
}

TEST_CASE("nnls_unconstrained") {
    // With nonneg=false, solution can be negative
    const int k = 3, n = 5;
    DenseMatrix<double> G = DenseMatrix<double>::Identity(k, k);
    G.diagonal().array() += 1e-10;

    DenseMatrix<double> B = DenseMatrix<double>::Random(k, n);
    B(0, 0) = -1.0;  // Force negative RHS
    DenseMatrix<double> B_copy = B;
    DenseMatrix<double> X(k, n);
    X.setZero();

    nnls_batch<CPU, double>(G, B, X, 100, 1e-8, 0, 0, false, 1);

    // Without NN constraint, should recover approximately B for identity G
    CHECK_NEAR(X(0, 0), B_copy(0, 0), 1e-4);
}

TEST_CASE("nnls_known_solution") {
    // Set up a problem with a known positive solution
    //   G = [2 1; 1 2], b = [3; 3]
    //   True solution: x = [1; 1]
    const int k = 2, n = 1;
    DenseMatrix<double> G(k, k);
    G << 2.0, 1.0,
         1.0, 2.0;
    G.diagonal().array() += 1e-10;

    DenseMatrix<double> B(k, n);
    B << 3.0, 3.0;
    DenseMatrix<double> X(k, n);
    X.setZero();

    nnls_batch<CPU, double>(G, B, X, 100, 1e-8, 0, 0, true, 1);

    CHECK_NEAR(X(0, 0), 1.0, 1e-4);
    CHECK_NEAR(X(1, 0), 1.0, 1e-4);
}

TEST_CASE("nnls_l1_sparsity") {
    // L1 penalty should push some coefficients to zero
    const int k = 4, n = 10;
    DenseMatrix<double> H = DenseMatrix<double>::Random(k, 50).cwiseAbs();
    DenseMatrix<double> G(k, k);
    gram<CPU, double>(H, G);

    DenseMatrix<double> B = DenseMatrix<double>::Random(k, n).cwiseAbs();
    DenseMatrix<double> X_no_l1(k, n), X_l1(k, n);
    X_no_l1.setZero();
    X_l1.setZero();

    DenseMatrix<double> B1 = B, B2 = B;

    nnls_batch<CPU, double>(G, B1, X_no_l1, 100, 1e-8, 0, 0, true, 1);
    nnls_batch<CPU, double>(G, B2, X_l1, 100, 1e-8, 1.0, 0, true, 1);

    // L1 solution should have smaller or equal L1 norm
    double norm_no_l1 = X_no_l1.array().abs().sum();
    double norm_l1 = X_l1.array().abs().sum();
    CHECK_GE(norm_no_l1, norm_l1);
}

TEST_CASE("nnls_warm_start") {
    const int k = 3, n = 5;
    DenseMatrix<double> H = DenseMatrix<double>::Random(k, 20).cwiseAbs();
    DenseMatrix<double> G(k, k);
    gram<CPU, double>(H, G);

    DenseMatrix<double> B = DenseMatrix<double>::Random(k, n).cwiseAbs();
    DenseMatrix<double> B1 = B, B2 = B;
    DenseMatrix<double> X_cold(k, n), X_warm(k, n);
    X_cold.setZero();

    // Cold start
    nnls_batch<CPU, double>(G, B1, X_cold, 100, 1e-8, 0, 0, true, 1);

    // Warm start from cold solution
    X_warm = X_cold;
    nnls_batch<CPU, double>(G, B2, X_warm, 100, 1e-8, 0, 0, true, 1, 0, true);

    // Should get same result
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < k; ++i) {
            CHECK_NEAR(X_cold(i, j), X_warm(i, j), 1e-4);
        }
    }
}
