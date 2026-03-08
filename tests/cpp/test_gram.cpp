// test_gram.cpp — Gram matrix computation tests

#include "test_framework.hpp"
#include <FactorNet/primitives/primitives.hpp>
#include <FactorNet/primitives/cpu/gram.hpp>

using namespace FactorNet;
using namespace FactorNet::primitives;

TEST_CASE("gram_identity") {
    // H = I → G = I + tiny_num*I
    const int k = 3;
    DenseMatrix<double> H = DenseMatrix<double>::Identity(k, k);
    DenseMatrix<double> G(k, k);
    gram<CPU, double>(H, G);

    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            if (i == j) {
                CHECK_NEAR(G(i, j), 1.0, 1e-6);  // 1 + tiny_num ≈ 1
            } else {
                CHECK_NEAR(G(i, j), 0.0, 1e-10);
            }
        }
    }
}

TEST_CASE("gram_symmetric") {
    const int k = 4, n = 20;
    DenseMatrix<double> H = DenseMatrix<double>::Random(k, n);
    DenseMatrix<double> G(k, k);
    gram<CPU, double>(H, G);

    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            CHECK_NEAR(G(i, j), G(j, i), 1e-12);
        }
    }
}

TEST_CASE("gram_positive_diagonal") {
    const int k = 4, n = 20;
    DenseMatrix<double> H = DenseMatrix<double>::Random(k, n);
    DenseMatrix<double> G(k, k);
    gram<CPU, double>(H, G);

    for (int i = 0; i < k; ++i) {
        CHECK_GT(G(i, i), 0.0);
    }
}

TEST_CASE("gram_matches_manual") {
    const int k = 3, n = 10;
    DenseMatrix<double> H = DenseMatrix<double>::Random(k, n);
    DenseMatrix<double> G(k, k);
    gram<CPU, double>(H, G);

    // Manual computation: G_expected = H * H^T
    DenseMatrix<double> G_expected = H * H.transpose();

    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            CHECK_NEAR(G(i, j), G_expected(i, j), 1e-8);
        }
    }
}

TEST_CASE("gram_scaled_input") {
    // G(alpha*H) = alpha^2 * G(H) (approximately, ignoring tiny_num)
    const int k = 3, n = 15;
    DenseMatrix<double> H = DenseMatrix<double>::Random(k, n);
    DenseMatrix<double> G1(k, k), G2(k, k);

    gram<CPU, double>(H, G1);
    DenseMatrix<double> H2 = 2.0 * H;
    gram<CPU, double>(H2, G2);

    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            CHECK_NEAR(G2(i, j), 4.0 * G1(i, j), 1e-6);
        }
    }
}
