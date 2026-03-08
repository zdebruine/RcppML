// test_rng.cpp — RNG determinism and correctness tests

#include "test_framework.hpp"
#include <FactorNet/core/types.hpp>
#include <FactorNet/rng/rng.hpp>

using namespace FactorNet;

TEST_CASE("rng_deterministic_same_seed") {
    rng::SplitMix64 r1(42);
    rng::SplitMix64 r2(42);
    for (int i = 0; i < 100; ++i) {
        CHECK_EQ(r1.next(), r2.next());
    }
}

TEST_CASE("rng_different_seeds_differ") {
    rng::SplitMix64 r1(42);
    rng::SplitMix64 r2(99);
    int same = 0;
    for (int i = 0; i < 100; ++i) {
        if (r1.next() == r2.next()) same++;
    }
    CHECK_LT(same, 5);  // Should be statistically unlikely
}

TEST_CASE("rng_uniform_range") {
    rng::SplitMix64 r(12345);
    for (int i = 0; i < 1000; ++i) {
        double v = r.uniform();
        CHECK_GE(v, 0.0);
        CHECK_LT(v, 1.0);
    }
}

TEST_CASE("rng_zero_seed_remapped") {
    rng::SplitMix64 r(0);
    CHECK(r.state() == 12345);  // 0 is remapped to default
}

TEST_CASE("rng_hash_deterministic") {
    uint64_t h1 = rng::SplitMix64::hash(42, 5, 10);
    uint64_t h2 = rng::SplitMix64::hash(42, 5, 10);
    CHECK_EQ(h1, h2);
}

TEST_CASE("rng_hash_position_sensitive") {
    uint64_t h1 = rng::SplitMix64::hash(42, 5, 10);
    uint64_t h2 = rng::SplitMix64::hash(42, 10, 5);
    CHECK(h1 != h2);
}

TEST_CASE("rng_is_holdout_deterministic") {
    for (int i = 0; i < 100; ++i) {
        bool a = rng::SplitMix64::is_holdout(42, i, 0, 10);
        bool b = rng::SplitMix64::is_holdout(42, i, 0, 10);
        CHECK_EQ(a, b);
    }
}

TEST_CASE("rng_holdout_fraction_approx") {
    // inv_prob=10 → ~10% holdout
    int count = 0;
    int N = 10000;
    for (int i = 0; i < 100; ++i) {
        for (int j = 0; j < 100; ++j) {
            if (rng::SplitMix64::is_holdout(42, i, j, 10)) count++;
        }
    }
    double frac = static_cast<double>(count) / N;
    CHECK_GT(frac, 0.05);  // Should be ~0.10
    CHECK_LT(frac, 0.15);
}

TEST_CASE("rng_fill_uniform_matrix") {
    rng::SplitMix64 r(42);
    DenseMatrix<double> M(5, 10);
    r.fill_uniform(M);
    // All values in [0, 1)
    CHECK_GE(M.minCoeff(), 0.0);
    CHECK_LT(M.maxCoeff(), 1.0);
    // Not all the same
    CHECK_GT(M.maxCoeff() - M.minCoeff(), 0.1);
}
