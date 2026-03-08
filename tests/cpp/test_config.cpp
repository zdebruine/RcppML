// test_config.cpp — NMFConfig validation and defaults

#include "test_framework.hpp"
#include <FactorNet/core/config.hpp>

using namespace FactorNet;

TEST_CASE("config_defaults") {
    NMFConfig<double> cfg;
    CHECK_EQ(cfg.rank, 10);
    CHECK_EQ(cfg.max_iter, 100);
    CHECK_NEAR(cfg.tol, 1e-4, 1e-10);
    CHECK_EQ(cfg.verbose, false);
    CHECK_EQ(cfg.projective, false);
    CHECK_EQ(cfg.symmetric, false);
}

TEST_CASE("config_validate_valid") {
    NMFConfig<double> cfg;
    cfg.rank = 5;
    cfg.max_iter = 10;
    cfg.validate();  // Should not throw
    CHECK(true);
}

TEST_CASE("config_validate_invalid_rank") {
    NMFConfig<double> cfg;
    cfg.rank = 0;
    CHECK_THROWS(cfg.validate());
}

TEST_CASE("config_validate_negative_rank") {
    NMFConfig<double> cfg;
    cfg.rank = -1;
    CHECK_THROWS(cfg.validate());
}

TEST_CASE("config_is_cv") {
    NMFConfig<double> cfg;
    // Default holdout_fraction is 0.1, so is_cv() is true by default
    CHECK_EQ(cfg.is_cv(), true);

    cfg.holdout_fraction = 0.0;
    CHECK_EQ(cfg.is_cv(), false);

    cfg.holdout_fraction = 0.2;
    CHECK_EQ(cfg.is_cv(), true);
}

TEST_CASE("config_requires_irls") {
    NMFConfig<double> cfg;
    CHECK_EQ(cfg.requires_irls(), false);

    cfg.loss.type = LossType::GP;
    CHECK_EQ(cfg.requires_irls(), true);

    cfg.loss.type = LossType::NB;
    CHECK_EQ(cfg.requires_irls(), true);

    cfg.loss.type = LossType::MSE;
    CHECK_EQ(cfg.requires_irls(), false);
}

TEST_CASE("config_effective_cv_seed") {
    NMFConfig<double> cfg;
    cfg.seed = 42;
    cfg.cv_seed = 0;
    CHECK_EQ(cfg.effective_cv_seed(), 42u);

    cfg.cv_seed = 99;
    CHECK_EQ(cfg.effective_cv_seed(), 99u);
}

TEST_CASE("loss_config_defaults") {
    LossConfig<double> lc;
    CHECK(lc.type == LossType::MSE);
    CHECK_NEAR(lc.huber_delta, 1.0, 1e-10);
    CHECK_EQ(lc.robust_delta, 0.0);
}
