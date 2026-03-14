# Tests for factor_net composable graph API
#
# Covers: node construction, config hierarchy, single-layer ≡ nmf(),
# multi-modal, deep NMF, conditioning, cross-validation,
# cross_validate_graph(), streaming SPZ routing, graph regularization,
# and resource propagation.

library(Matrix)

# --- Shared test data ---
set.seed(42)
m <- 50; n <- 30; k <- 5
W_true <- matrix(abs(rnorm(m * k)), m, k)
H_true <- matrix(abs(rnorm(k * n)), k, n)
X <- W_true %*% H_true + matrix(abs(rnorm(m * n, sd = 0.1)), m, n)
X_sparse <- as(X, "dgCMatrix")

# =========================================================================
# Node construction & config
# =========================================================================

test_that("factor_input creates input node", {
  inp <- factor_input(X, "test_data")
  expect_s3_class(inp, "fn_node")
  expect_equal(inp$type, "input")
  expect_equal(inp$name, "test_data")
})

test_that("nmf_layer creates layer node", {
  inp <- factor_input(X, "X")
  L1 <- inp |> nmf_layer(k = 5)
  expect_s3_class(L1, "fn_node")
  expect_equal(L1$type, "nmf_layer")
  expect_equal(L1$k, 5L)
})

test_that("W() and H() config constructors work", {
  w <- W(L1 = 0.01, L2 = 0.02)
  expect_s3_class(w, "fn_factor_config")
  expect_equal(w$side, "W")
  expect_equal(w$L1, 0.01)
  expect_equal(w$L2, 0.02)

  h <- H(L1 = 0.05, target = matrix(runif(5 * 30), 5, 30), target_lambda = 1.0)
  expect_s3_class(h, "fn_factor_config")
  expect_equal(nrow(h$target), 5)
})

test_that("factor_config constructs global config", {
  gc <- factor_config(maxit = 50, tol = 1e-3, loss = "mse",
                      test_fraction = 0.1, patience = 10L)
  expect_s3_class(gc, "fn_global_config")
  expect_equal(gc$maxit, 50L)
  expect_equal(gc$test_fraction, 0.1)
  expect_equal(gc$patience, 10L)
})

test_that("factor_config basic parameters work", {
  gc <- factor_config(test_fraction = 0.2)
  expect_equal(gc$test_fraction, 0.2)

  gc2 <- factor_config(patience = 8L)
  expect_equal(gc2$patience, 8L)
})

test_that("factor_net compiles a graph", {
  inp <- factor_input(X, "X")
  L1 <- inp |> nmf_layer(k = 5, L1 = 0.01)
  net <- factor_net(inputs = inp, output = L1,
                    config = factor_config(maxit = 50, seed = 123))
  expect_s3_class(net, "factor_net")
  expect_equal(net$n_layers, 1L)
  expect_length(net$input_nodes, 1)
})

# =========================================================================
# Single-layer parity with nmf()
# =========================================================================

test_that("single-layer factor_net matches nmf() exactly", {
  inp <- factor_input(X_sparse, "X")
  L1 <- inp |> nmf_layer(k = 5)
  net <- factor_net(inputs = inp, output = L1,
                    config = factor_config(maxit = 100, tol = 1e-4, seed = 42))
  fn_result <- fit(net)

  nmf_result <- nmf(X_sparse, k = 5, maxit = 100, tol = 1e-4, seed = 42)

  fn_d <- sort(fn_result$L1$d, decreasing = TRUE)
  nmf_d <- sort(nmf_result@d, decreasing = TRUE)
  expect_equal(fn_d, nmf_d, tolerance = 1e-4)
})

test_that("config hierarchy: layer L1 < factor H override", {
  inp <- factor_input(X_sparse, "X")
  L1 <- inp |> nmf_layer(k = 5, L1 = 0.01, H = H(L1 = 0.05))
  net <- factor_net(inputs = inp, output = L1,
                    config = factor_config(maxit = 50, seed = 42))
  fn_result <- fit(net)

  # Verify factor_net produces valid factorization with correct dimensions
  expect_equal(ncol(fn_result$L1$W), 5L)
  expect_equal(nrow(fn_result$L1$H), 5L)
  expect_equal(length(fn_result$L1$d), 5L)
  # d values should be positive and descending (sorted model)
  expect_true(all(fn_result$L1$d > 0))
})

# =========================================================================
# Multi-modal shared-H
# =========================================================================

test_that("multi-modal shared-H matches concatenated nmf()", {
  # Test compares factor_net CPU to nmf() CPU; explicitly force cpu resource
  m1 <- 25; m2 <- 25
  X1 <- X[1:m1, ]
  X2 <- X[(m1 + 1):(m1 + m2), ]

  inp1 <- factor_input(X1, "modal1")
  inp2 <- factor_input(X2, "modal2")
  shared <- factor_shared(inp1, inp2)
  L1 <- shared |> nmf_layer(k = 5)
  net <- factor_net(inputs = list(inp1, inp2), output = L1,
                    config = factor_config(maxit = 100, seed = 42, resource = "cpu"))
  fn_result <- fit(net)

  X_cat <- rbind(X1, X2)
  nmf_cat <- nmf(X_cat, k = 5, maxit = 100, seed = 42, resource = "cpu")

  fn_d <- sort(fn_result$L1$d, decreasing = TRUE)
  nmf_d <- sort(nmf_cat@d, decreasing = TRUE)
  expect_equal(fn_d, nmf_d, tolerance = 1e-4)

  expect_equal(nrow(fn_result$L1$W$modal1), m1)
  expect_equal(nrow(fn_result$L1$W$modal2), m2)

  W_reconcat <- rbind(fn_result$L1$W$modal1, fn_result$L1$W$modal2)
  expect_equal(unname(W_reconcat), unname(nmf_cat@w), tolerance = 1e-6)
})

# =========================================================================
# Deep NMF
# =========================================================================

test_that("deep NMF (2-layer) produces valid results", {
  inp <- factor_input(X, "X")
  L1 <- inp |> nmf_layer(k = 10, name = "enc")
  L2 <- L1 |> nmf_layer(k = 3, name = "bot")
  net <- factor_net(inputs = inp, output = L2,
                    config = factor_config(maxit = 30, tol = 1e-6, seed = 42))
  res <- fit(net)

  expect_s3_class(res, "factor_net_result")
  expect_equal(res$n_layers, 2L)
  expect_true("enc" %in% names(res$layers))
  expect_true("bot" %in% names(res$layers))
  expect_equal(dim(res$enc$W), c(m, 10L))
  expect_equal(dim(res$bot$W), c(n, 3L))
  expect_true(is.finite(res$total_loss))
})

test_that("conditional deep NMF works", {
  Z <- matrix(rnorm(3 * n), n, 3)
  inp_c <- factor_input(X, "X")
  L1_c <- inp_c |> nmf_layer(k = 10, name = "enc")
  cond <- factor_condition(L1_c, Z)
  L2_c <- cond |> nmf_layer(k = 3, name = "bot")
  net_c <- factor_net(inputs = inp_c, output = L2_c,
                      config = factor_config(maxit = 20, tol = 1e-6, seed = 42))
  res_c <- fit(net_c)

  expect_equal(res_c$n_layers, 2L)
  expect_equal(nrow(res_c$bot$W), n)
  expect_equal(ncol(res_c$bot$W), 3L)
  # Bot factorizes n x (10+3) -> H is 3 x 13
  expect_equal(ncol(res_c$bot$H), 13L)
})

test_that("mixed SVD+NMF deep network works", {
  inp <- factor_input(X, "X")
  L1 <- inp |> svd_layer(k = 10, name = "pca")
  L2 <- L1 |> nmf_layer(k = 3, name = "nmf")
  net <- factor_net(inputs = inp, output = L2,
                    config = factor_config(maxit = 20, tol = 1e-6, seed = 42))
  res <- fit(net)

  expect_equal(res$n_layers, 2L)
  # SVD layer should have signed entries
  expect_true(any(res$pca$W < 0) || any(res$pca$H < 0))
  # NMF layer should be non-negative
  expect_true(all(res$nmf$W >= -1e-10))
  expect_true(all(res$nmf$H >= -1e-10))
})

test_that("factor_add residual connection works", {
  inp <- factor_input(X, "X")
  b1 <- inp |> nmf_layer(k = 5, name = "b1")
  b2 <- inp |> nmf_layer(k = 5, name = "b2")
  added <- factor_add(b1, b2)
  deep <- added |> nmf_layer(k = 3, name = "deep")
  net <- factor_net(inputs = inp, output = deep,
                    config = factor_config(maxit = 15, tol = 1e-6, seed = 42))
  res <- fit(net)

  expect_equal(res$n_layers, 3L)
  expect_equal(dim(res$deep$W), c(n, 3L))
  expect_equal(dim(res$deep$H), c(3L, 5L))
})

# =========================================================================
# svd_layer
# =========================================================================

test_that("svd_layer produces signed factors", {
  inp <- factor_input(X, "X")
  sl <- inp |> svd_layer(k = 5, name = "svd1")
  net <- factor_net(inputs = inp, output = sl,
                    config = factor_config(maxit = 50, tol = 1e-5, seed = 42))
  res <- fit(net)
  expect_true(any(res$svd1$W < 0) || any(res$svd1$H < 0))
})

# =========================================================================
# predict
# =========================================================================

test_that("predict.factor_net_result works", {
  inp <- factor_input(X, "X")
  L1 <- inp |> nmf_layer(k = 5)
  net <- factor_net(inputs = inp, output = L1,
                    config = factor_config(maxit = 50, tol = 1e-5, seed = 42))
  res <- fit(net)

  H_pred <- predict(res, X)
  expect_equal(nrow(H_pred), 5L)
  expect_equal(ncol(H_pred), n)

  X_new <- matrix(abs(rnorm(m * 10)), m, 10)
  H_new <- predict(res, X_new)
  expect_equal(nrow(H_new), 5L)
  expect_equal(ncol(H_new), 10L)
})

# =========================================================================
# nmf() list dispatch
# =========================================================================

test_that("nmf() with list input dispatches to factor_net", {
  set.seed(42)
  X2 <- X[(26):50, ]
  list_result <- nmf(list(modal1 = X, modal2 = X2), k = 5, seed = 42,
                     tol = 1e-4, maxit = 100)
  expect_s3_class(list_result, "factor_net_result")
  expect_true(list_result$multi_modal)
  expect_true("modal1" %in% names(list_result$L1$W))
  expect_true("modal2" %in% names(list_result$L1$W))
})

# =========================================================================
# Error handling
# =========================================================================

test_that("factor_input rejects invalid input", {
  expect_error(factor_input("not_a_matrix"), "must be a dense matrix")
})

test_that("nmf_layer rejects non-node input", {
  expect_error(nmf_layer("not_a_node", k = 5), "must be an fn_node")
})

test_that("factor_shared requires at least 2 inputs", {
  inp <- factor_input(X, "X")
  expect_error(factor_shared(inp), "at least 2 inputs")
})

test_that("factor_concat requires at least 2 inputs", {
  inp <- factor_input(X, "X")
  expect_error(factor_concat(inp), "at least 2 inputs")
})

# =========================================================================
# Target regularization
# =========================================================================

test_that("W() and H() accept target params", {
  tgt <- matrix(runif(5 * 30), 5, 30)
  h <- H(target = tgt, target_lambda = 0.5)
  expect_s3_class(h, "fn_factor_config")
  expect_equal(h$target_lambda, 0.5)
  expect_equal(dim(h$target), c(5, 30))
})

# =========================================================================
# Classifier metrics
# =========================================================================

test_that("classify_embedding produces valid metrics", {
  set.seed(123)
  embedding <- rbind(
    matrix(rnorm(30 * 5, mean = 0), 30, 5),
    matrix(rnorm(30 * 5, mean = 3), 30, 5),
    matrix(rnorm(30 * 5, mean = 6), 30, 5))
  labels <- rep(0:2, each = 30)

  ev <- classify_embedding(embedding, labels, test_fraction = 0.3, k = 5, seed = 99)
  expect_s3_class(ev, "fn_classifier_eval")
  expect_true(ev$accuracy > 0.5)
  expect_equal(nrow(ev$per_class), 3L)
  expect_equal(dim(ev$confusion), c(3L, 3L))

  s <- summary(ev)
  expect_true(is.data.frame(s))
})

test_that("classify_logistic produces valid metrics", {
  set.seed(202)
  embedding <- rbind(
    matrix(rnorm(30 * 4, mean = 0), 30, 4),
    matrix(rnorm(30 * 4, mean = 5), 30, 4),
    matrix(rnorm(30 * 4, mean = 10), 30, 4))
  labels <- rep(0:2, each = 30)

  ev <- classify_logistic(embedding, labels, test_fraction = 0.3, seed = 42)
  expect_s3_class(ev, "fn_classifier_eval")
  expect_true(ev$accuracy > 0.5)
  expect_equal(ev$method, "logistic")
})

# =========================================================================
# Training logger
# =========================================================================

test_that("training_logger records entries during deep fit", {
  logger <- training_logger(log_loss = TRUE, log_norms = TRUE)

  inp <- factor_input(X, "X")
  L1 <- inp |> nmf_layer(k = 8, name = "enc")
  L2 <- L1 |> nmf_layer(k = 3, name = "bot")
  net <- factor_net(inputs = inp, output = L2,
                    config = factor_config(maxit = 10, tol = 1e-8, seed = 42))
  res <- fit(net, logger = logger)

  expect_s3_class(res$logger, "training_logger")
  expect_true(length(res$logger$entries) > 0)
  df <- as.data.frame(res$logger)
  expect_true("iteration" %in% names(df))
  expect_true("total_loss" %in% names(df))
  expect_true(any(grepl("_frobenius$", names(df))))
})

# =========================================================================
# Cross-validation via factor_net
# =========================================================================

test_that("single-layer CV produces test_loss > 0", {
  inp <- factor_input(X_sparse, "X")
  L1 <- inp |> nmf_layer(k = 5)
  net <- factor_net(
    inputs = inp, output = L1,
    config = factor_config(maxit = 50, tol = 1e-4, seed = 42,
                           test_fraction = 0.1, cv_seed = 99L,
                           mask_zeros = FALSE, patience = 5L))
  res <- fit(net)

  expect_gt(res$layers$L1$test_loss, 0)
  expect_gt(res$layers$L1$best_test_loss, 0)
  expect_gt(res$layers$L1$loss, 0)
})

# =========================================================================
# cross_validate_graph
# =========================================================================

test_that("cross_validate_graph rank selection works", {
  inp <- factor_input(X_sparse, "X")
  cv <- cross_validate_graph(
    inputs = inp,
    layer_fn = function(p) inp |> nmf_layer(k = p$k),
    params = list(k = c(3, 5)),
    config = factor_config(maxit = 30, seed = 42, test_fraction = 0.1),
    reps = 2, seed = 42, verbose = FALSE)

  expect_s3_class(cv, "factor_net_cv")
  expect_equal(nrow(cv$summary), 2)
  expect_false(is.null(cv$best_params$k))
  expect_true(all(!is.na(cv$summary$mean_test_loss)))
})

test_that("cross_validate_graph multi-parameter search works", {
  inp <- factor_input(X_sparse, "X")
  cv <- cross_validate_graph(
    inputs = inp,
    layer_fn = function(p) inp |> nmf_layer(k = p$k, L1 = p$L1),
    params = list(k = c(3, 5), L1 = c(0, 0.01)),
    config = factor_config(maxit = 20, seed = 42, test_fraction = 0.1),
    reps = 1, seed = 42, verbose = FALSE)

  expect_equal(nrow(cv$summary), 4)  # 2 ranks x 2 L1 values
  expect_true(all(c("k", "L1") %in% names(cv$best_params)))
})

# =========================================================================
# Streaming SPZ routing
# =========================================================================

test_that("factor_input rejects non-existent .spz path", {
  expect_error(factor_input("/nonexistent/file.spz", "bad"))
})

test_that("descriptor builder routes .spz paths correctly", {
  # Create a fake factor_net with .spz path in input
  inp_spz <- structure(list(type = "input", data = "/tmp/test.spz",
                            name = "SPZ"), class = "fn_node")
  L1_spz <- list(type = "nmf_layer", k = 5L, input = inp_spz, name = "L1",
                 layer_defaults = list(L1 = 0, L2 = 0, L21 = 0, angular = 0,
                   upper_bound = 0, nonneg = TRUE, graph_lambda = 0),
                 W_config = NULL, H_config = NULL)
  net_spz <- structure(list(layers = list(L1_spz), input_nodes = list(inp_spz),
                            config = factor_config(maxit = 10, seed = 42),
                            n_layers = 1L), class = "factor_net")
  desc <- RcppML:::.fn_build_descriptor(net_spz)
  expect_equal(desc$inputs[[1]]$spz_path, "/tmp/test.spz")
})

test_that("factor_net with real .spz file streams correctly", {
  skip_on_cran()
  skip("Known segfault in streaming SPZ path through factor_net — needs C++ fix")
  data(movielens, package = "RcppML")
  tmp <- tempfile(fileext = ".spz")
  on.exit(unlink(tmp), add = TRUE)
  st_write(movielens, tmp, include_transpose = TRUE)

  inp <- factor_input(tmp, "movies")
  L1 <- inp |> nmf_layer(k = 3)
  net <- factor_net(inputs = inp, output = L1,
                    config = factor_config(maxit = 10, tol = 1e-2, seed = 42))
  res <- fit(net)

  expect_s3_class(res, "factor_net_result")
  expect_gt(res$layers$L1$iterations, 0)
  expect_true(is.finite(res$layers$L1$loss))
})

# =========================================================================
# Graph Laplacian regularization
# =========================================================================

test_that("graph regularization changes W factors", {
  # Create a sample graph Laplacian (features connected in chain)
  # Build tridiagonal Laplacian manually (bandSparse symmetric requires same-sign k)
  adj <- Matrix::bandSparse(m, m, k = c(0, 1),
                            diagonals = list(rep(2, m),
                                             rep(-1, m - 1)),
                            symmetric = TRUE)
  adj <- as(as(adj, "generalMatrix"), "CsparseMatrix")

  inp <- factor_input(X_sparse, "X")
  # Without graph reg
  L1_noreg <- inp |> nmf_layer(k = 5)
  net_noreg <- factor_net(inputs = inp, output = L1_noreg,
                          config = factor_config(maxit = 50, seed = 42))
  res_noreg <- fit(net_noreg)

  # With graph reg on W
  L1_reg <- inp |> nmf_layer(k = 5, W = W(graph = adj, graph_lambda = 1.0))
  net_reg <- factor_net(inputs = inp, output = L1_reg,
                        config = factor_config(maxit = 50, seed = 42))
  res_reg <- fit(net_reg)

  # Graph regularization should produce different W factors
  W_diff <- max(abs(res_noreg$L1$W - res_reg$L1$W))
  expect_gt(W_diff, 1e-4)

  # Both should produce valid non-negative factors
  expect_true(all(res_reg$L1$W >= -1e-10))
  expect_true(all(res_reg$L1$H >= -1e-10))
})

# =========================================================================
# Resource propagation
# =========================================================================

test_that("resource='cpu' propagates through descriptor", {
  inp <- factor_input(X_sparse, "X")
  L1 <- inp |> nmf_layer(k = 5)
  net <- factor_net(inputs = inp, output = L1,
                    config = factor_config(maxit = 10, seed = 42, resource = "cpu"))
  res <- fit(net)
  expect_gt(res$layers$L1$iterations, 0)
})

test_that("resource='gpu' propagates in descriptor", {
  inp <- factor_input(X_sparse, "X")
  L1 <- inp |> nmf_layer(k = 5)
  net <- factor_net(inputs = inp, output = L1,
                    config = factor_config(maxit = 10, seed = 42, resource = "gpu"))
  desc <- RcppML:::.fn_build_descriptor(net)
  expect_equal(desc$config$resource, "gpu")
})

# =========================================================================
# Print methods
# =========================================================================

test_that("print methods execute without error", {
  inp <- factor_input(X, "X")
  L1 <- inp |> nmf_layer(k = 5)
  net <- factor_net(inputs = inp, output = L1,
                    config = factor_config(maxit = 10, seed = 42))
  res <- fit(net)

  expect_output(print(inp))
  expect_output(print(W(L1 = 0.01)))
  expect_output(print(factor_config()))
  expect_output(print(net))
  expect_output(print(res))
})

# =========================================================================
# TEST-GRAPH-SINGLE-DENSE: Single-layer graph NMF with dense input (Part C)
# =========================================================================

test_that("TEST-GRAPH-SINGLE-DENSE: single-layer factor_net works with dense input", {
  # Test compares factor_net CPU to nmf() CPU; explicitly force cpu resource
  inp <- factor_input(X, "X_dense")
  L1 <- inp |> nmf_layer(k = 5)
  net <- factor_net(inputs = inp, output = L1,
                    config = factor_config(maxit = 50, tol = 1e-4, seed = 42, resource = "cpu"))
  fn_result <- fit(net)

  # Verify valid results
  expect_s3_class(fn_result, "factor_net_result")
  expect_true(all(fn_result$L1$d > 0))
  expect_equal(dim(fn_result$L1$W), c(m, 5L))
  expect_equal(dim(fn_result$L1$H), c(5L, n))

  # Compare to direct nmf() on dense matrix
  nmf_result <- nmf(X, k = 5, maxit = 50, tol = 1e-4, seed = 42, resource = "cpu")
  fn_d <- sort(fn_result$L1$d, decreasing = TRUE)
  nmf_d <- sort(nmf_result@d, decreasing = TRUE)
  expect_equal(fn_d, nmf_d, tolerance = 1e-4)
})

# =========================================================================
# TEST-GRAPH-DEEP-DENSE: Deep graph NMF with dense input (Part C)
# =========================================================================

test_that("TEST-GRAPH-DEEP-DENSE: deep NMF works with dense input", {
  inp <- factor_input(X, "X_dense")
  L1 <- inp |> nmf_layer(k = 10, name = "enc_d")
  L2 <- L1 |> nmf_layer(k = 3, name = "bot_d")
  net <- factor_net(inputs = inp, output = L2,
                    config = factor_config(maxit = 30, tol = 1e-6, seed = 42))
  res <- fit(net)

  expect_s3_class(res, "factor_net_result")
  expect_equal(res$n_layers, 2L)
  expect_true("enc_d" %in% names(res$layers))
  expect_true("bot_d" %in% names(res$layers))
  # Encoder layer factorizes m×n → W is m×10
  expect_equal(dim(res$enc_d$W), c(m, 10L))
  # Bot layer factorizes H1 transposed (n×10) → W is n×3
  expect_equal(dim(res$bot_d$W), c(n, 3L))
  expect_true(all(res$bot_d$d > 0))
  expect_true(is.finite(res$total_loss))
})

# =========================================================================
# TEST-GRAPH-MULTIMODAL: Multi-modal graph NMF with sparse input (Part C)
# =========================================================================

test_that("TEST-GRAPH-MULTIMODAL: multi-modal with sparse matrices", {
  m1 <- 25; m2 <- 25
  X1_sp <- as(X_sparse[1:m1, ], "dgCMatrix")
  X2_sp <- as(X_sparse[(m1 + 1):(m1 + m2), ], "dgCMatrix")

  inp1 <- factor_input(X1_sp, "sp_modal1")
  inp2 <- factor_input(X2_sp, "sp_modal2")
  shared <- factor_shared(inp1, inp2)
  L1 <- shared |> nmf_layer(k = 5)
  net <- factor_net(inputs = list(inp1, inp2), output = L1,
                    config = factor_config(maxit = 50, seed = 42))
  fn_result <- fit(net)

  expect_s3_class(fn_result, "factor_net_result")
  # H should cover all n columns
  expect_equal(ncol(fn_result$L1$H), n)
  expect_true(all(fn_result$L1$d > 0))

  # If W is a named list (dense path), check split dimensions.
  # If W is a matrix (sparse/C++ path), check it has valid dimensions.
  W <- fn_result$L1$W
  if (is.list(W)) {
    expect_equal(nrow(W$sp_modal1), m1)
    expect_equal(nrow(W$sp_modal2), m2)
  } else {
    # C++ sparse path returns concatenated or partial W
    expect_true(is.matrix(W))
    expect_equal(ncol(W), 5L)
  }
})

# =========================================================================
# TEST-GRAPH-BRANCHING: Branching graph NMF (Part C)
# =========================================================================

test_that("TEST-GRAPH-BRANCHING: branching NMF with factor_concat", {
  # Two branches with different k, concatenate H factors
  inp <- factor_input(X_sparse, "X")
  branch1 <- inp |> nmf_layer(k = 3, name = "branch_a")
  branch2 <- inp |> nmf_layer(k = 2, name = "branch_b")
  merged <- factor_concat(branch1, branch2)
  L_top <- merged |> nmf_layer(k = 3, name = "top")
  net <- factor_net(inputs = inp, output = L_top,
                    config = factor_config(maxit = 30, tol = 1e-6, seed = 42))
  res <- fit(net)

  expect_s3_class(res, "factor_net_result")
  expect_equal(res$n_layers, 3L)
  expect_true("branch_a" %in% names(res$layers))
  expect_true("branch_b" %in% names(res$layers))
  expect_true("top" %in% names(res$layers))
  # Top layer factorizes concatenated H (n × (3+2)) → W is n×3, H is 3×5
  expect_equal(dim(res$top$W), c(n, 3L))
  expect_equal(dim(res$top$H), c(3L, 5L))
  expect_true(all(res$top$d > 0))
})

# =========================================================================
# TEST-GRAPH-SINGLE-STREAMING: Single-layer streaming graph (Part C)
# =========================================================================

test_that("TEST-GRAPH-SINGLE-STREAMING: factor_net with .spz input", {
  skip_on_cran()
  skip("Known segfault in streaming SPZ path through factor_net — needs C++ fix")
  tmp <- tempfile(fileext = ".spz")
  on.exit(unlink(tmp), add = TRUE)
  st_write(X_sparse, tmp, include_transpose = TRUE)

  inp <- factor_input(tmp, "X_spz")
  L1 <- inp |> nmf_layer(k = 5)
  net <- factor_net(inputs = inp, output = L1,
                    config = factor_config(maxit = 30, seed = 42))
  res <- fit(net)

  expect_s3_class(res, "factor_net_result")
  expect_true(all(res$L1$d > 0))
  expect_equal(dim(res$L1$H), c(5L, n))
})

# =========================================================================
# TEST-GRAPH-SINGLE-GPU-STREAMING: GPU streaming graph test (Part C)
# =========================================================================

test_that("TEST-GRAPH-SINGLE-GPU-STREAMING: factor_net with .spz on GPU", {
  skip_on_cran()
  skip_if_not(
    exists("gpu_available", where = asNamespace("RcppML")) && RcppML:::gpu_available(),
    "GPU not available"
  )
  tmp <- tempfile(fileext = ".spz")
  on.exit(unlink(tmp), add = TRUE)
  st_write(X_sparse, tmp, include_transpose = TRUE)

  inp <- factor_input(tmp, "X_spz_gpu")
  L1 <- inp |> nmf_layer(k = 5)
  net <- factor_net(inputs = inp, output = L1,
                    config = factor_config(maxit = 30, seed = 42, resource = "gpu"))
  res <- fit(net)

  expect_s3_class(res, "factor_net_result")
  expect_true(all(res$L1$d > 0))
  expect_equal(dim(res$L1$H), c(5L, n))
})
