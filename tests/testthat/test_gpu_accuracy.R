# Test GPU Accuracy Regression
#
# Verifies that GPU NMF produces results equivalent to CPU NMF.
# Skipped automatically if GPU is not available.

# Factor agreement (best cosine similarity matching)
factor_agreement <- function(model1, model2) {
  W1 <- get_W(model1); W2 <- get_W(model2)
  k <- ncol(W1)
  sims <- numeric(k)
  for (i in seq_len(k)) {
    best <- -1
    for (j in seq_len(k)) {
      cs <- sum(W1[, i] * W2[, j]) /
            (sqrt(sum(W1[, i]^2)) * sqrt(sum(W2[, j]^2)) + 1e-16)
      if (abs(cs) > abs(best)) best <- cs
    }
    sims[i] <- abs(best)
  }
  mean(sims)
}

test_that("GPU produces same results as CPU (standard NMF)", {
  skip_if_not(exists("gpu_available", where = asNamespace("RcppML")) && RcppML:::gpu_available(), "GPU not available")
  
  m <- load_pbmc3k_matrix()
  # Use a small subset for fast testing
  A <- m[1:500, 1:200]
  A <- as(A, "dgCMatrix")
  
  k <- 8
  seed <- 42L
  maxit <- 30
  tol <- 1e-10
  
  # CPU reference
  options(RcppML.gpu = FALSE)
  cpu_result <- nmf(A, k = k, maxit = maxit, tol = tol, seed = seed, verbose = FALSE)
  
  # GPU result
  options(RcppML.gpu = TRUE)
  gpu_result <- nmf(A, k = k, maxit = maxit, tol = tol, seed = seed, verbose = FALSE)
  options(RcppML.gpu = "auto")
  
  # Both should produce valid NMF objects

  expect_true(is(cpu_result, "nmf") || is.list(cpu_result))
  expect_true(is(gpu_result, "nmf") || is.list(gpu_result))
  
  # Compare using independently computed MSE (avoids loss scale differences)
  cpu_mse <- compute_mse(A, cpu_result)
  gpu_mse <- compute_mse(A, gpu_result)
  
  # MSE should be within 20% — CPU and GPU use different normalization
  # strategies so solutions may differ somewhat, especially at low maxit.
  expect_true(abs(cpu_mse - gpu_mse) / max(cpu_mse, 1e-16) < 0.20,
              label = sprintf("MSE difference: CPU=%.6e, GPU=%.6e", cpu_mse, gpu_mse))
  
  # Factor agreement: at least 0.85 cosine similarity on average
  agreement <- factor_agreement(cpu_result, gpu_result)
  expect_true(agreement >= 0.85,
              label = sprintf("Factor agreement: %.4f (threshold: 0.85)", agreement))
  
  # GPU and CPU internal loss should be on the same scale (both MSE)
  cpu_loss <- if (is(cpu_result, "nmf")) cpu_result@misc$loss else cpu_result$loss
  gpu_loss <- if (is(gpu_result, "nmf")) gpu_result@misc$loss else gpu_result$loss
  if (!is.null(cpu_loss) && !is.null(gpu_loss) && cpu_loss > 0 && gpu_loss > 0) {
    ratio <- gpu_loss / cpu_loss
    expect_true(ratio > 0.1 && ratio < 10,
                label = sprintf("GPU/CPU loss ratio: %.4f (CPU=%.6e, GPU=%.6e)",
                                ratio, cpu_loss, gpu_loss))
  }
})


test_that("GPU produces same results for k=2", {
  skip_if_not(exists("gpu_available", where = asNamespace("RcppML")) && RcppML:::gpu_available(), "GPU not available")
  
  m <- load_pbmc3k_matrix()
  A <- m[1:500, 1:200]
  A <- as(A, "dgCMatrix")
  
  k <- 2
  seed <- 123L
  maxit <- 30
  tol <- 1e-10
  
  options(RcppML.gpu = FALSE)
  cpu_result <- nmf(A, k = k, maxit = maxit, tol = tol, seed = seed, verbose = FALSE)
  
  options(RcppML.gpu = TRUE)
  gpu_result <- nmf(A, k = k, maxit = maxit, tol = tol, seed = seed, verbose = FALSE)
  options(RcppML.gpu = "auto")
  
  # Compare using independently computed MSE
  cpu_mse <- compute_mse(A, cpu_result)
  gpu_mse <- compute_mse(A, gpu_result)
  
  # k=2 uses specialized solver, MSE should still agree
  expect_true(abs(cpu_mse - gpu_mse) / max(cpu_mse, 1e-16) < 0.10,
              label = sprintf("k=2 MSE diff: CPU=%.6e, GPU=%.6e", cpu_mse, gpu_mse))
  
  # Factor agreement
  agreement <- factor_agreement(cpu_result, gpu_result)
  expect_true(agreement >= 0.85,
              label = sprintf("k=2 factor agreement: %.4f", agreement))
  
  # K2 should now compute loss (no longer returns 0)
  gpu_loss <- if (is(gpu_result, "nmf")) gpu_result@misc$loss else gpu_result$loss
  expect_true(!is.null(gpu_loss) && gpu_loss > 0,
              label = sprintf("k=2 GPU loss should be non-zero, got: %s", gpu_loss))
})


test_that("GPU fp32 converges to similar solution as fp64", {
  skip_if_not(exists("gpu_available", where = asNamespace("RcppML")) && RcppML:::gpu_available(), "GPU not available")
  
  m <- load_pbmc3k_matrix()
  A <- m[1:500, 1:200]
  A <- as(A, "dgCMatrix")
  
  k <- 8
  seed <- 42L
  maxit <- 30
  tol <- 1e-10
  
  # fp64 reference (via options escape hatch)
  old_opt <- getOption("RcppML.precision")
  on.exit(options(RcppML.precision = old_opt))
  
  options(RcppML.gpu = TRUE, RcppML.precision = "double")
  ref <- nmf(A, k = k, maxit = maxit, tol = tol, seed = seed,
             verbose = FALSE)
  
  # fp32 (default)
  options(RcppML.precision = "float")
  fp32_result <- nmf(A, k = k, maxit = maxit, tol = tol, seed = seed,
              verbose = FALSE)
  options(RcppML.gpu = "auto")
  
  ref_mse <- compute_mse(A, ref)
  fp32_mse <- compute_mse(A, fp32_result)
  
  # fp32 MSE should be within 20% of fp64 (wider tolerance for precision)
  expect_true(abs(ref_mse - fp32_mse) / max(ref_mse, 1e-16) < 0.20,
              label = sprintf("fp32 vs fp64 MSE: fp64=%.6e, fp32=%.6e", ref_mse, fp32_mse))
  
  # Factor agreement between fp32 and fp64
  agreement <- factor_agreement(ref, fp32_result)
  expect_true(agreement >= 0.85,
              label = sprintf("fp32 vs fp64 factor agreement: %.4f", agreement))
})


test_that("GPU auto-dispatch respects RcppML.gpu option", {
  skip_if_not(exists("gpu_available", where = asNamespace("RcppML")) && RcppML:::gpu_available(), "GPU not available")
  
  m <- load_pbmc3k_matrix()
  A <- m[1:500, 1:200]
  A <- as(A, "dgCMatrix")
  
  # Force CPU
  options(RcppML.gpu = FALSE)
  cpu_result <- nmf(A, k = 4, maxit = 5, tol = 1, verbose = FALSE)
  expect_true(is(cpu_result, "nmf"))  # CPU path returns nmf S4 object
  
  # Force GPU
  options(RcppML.gpu = TRUE)
  gpu_result <- nmf(A, k = 4, maxit = 5, tol = 1, verbose = FALSE)
  # GPU path may return nmf S4 object or list with backend = "gpu"
  expect_true(is(gpu_result, "nmf") || 
              (is.list(gpu_result) && identical(gpu_result$backend, "gpu")))
  
  # Auto mode (default)
  options(RcppML.gpu = "auto")
})


test_that("Multi-GPU produces valid results", {
  skip_if_not(exists("gpu_available", where = asNamespace("RcppML")) && RcppML:::gpu_available(), "GPU not available")
  
  info <- gpu_info()
  skip_if(nrow(info) < 2, "Only 1 GPU available, skipping multi-GPU test")
  
  m <- load_pbmc3k_matrix()
  A <- m[1:500, 1:200]
  A <- as(A, "dgCMatrix")
  
  k <- 8
  seed <- 42L
  maxit <- 20
  tol <- 1e-10
  
  # Single GPU
  options(RcppML.gpu = TRUE, RcppML.max_gpus = 1L)
  single <- nmf(A, k = k, maxit = maxit, tol = tol, seed = seed, verbose = FALSE)
  
  # Multi GPU
  options(RcppML.gpu = TRUE, RcppML.max_gpus = 0L)  # 0 = all available
  multi <- nmf(A, k = k, maxit = maxit, tol = tol, seed = seed, verbose = FALSE)
  options(RcppML.gpu = "auto", RcppML.max_gpus = 0L)
  
  single_mse <- compute_mse(A, single)
  multi_mse <- compute_mse(A, multi)
  
  # Both should converge to similar MSE
  expect_true(abs(single_mse - multi_mse) / max(single_mse, 1e-16) < 0.15,
              label = sprintf("Single GPU MSE=%.6e, Multi GPU MSE=%.6e",
                              single_mse, multi_mse))
  
  # Factor agreement
  agreement <- factor_agreement(single, multi)
  expect_true(agreement >= 0.85,
              label = sprintf("Single vs Multi GPU factor agreement: %.4f", agreement))
})


# ============================================================================
# Tight GPU vs CPU parity (P0 hardening)
# ============================================================================

test_that("GPU sparse NMF tightly matches CPU (loss <10%, factor agreement >0.95)", {
  skip_if_not(exists("gpu_available", where = asNamespace("RcppML")) &&
                RcppML:::gpu_available(), "GPU not available")

  m <- load_pbmc3k_matrix()
  A <- m[1:500, 1:200]
  A <- as(A, "dgCMatrix")

  k <- 8; seed <- 42L; maxit <- 100; tol <- 1e-10

  cpu_result <- nmf(A, k = k, seed = seed, maxit = maxit, tol = tol,
                    resource = "cpu", verbose = FALSE)
  gpu_result <- nmf(A, k = k, seed = seed, maxit = maxit, tol = tol,
                    resource = "gpu", verbose = FALSE)

  # Full MSE (all entries, not just nonzeros)
  cpu_mse <- mse(cpu_result@w, cpu_result@d, cpu_result@h, A)
  gpu_mse <- mse(gpu_result@w, gpu_result@d, gpu_result@h, A)

  # Loss within 10% (tighter than existing 20%, GPU uses fp32)
  rel_diff <- abs(cpu_mse - gpu_mse) / max(cpu_mse, 1e-16)
  expect_true(rel_diff < 0.10,
              label = sprintf("Tight loss parity: CPU=%.6e, GPU=%.6e, rel_diff=%.4f",
                              cpu_mse, gpu_mse, rel_diff))

  # Factor agreement > 0.95 (tighter than existing 0.85)
  agreement <- factor_agreement(cpu_result, gpu_result)
  expect_true(agreement >= 0.95,
              label = sprintf("Tight factor agreement: %.4f (threshold: 0.95)", agreement))
})
