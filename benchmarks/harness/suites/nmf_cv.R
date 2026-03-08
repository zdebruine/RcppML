# nmf_cv.R — Cross-validation NMF benchmark suite
#
# Benchmarks CV NMF on CPU (and GPU if available).
# k=16, 20 iterations, 5 replicates.

run_suite <- function(datasets, metadata) {
  results <- list()

  k <- 16L
  maxit <- 20L
  n_reps <- 5L
  A <- datasets$sparse_large

  has_gpu <- isTRUE(tryCatch(RcppML:::gpu_available(), error = function(e) FALSE))
  backends <- if (has_gpu) c("cpu", "gpu") else "cpu"

  for (backend in backends) {
    bench_name <- sprintf("nmf_cv_%s_sparse_k%d", backend, k)
    cat(sprintf("  %s ... ", bench_name))

    timing <- bench_time(function()
      nmf(A, k = k, tol = 1e-10, maxit = maxit,
          test_fraction = 0.1, seed = 42,
          verbose = FALSE, resource = backend),
      n_reps = n_reps)

    final_loss <- tryCatch(timing$result@misc$test_loss, error = function(e) NA_real_)

    results[[length(results) + 1]] <- list(
      name = bench_name,
      backend = backend,
      input = "sparse",
      distribution = "mse",
      rank = k,
      iterations = maxit,
      replicates = n_reps,
      mean_sec = timing$mean_sec,
      sd_sec = timing$sd_sec,
      min_sec = timing$min_sec,
      max_sec = timing$max_sec,
      final_loss = final_loss
    )
    cat(sprintf("%.3f +/- %.4f sec\n", timing$mean_sec, timing$sd_sec))
  }

  results
}
