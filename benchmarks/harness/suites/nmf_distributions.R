# nmf_distributions.R — Distribution benchmark suite
#
# Benchmarks GP, NB, Gamma distributions on CPU (and GPU if available).
# k=16, 10 iterations, 5 replicates.

run_suite <- function(datasets, metadata) {
  results <- list()

  distributions <- c("gp", "nb", "gamma")
  k <- 16L
  maxit <- 5L
  n_reps <- 3L
  A <- datasets$sparse_large

  has_gpu <- isTRUE(tryCatch(RcppML:::gpu_available(), error = function(e) FALSE))
  backends <- if (has_gpu) c("cpu", "gpu") else "cpu"

  for (dist in distributions) {
    for (backend in backends) {
      bench_name <- sprintf("nmf_%s_sparse_%s_k%d", backend, dist, k)
      cat(sprintf("  %s ... ", bench_name))

      timing <- bench_time(function()
        nmf(A, k = k, tol = 1e-10, maxit = maxit,
            loss = dist, seed = 42, verbose = FALSE, resource = backend),
        n_reps = n_reps)

      final_loss <- tryCatch(timing$result@misc$train_loss, error = function(e) NA_real_)

      results[[length(results) + 1]] <- list(
        name = bench_name,
        backend = backend,
        input = "sparse",
        distribution = dist,
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
  }

  results
}
