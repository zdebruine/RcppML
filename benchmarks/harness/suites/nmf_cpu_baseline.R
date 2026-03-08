# nmf_cpu_baseline.R — CPU NMF benchmark suite
#
# Benchmarks MSE NMF on CPU with sparse and dense inputs.
# k in {8, 16, 32, 64}, 20 iterations, 5 replicates.

run_suite <- function(datasets, metadata) {
  results <- list()

  ranks <- c(8L, 16L, 32L, 64L)
  maxit <- 20L
  n_reps <- 5L

  input_configs <- list(
    list(name = "sparse", data = datasets$sparse_large),
    list(name = "dense",  data = datasets$dense_medium)
  )

  for (ic in input_configs) {
    for (k in ranks) {
      bench_name <- sprintf("nmf_cpu_%s_mse_k%d", ic$name, k)
      cat(sprintf("  %s ... ", bench_name))

      timing <- bench_time(function()
        nmf(ic$data, k = k, tol = 1e-10, maxit = maxit,
            seed = 42, verbose = FALSE, resource = "cpu"),
        n_reps = n_reps)

      # Extract loss from result if available
      final_loss <- tryCatch(timing$result@misc$train_loss, error = function(e) NA_real_)

      results[[length(results) + 1]] <- list(
        name = bench_name,
        backend = "cpu",
        input = ic$name,
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
  }

  results
}
