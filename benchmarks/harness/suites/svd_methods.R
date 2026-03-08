# svd_methods.R — SVD benchmark suite
#
# Benchmarks all 5 SVD methods on CPU.
# k in {5, 10, 20}, sparse + dense, 5 replicates.

run_suite <- function(datasets, metadata) {
  results <- list()

  methods <- c("deflation", "krylov", "lanczos", "irlba", "randomized")
  ranks <- c(5L, 10L, 20L)
  n_reps <- 5L

  input_configs <- list(
    list(name = "sparse", data = datasets$sparse_large),
    list(name = "dense",  data = datasets$dense_medium)
  )

  for (ic in input_configs) {
    for (method in methods) {
      for (k in ranks) {
        bench_name <- sprintf("svd_cpu_%s_%s_k%d", ic$name, method, k)
        cat(sprintf("  %s ... ", bench_name))

        timing <- bench_time(function()
          svd(ic$data, k = k, method = method, tol = 1e-10),
          n_reps = n_reps)

        results[[length(results) + 1]] <- list(
          name = bench_name,
          backend = "cpu",
          input = ic$name,
          method = method,
          rank = k,
          replicates = n_reps,
          mean_sec = timing$mean_sec,
          sd_sec = timing$sd_sec,
          min_sec = timing$min_sec,
          max_sec = timing$max_sec
        )
        cat(sprintf("%.3f +/- %.4f sec\n", timing$mean_sec, timing$sd_sec))
      }
    }
  }

  results
}
