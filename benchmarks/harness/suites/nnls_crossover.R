# nnls_crossover.R — CD vs Cholesky NNLS solver crossover benchmark
#
# Benchmarks NMF with CD vs Cholesky solver across ranks.
# k in {4, 8, 16, 24, 32, 48, 64}, 20 iterations, 5 replicates.
# Used to determine the optimal crossover point for solver="auto".

run_suite <- function(datasets, metadata) {
  results <- list()

  ranks <- c(4L, 8L, 16L, 24L, 32L, 48L, 64L)
  solvers <- c("cd", "cholesky")
  maxit <- 20L
  n_reps <- 3L
  A <- datasets$sparse_large

  for (solver in solvers) {
    for (k in ranks) {
      bench_name <- sprintf("nnls_%s_sparse_k%d", solver, k)
      cat(sprintf("  %s ... ", bench_name))

      timing <- bench_time(function()
        nmf(A, k = k, tol = 1e-10, maxit = maxit,
            solver = solver, seed = 42, verbose = FALSE, resource = "cpu"),
        n_reps = n_reps)

      final_loss <- tryCatch(timing$result@misc$train_loss, error = function(e) NA_real_)

      results[[length(results) + 1]] <- list(
        name = bench_name,
        backend = "cpu",
        input = "sparse",
        solver = solver,
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
