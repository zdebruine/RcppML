# nmf_streaming.R — Streaming vs in-memory NMF benchmark suite
#
# Compares streaming (SPZ) vs in-memory NMF performance.
# k=16, 20 iterations, 5 replicates.

run_suite <- function(datasets, metadata) {
  results <- list()

  k <- 16L
  maxit <- 20L
  n_reps <- 5L

  has_gpu <- isTRUE(tryCatch(RcppML:::gpu_available(), error = function(e) FALSE))
  backends <- if (has_gpu) c("cpu", "gpu") else "cpu"

  input_configs <- list(
    list(name = "sparse", data = datasets$sparse_large, writer = "sp_write"),
    list(name = "dense",  data = datasets$dense_medium, writer = "sp_write_dense")
  )

  for (ic in input_configs) {
    # Write SPZ file
    spz_path <- tempfile(fileext = ".spz")
    if (ic$writer == "sp_write_dense") {
      sp_write_dense(as.matrix(ic$data), spz_path, include_transpose = TRUE)
    } else {
      sp_write(as(ic$data, "dgCMatrix"), spz_path, include_transpose = TRUE)
    }
    on.exit(unlink(spz_path), add = TRUE)

    for (backend in backends) {
      # In-memory benchmark
      bench_name_mem <- sprintf("nmf_%s_%s_inmem_k%d", backend, ic$name, k)
      cat(sprintf("  %s ... ", bench_name_mem))

      timing_mem <- bench_time(function()
        nmf(ic$data, k = k, tol = 1e-10, maxit = maxit,
            seed = 42, verbose = FALSE, resource = backend),
        n_reps = n_reps)

      results[[length(results) + 1]] <- list(
        name = bench_name_mem,
        backend = backend,
        input = ic$name,
        mode = "in_memory",
        rank = k,
        iterations = maxit,
        replicates = n_reps,
        mean_sec = timing_mem$mean_sec,
        sd_sec = timing_mem$sd_sec,
        min_sec = timing_mem$min_sec,
        max_sec = timing_mem$max_sec
      )
      cat(sprintf("%.3f +/- %.4f sec\n", timing_mem$mean_sec, timing_mem$sd_sec))

      # Streaming benchmark
      bench_name_str <- sprintf("nmf_%s_%s_streaming_k%d", backend, ic$name, k)
      cat(sprintf("  %s ... ", bench_name_str))

      timing_str <- bench_time(function()
        nmf(spz_path, k = k, tol = 1e-10, maxit = maxit,
            seed = 42, verbose = FALSE, resource = backend),
        n_reps = n_reps)

      results[[length(results) + 1]] <- list(
        name = bench_name_str,
        backend = backend,
        input = ic$name,
        mode = "streaming",
        rank = k,
        iterations = maxit,
        replicates = n_reps,
        mean_sec = timing_str$mean_sec,
        sd_sec = timing_str$sd_sec,
        min_sec = timing_str$min_sec,
        max_sec = timing_str$max_sec
      )
      cat(sprintf("%.3f +/- %.4f sec\n", timing_str$mean_sec, timing_str$sd_sec))
    }
  }

  results
}
