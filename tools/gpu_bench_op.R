#!/usr/bin/env Rscript
# Single-operation GPU benchmark. Pass op number as arg.
# Usage: Rscript tools/gpu_bench_op.R <op_num>
suppressMessages(library(RcppML))
suppressMessages(library(Matrix))

args <- commandArgs(trailingOnly = TRUE)
op_num <- as.integer(args[1])
cat("Running op", op_num, "\n")
cat("GPU available:", gpu_available(), "\n")
cat("OMP_NUM_THREADS:", Sys.getenv("OMP_NUM_THREADS", "8"), "\n")

bench1 <- function(label, cpu_expr, gpu_expr, reps = 1) {
  cat(sprintf("=== %s ===\n", label))
  tryCatch(eval(gpu_expr), error = function(e) cat("GPU warmup error:", e$message, "\n"))
  cat("  warmup done\n")
  cpu_times <- gpu_times <- numeric(reps)
  for (r in seq_len(reps)) {
    gc(verbose = FALSE)
    cpu_times[r] <- system.time(eval(cpu_expr))[["elapsed"]]
    cat(sprintf("  rep %d CPU: %.2f s\n", r, cpu_times[r]))
    gc(verbose = FALSE)
    gpu_times[r] <- system.time(eval(gpu_expr))[["elapsed"]]
    cat(sprintf("  rep %d GPU: %.2f s\n", r, gpu_times[r]))
  }
  cpu_med <- median(cpu_times); gpu_med <- median(gpu_times)
  speedup <- cpu_med / gpu_med
  cat(sprintf("  RESULT => CPU: %.2f s  GPU: %.2f s  Speedup: %.1fx\n", cpu_med, gpu_med, speedup))
  data.frame(Operation = label, CPU_s = cpu_med, GPU_s = gpu_med, Speedup = speedup, stringsAsFactors = FALSE)
}

# Load pbmc3k (always needed, fast)
data(pbmc3k, package = "RcppML")
tmp <- tempfile(fileext = ".spz"); writeBin(pbmc3k, tmp); pbmc <- st_read(tmp)
pbmc_norm <- log1p(pbmc %*% Diagonal(x = 1e4 / Matrix::colSums(pbmc)))
pbmc_norm <- as(pbmc_norm, "dgCMatrix")
cat(sprintf("pbmc3k: %d x %d\n", nrow(pbmc_norm), ncol(pbmc_norm)))

# Load hcabm40k if needed
if (op_num %in% c(7, 8, 9, 10, 11)) {
  suppressMessages(library(Seurat)); suppressMessages(library(SeuratData))
  cat("Loading hcabm40k...\n")
  data("hcabm40k"); hca <- UpdateSeuratObject(hcabm40k); rm(hcabm40k)
  rna <- hca[["RNA"]]$counts; rm(hca); gc(verbose = FALSE)
  gv <- Matrix::rowMeans(rna^2) - Matrix::rowMeans(rna)^2
  top5k <- head(order(gv, decreasing = TRUE), 5000)
  hca5k <- rna[top5k, ]; ls <- Matrix::colSums(hca5k)
  hca_norm <- as(log1p(hca5k %*% Diagonal(x = 1e4 / ls)), "dgCMatrix")
  set.seed(42); hca_10k <- as(hca_norm[, sample(ncol(hca_norm), 10000)], "dgCMatrix")
  rm(rna, hca5k); gc(verbose = FALSE)
  cat(sprintf("hcabm40k: %d x %d, 10K: %d x %d\n", nrow(hca_norm), ncol(hca_norm), nrow(hca_10k), ncol(hca_10k)))
}

result <- switch(as.character(op_num),
  # IRLS KL on pbmc3k (fast)
  "5" = bench1("KL-divergence NMF (k=16, pbmc3k)",
    quote(nmf(pbmc_norm, k = 16, loss = "kl", resource = "cpu", seed = 42, maxit = 50, tol = 0, irls_max_iter = 5)),
    quote(nmf(pbmc_norm, k = 16, loss = "kl", resource = "gpu", seed = 42, maxit = 50, tol = 0, irls_max_iter = 5)),
    reps = 2),

  # IRLS KL on pbmc3k (k=8 for variety)
  "6" = bench1("KL-divergence NMF (k=8, pbmc3k)",
    quote(nmf(pbmc_norm, k = 8, loss = "kl", resource = "cpu", seed = 42, maxit = 50, tol = 0, irls_max_iter = 5)),
    quote(nmf(pbmc_norm, k = 8, loss = "kl", resource = "gpu", seed = 42, maxit = 50, tol = 0, irls_max_iter = 5)),
    reps = 2),

  # mask_zeros NMF on 10K cells
  "7" = bench1("Sparse-mask NMF (k=20, 10K cells)",
    quote(nmf(hca_10k, k = 20, mask_zeros = TRUE, resource = "cpu", seed = 42, maxit = 20, tol = 0)),
    quote(nmf(hca_10k, k = 20, mask_zeros = TRUE, resource = "gpu", seed = 42, maxit = 20, tol = 0))),

  # SVD deflation on full 40K
  "8" = bench1("SVD deflation (k=30, 40K cells)",
    quote(svd(hca_norm, k = 30, method = "deflation", resource = "cpu", seed = 42)),
    quote(svd(hca_norm, k = 30, method = "deflation", resource = "gpu", seed = 42)),
    reps = 2),

  # SVD lanczos on full 40K
  "9" = bench1("SVD Lanczos (k=30, 40K cells)",
    quote(svd(hca_norm, k = 30, method = "lanczos", resource = "cpu", seed = 42)),
    quote(svd(hca_norm, k = 30, method = "lanczos", resource = "gpu", seed = 42)),
    reps = 2),

  # SVD krylov on full 40K
  "10" = bench1("SVD Krylov (k=30, 40K cells)",
    quote(svd(hca_norm, k = 30, method = "krylov", resource = "cpu", seed = 42)),
    quote(svd(hca_norm, k = 30, method = "krylov", resource = "gpu", seed = 42)),
    reps = 2),

  # Bipartition on full 40K
  "11" = bench1("Bipartition (40K cells)",
    quote(bipartition(hca_norm, seed = 42, resource = "cpu")),
    quote(bipartition(hca_norm, seed = 42, resource = "gpu")),
    reps = 2),

  # NMF pbmc3k k=20
  "12" = bench1("NMF (k=20, pbmc3k)",
    quote(nmf(pbmc_norm, k = 20, resource = "cpu", seed = 42, maxit = 50, tol = 0)),
    quote(nmf(pbmc_norm, k = 20, resource = "gpu", seed = 42, maxit = 50, tol = 0)),
    reps = 2),

  stop("Unknown op: ", op_num)
)

# Save single result
f <- sprintf("tools/gpu_bench_op%d.rds", op_num)
saveRDS(result, f)
cat(sprintf("\nSaved to %s\n", f))
