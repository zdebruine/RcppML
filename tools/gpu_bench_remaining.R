#!/usr/bin/env Rscript
# Benchmark remaining ops (5-12) using 10K subset for expensive IRLS/mask_zeros
suppressMessages(library(RcppML))
suppressMessages(library(Matrix))
suppressMessages(library(Seurat))
suppressMessages(library(SeuratData))

cat("GPU available:", gpu_available(), "\n")
export_OMP <- Sys.getenv("OMP_NUM_THREADS", "8")
cat("OMP_NUM_THREADS:", export_OMP, "\n\n")

progress_file <- "tools/gpu_bench_remaining_progress.txt"
log_progress <- function(...) {
  msg <- paste0(...)
  cat(msg, "\n")
  cat(msg, "\n", file = progress_file, append = TRUE)
}

bench <- function(label, cpu_expr, gpu_expr, reps = 2, warmup = TRUE) {
  log_progress(sprintf("=== %s ===", label))
  if (warmup) {
    tryCatch(eval(gpu_expr), error = function(e) log_progress("  GPU warmup error: ", e$message))
  }
  log_progress("  warmup done")
  cpu_times <- numeric(reps)
  gpu_times <- numeric(reps)
  for (r in seq_len(reps)) {
    gc(verbose = FALSE)
    cpu_times[r] <- system.time(eval(cpu_expr))[["elapsed"]]
    log_progress(sprintf("  rep %d CPU: %.2f s", r, cpu_times[r]))
    gc(verbose = FALSE)
    gpu_times[r] <- system.time(eval(gpu_expr))[["elapsed"]]
    log_progress(sprintf("  rep %d GPU: %.2f s", r, gpu_times[r]))
  }
  cpu_med <- median(cpu_times)
  gpu_med <- median(gpu_times)
  speedup <- cpu_med / gpu_med
  log_progress(sprintf("  RESULT => CPU: %.2f s  GPU: %.2f s  Speedup: %.1fx", cpu_med, gpu_med, speedup))
  data.frame(Operation = label, CPU_s = cpu_med, GPU_s = gpu_med, Speedup = speedup,
             stringsAsFactors = FALSE)
}

# Load data
cat("Loading hcabm40k...\n")
data("hcabm40k")
hca <- UpdateSeuratObject(hcabm40k)
rm(hcabm40k)
rna_counts <- hca[["RNA"]]$counts
rm(hca); gc(verbose = FALSE)

gene_means <- Matrix::rowMeans(rna_counts)
gene_vars <- Matrix::rowMeans(rna_counts^2) - gene_means^2
top5k <- head(order(gene_vars, decreasing = TRUE), 5000)
hca5k <- rna_counts[top5k, ]
lib_sizes <- Matrix::colSums(hca5k)
hca_norm <- hca5k %*% Diagonal(x = 1e4 / lib_sizes)
hca_norm <- log1p(hca_norm)
hca_norm <- as(hca_norm, "dgCMatrix")
cat(sprintf("hcabm40k full: %d x %d, nnz = %d\n", nrow(hca_norm), ncol(hca_norm), length(hca_norm@x)))

set.seed(42)
hca_10k <- hca_norm[, sample(ncol(hca_norm), 10000)]
hca_10k <- as(hca_10k, "dgCMatrix")
cat(sprintf("hcabm40k 10K: %d x %d, nnz = %d\n", nrow(hca_10k), ncol(hca_10k), length(hca_10k@x)))
rm(rna_counts, hca5k); gc(verbose = FALSE)

cat("Loading pbmc3k...\n")
data(pbmc3k, package = "RcppML")
tmp <- tempfile(fileext = ".spz")
writeBin(pbmc3k, tmp)
pbmc <- st_read(tmp)
plib <- Matrix::colSums(pbmc)
pbmc_norm <- pbmc %*% Diagonal(x = 1e4 / plib)
pbmc_norm <- log1p(pbmc_norm)
pbmc_norm <- as(pbmc_norm, "dgCMatrix")
cat(sprintf("pbmc3k: %d x %d, nnz = %d\n", nrow(pbmc_norm), ncol(pbmc_norm), length(pbmc_norm@x)))

cat("", file = progress_file)
results <- list()

# 5) IRLS KL on 10K subset (not full 40K — too slow on CPU)
log_progress("--- 5. IRLS KL loss: hcabm40k 10K k=20 ---")
results[[1]] <- bench("IRLS KL NMF hcabm40k (10K) k=20",
  quote(nmf(hca_10k, k = 20, loss = "kl", resource = "cpu", seed = 42, maxit = 20, tol = 0)),
  quote(nmf(hca_10k, k = 20, loss = "kl", resource = "gpu", seed = 42, maxit = 20, tol = 0)),
  reps = 1)

# 6) IRLS KL on pbmc3k
log_progress("--- 6. IRLS KL loss: pbmc3k k=16 ---")
results[[2]] <- bench("IRLS KL NMF pbmc3k k=16",
  quote(nmf(pbmc_norm, k = 16, loss = "kl", resource = "cpu", seed = 42, maxit = 50, tol = 0)),
  quote(nmf(pbmc_norm, k = 16, loss = "kl", resource = "gpu", seed = 42, maxit = 50, tol = 0)),
  reps = 2)

# 7) mask_zeros NMF on 10K subset
log_progress("--- 7. Sparse (mask_zeros) NMF: hcabm40k 10K k=20 ---")
results[[3]] <- bench("mask_zeros NMF hcabm40k (10K) k=20",
  quote(nmf(hca_10k, k = 20, mask_zeros = TRUE, resource = "cpu", seed = 42, maxit = 20, tol = 0)),
  quote(nmf(hca_10k, k = 20, mask_zeros = TRUE, resource = "gpu", seed = 42, maxit = 20, tol = 0)),
  reps = 1)

# 8) SVD deflation on full 40K
log_progress("--- 8. SVD deflation: hcabm40k k=30 ---")
results[[4]] <- bench("SVD deflation hcabm40k k=30",
  quote(svd(hca_norm, k = 30, method = "deflation", resource = "cpu", seed = 42)),
  quote(svd(hca_norm, k = 30, method = "deflation", resource = "gpu", seed = 42)),
  reps = 2)

# 9) SVD lanczos on full 40K
log_progress("--- 9. SVD lanczos: hcabm40k k=30 ---")
results[[5]] <- bench("SVD lanczos hcabm40k k=30",
  quote(svd(hca_norm, k = 30, method = "lanczos", resource = "cpu", seed = 42)),
  quote(svd(hca_norm, k = 30, method = "lanczos", resource = "gpu", seed = 42)),
  reps = 2)

# 10) SVD krylov on full 40K
log_progress("--- 10. SVD krylov: hcabm40k k=30 ---")
results[[6]] <- bench("SVD krylov hcabm40k k=30",
  quote(svd(hca_norm, k = 30, method = "krylov", resource = "cpu", seed = 42)),
  quote(svd(hca_norm, k = 30, method = "krylov", resource = "gpu", seed = 42)),
  reps = 2)

# 11) Bipartition on full 40K
log_progress("--- 11. Bipartition: hcabm40k ---")
results[[7]] <- bench("Bipartition hcabm40k",
  quote(bipartition(hca_norm, seed = 42, resource = "cpu")),
  quote(bipartition(hca_norm, seed = 42, resource = "gpu")),
  reps = 2)

# 12) NMF pbmc3k k=20
log_progress("--- 12. NMF: pbmc3k k=20 ---")
results[[8]] <- bench("NMF pbmc3k k=20",
  quote(nmf(pbmc_norm, k = 20, resource = "cpu", seed = 42, maxit = 50, tol = 0)),
  quote(nmf(pbmc_norm, k = 20, resource = "gpu", seed = 42, maxit = 50, tol = 0)),
  reps = 2)

# Summary
all_new <- do.call(rbind, results)

# Load ops 1-4 from previous run
prev <- readRDS("tools/gpu_bench_vignette_results.rds")
cat("\n=== Previous results (ops 1-4) ===\n")
print(prev)

all_results <- rbind(prev, all_new)
all_results <- all_results[order(all_results$Speedup, decreasing = TRUE), ]
cat("\n============================\n")
cat("FULL SUMMARY (sorted by speedup)\n")
cat("============================\n")
print(all_results, row.names = FALSE)

saveRDS(all_results, "tools/gpu_bench_all_results.rds")
cat("\nAll results saved to tools/gpu_bench_all_results.rds\n")
