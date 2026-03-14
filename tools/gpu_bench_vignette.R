#!/usr/bin/env Rscript
# GPU benchmark script for the gpu-acceleration vignette
# Run on a GPU node: ssh g051 'cd /mnt/home/debruinz/RcppML-2 && module load r/4.5.2 cuda/12.8.1 && Rscript tools/gpu_bench_vignette.R'

suppressMessages(library(RcppML))
suppressMessages(library(Matrix))
suppressMessages(library(Seurat))
suppressMessages(library(SeuratData))

cat("GPU available:", gpu_available(), "\n")
print(gpu_info())

export_OMP <- Sys.getenv("OMP_NUM_THREADS", "8")
cat("OMP_NUM_THREADS:", export_OMP, "\n\n")

# ============================================================
# Helper: time an operation with warm-up
# ============================================================
progress_file <- "tools/gpu_bench_progress.txt"
log_progress <- function(...) {
  msg <- paste0(...)
  cat(msg, "\n")
  cat(msg, "\n", file = progress_file, append = TRUE)
}

bench <- function(label, cpu_expr, gpu_expr, reps = 3, warmup = TRUE) {
  log_progress(sprintf("=== %s ===", label))

  # Warm-up GPU (first call has overhead)
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

# ============================================================
# Load and prep hcabm40k
# ============================================================
cat("Loading hcabm40k...\n")
data("hcabm40k")
hca <- UpdateSeuratObject(hcabm40k)
rm(hcabm40k)
rna_counts <- hca[["RNA"]]$counts  # 17369 x 40000
rm(hca)
gc(verbose = FALSE)

# Top 5000 variable genes
gene_means <- Matrix::rowMeans(rna_counts)
gene_vars <- Matrix::rowMeans(rna_counts^2) - gene_means^2
top5k <- head(order(gene_vars, decreasing = TRUE), 5000)
hca5k <- rna_counts[top5k, ]

# Log-normalize
lib_sizes <- Matrix::colSums(hca5k)
hca_norm <- hca5k %*% Diagonal(x = 1e4 / lib_sizes)
hca_norm <- log1p(hca_norm)
hca_norm <- as(hca_norm, "dgCMatrix")

cat(sprintf("hcabm40k full: %d x %d, nnz = %d, density = %.3f\n",
            nrow(hca_norm), ncol(hca_norm), length(hca_norm@x),
            length(hca_norm@x) / (nrow(hca_norm) * ncol(hca_norm))))

# Smaller subset for expensive k=64 ops (10K cells)
set.seed(42)
cells_10k <- sample(ncol(hca_norm), 10000)
hca_10k <- hca_norm[, cells_10k]
hca_10k <- as(hca_10k, "dgCMatrix")
cat(sprintf("hcabm40k 10K subset: %d x %d, nnz = %d\n",
            nrow(hca_10k), ncol(hca_10k), length(hca_10k@x)))
rm(rna_counts, hca5k)
gc(verbose = FALSE)

# Also prep pbmc3k from RcppML's own data
cat("Loading pbmc3k...\n")
data(pbmc3k, package = "RcppML")
tmp <- tempfile(fileext = ".spz")
writeBin(pbmc3k, tmp)
pbmc <- st_read(tmp)
cat(sprintf("pbmc3k: %d x %d, nnz = %d\n", nrow(pbmc), ncol(pbmc), length(pbmc@x)))

# Log-normalize pbmc3k too
plib <- Matrix::colSums(pbmc)
pbmc_norm <- pbmc %*% Diagonal(x = 1e4 / plib)
pbmc_norm <- log1p(pbmc_norm)
pbmc_norm <- as(pbmc_norm, "dgCMatrix")

# Clear progress file
cat("", file = progress_file)

# ============================================================
# Benchmarks
# ============================================================
results <- list()

save_partial <- function() {
  done <- Filter(Negate(is.null), results)
  if (length(done) > 0) {
    partial <- do.call(rbind, done)
    saveRDS(partial, "tools/gpu_bench_vignette_results.rds")
  }
}

# 1) NMF: hcabm40k k=64 (10K cells, high rank)
log_progress("--- 1. NMF: hcabm40k k=64 ---")
results[[1]] <- bench("NMF hcabm40k k=64",
  quote(nmf(hca_10k, k = 64, resource = "cpu", seed = 42, maxit = 10, tol = 0)),
  quote(nmf(hca_10k, k = 64, resource = "gpu", seed = 42, maxit = 10, tol = 0)),
  reps = 1)
save_partial()

# 2) NMF: hcabm40k k=20 (full 40K cells, moderate rank)
log_progress("--- 2. NMF: hcabm40k k=20 ---")
results[[2]] <- bench("NMF hcabm40k k=20",
  quote(nmf(hca_norm, k = 20, resource = "cpu", seed = 42, maxit = 30, tol = 0)),
  quote(nmf(hca_norm, k = 20, resource = "gpu", seed = 42, maxit = 30, tol = 0)),
  reps = 2)
save_partial()

# 3) Cross-validation: hcabm40k k=64 (10K cells)
log_progress("--- 3. CV NMF: hcabm40k k=64 ---")
results[[3]] <- bench("CV NMF hcabm40k k=64",
  quote(nmf(hca_10k, k = 64, resource = "cpu", seed = 42, maxit = 10, tol = 0,
            test_fraction = 0.1, cv_seed = 42)),
  quote(nmf(hca_10k, k = 64, resource = "gpu", seed = 42, maxit = 10, tol = 0,
            test_fraction = 0.1, cv_seed = 42)),
  reps = 1)
save_partial()

# 4) Cross-validation: pbmc3k k=16
log_progress("--- 4. CV NMF: pbmc3k k=16 ---")
results[[4]] <- bench("CV NMF pbmc3k k=16",
  quote(nmf(pbmc_norm, k = 16, resource = "cpu", seed = 42, maxit = 50, tol = 0,
            test_fraction = 0.1, cv_seed = 42)),
  quote(nmf(pbmc_norm, k = 16, resource = "gpu", seed = 42, maxit = 50, tol = 0,
            test_fraction = 0.1, cv_seed = 42)),
  reps = 3)
save_partial()

# 5) IRLS NMF with KL loss: hcabm40k k=20
log_progress("--- 5. IRLS KL loss: hcabm40k k=20 ---")
results[[5]] <- bench("IRLS KL NMF hcabm40k k=20",
  quote(nmf(hca_norm, k = 20, loss = "kl", resource = "cpu", seed = 42, maxit = 20, tol = 0)),
  quote(nmf(hca_norm, k = 20, loss = "kl", resource = "gpu", seed = 42, maxit = 20, tol = 0)),
  reps = 2)
save_partial()

# 6) IRLS NMF with KL loss: pbmc3k k=16
log_progress("--- 6. IRLS KL loss: pbmc3k k=16 ---")
results[[6]] <- bench("IRLS KL NMF pbmc3k k=16",
  quote(nmf(pbmc_norm, k = 16, loss = "kl", resource = "cpu", seed = 42, maxit = 50, tol = 0)),
  quote(nmf(pbmc_norm, k = 16, loss = "kl", resource = "gpu", seed = 42, maxit = 50, tol = 0)),
  reps = 3)
save_partial()

# 7) mask_zeros (sparse=TRUE) NMF: hcabm40k k=20
log_progress("--- 7. Sparse (mask_zeros) NMF: hcabm40k k=20 ---")
results[[7]] <- bench("mask_zeros NMF hcabm40k k=20",
  quote(nmf(hca_norm, k = 20, mask_zeros = TRUE, resource = "cpu", seed = 42, maxit = 20, tol = 0)),
  quote(nmf(hca_norm, k = 20, mask_zeros = TRUE, resource = "gpu", seed = 42, maxit = 20, tol = 0)),
  reps = 2)
save_partial()

# 8) SVD: hcabm40k k=30 (deflation)
log_progress("--- 8. SVD deflation: hcabm40k k=30 ---")
results[[8]] <- bench("SVD deflation hcabm40k k=30",
  quote(svd(hca_norm, k = 30, method = "deflation", resource = "cpu", seed = 42)),
  quote(svd(hca_norm, k = 30, method = "deflation", resource = "gpu", seed = 42)),
  reps = 2)
save_partial()

# 9) SVD: hcabm40k k=30 (lanczos)
log_progress("--- 9. SVD lanczos: hcabm40k k=30 ---")
results[[9]] <- bench("SVD lanczos hcabm40k k=30",
  quote(svd(hca_norm, k = 30, method = "lanczos", resource = "cpu", seed = 42)),
  quote(svd(hca_norm, k = 30, method = "lanczos", resource = "gpu", seed = 42)),
  reps = 2)
save_partial()

# 10) SVD: hcabm40k k=30 (krylov)
log_progress("--- 10. SVD krylov: hcabm40k k=30 ---")
results[[10]] <- bench("SVD krylov hcabm40k k=30",
  quote(svd(hca_norm, k = 30, method = "krylov", resource = "cpu", seed = 42)),
  quote(svd(hca_norm, k = 30, method = "krylov", resource = "gpu", seed = 42)),
  reps = 2)
save_partial()

# 11) Bipartition: hcabm40k
log_progress("--- 11. Bipartition: hcabm40k ---")
results[[11]] <- bench("Bipartition hcabm40k",
  quote(bipartition(hca_norm, seed = 42, resource = "cpu")),
  quote(bipartition(hca_norm, seed = 42, resource = "gpu")),
  reps = 2)
save_partial()

# 12) NMF: pbmc3k k=20 (smaller dataset baseline)
log_progress("--- 12. NMF: pbmc3k k=20 ---")
results[[12]] <- bench("NMF pbmc3k k=20",
  quote(nmf(pbmc_norm, k = 20, resource = "cpu", seed = 42, maxit = 50, tol = 0)),
  quote(nmf(pbmc_norm, k = 20, resource = "gpu", seed = 42, maxit = 50, tol = 0)),
  reps = 3)
save_partial()

# ============================================================
# Summary
# ============================================================
all_results <- do.call(rbind, results)
all_results <- all_results[order(all_results$Speedup, decreasing = TRUE), ]

cat("\n============================\n")
cat("SUMMARY (sorted by speedup)\n")
cat("============================\n")
print(all_results, row.names = FALSE)

# Save for vignette use
saveRDS(all_results, "tools/gpu_bench_vignette_results.rds")
cat("\nResults saved to tools/gpu_bench_vignette_results.rds\n")
