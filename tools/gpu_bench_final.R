#!/usr/bin/env Rscript
# Final GPU benchmark: remaining ops 5-12 with optimized parameters
# Uses 10K subset for expensive IRLS/mask_zeros, reduced maxit for IRLS
suppressMessages(library(RcppML))
suppressMessages(library(Matrix))
suppressMessages(library(Seurat))
suppressMessages(library(SeuratData))

cat("GPU available:", gpu_available(), "\n")
cat("OMP_NUM_THREADS:", Sys.getenv("OMP_NUM_THREADS", "8"), "\n\n")

pf <- "tools/gpu_bench_final_progress.txt"
log_p <- function(...) { msg <- paste0(...); cat(msg, "\n"); cat(msg, "\n", file = pf, append = TRUE) }

bench <- function(label, cpu_expr, gpu_expr, reps = 1) {
  log_p(sprintf("=== %s ===", label))
  tryCatch(eval(gpu_expr), error = function(e) log_p("  GPU warmup error: ", e$message))
  log_p("  warmup done")
  cpu_times <- gpu_times <- numeric(reps)
  for (r in seq_len(reps)) {
    gc(verbose = FALSE)
    cpu_times[r] <- system.time(eval(cpu_expr))[["elapsed"]]
    log_p(sprintf("  rep %d CPU: %.2f s", r, cpu_times[r]))
    gc(verbose = FALSE)
    gpu_times[r] <- system.time(eval(gpu_expr))[["elapsed"]]
    log_p(sprintf("  rep %d GPU: %.2f s", r, gpu_times[r]))
  }
  cpu_med <- median(cpu_times); gpu_med <- median(gpu_times)
  speedup <- cpu_med / gpu_med
  log_p(sprintf("  RESULT => CPU: %.2f s  GPU: %.2f s  Speedup: %.1fx", cpu_med, gpu_med, speedup))
  data.frame(Operation = label, CPU_s = cpu_med, GPU_s = gpu_med, Speedup = speedup, stringsAsFactors = FALSE)
}

# ─── Data prep ───
cat("Loading hcabm40k...\n")
data("hcabm40k")
hca <- UpdateSeuratObject(hcabm40k); rm(hcabm40k)
rna_counts <- hca[["RNA"]]$counts; rm(hca); gc(verbose = FALSE)

gene_means <- Matrix::rowMeans(rna_counts)
gene_vars <- Matrix::rowMeans(rna_counts^2) - gene_means^2
top5k <- head(order(gene_vars, decreasing = TRUE), 5000)
hca5k <- rna_counts[top5k, ]
lib_sizes <- Matrix::colSums(hca5k)
hca_norm <- log1p(hca5k %*% Diagonal(x = 1e4 / lib_sizes))
hca_norm <- as(hca_norm, "dgCMatrix")
cat(sprintf("hcabm40k full: %d x %d, nnz = %d\n", nrow(hca_norm), ncol(hca_norm), length(hca_norm@x)))

set.seed(42)
hca_10k <- as(hca_norm[, sample(ncol(hca_norm), 10000)], "dgCMatrix")
cat(sprintf("hcabm40k 10K: %d x %d, nnz = %d\n", nrow(hca_10k), ncol(hca_10k), length(hca_10k@x)))
rm(rna_counts, hca5k); gc(verbose = FALSE)

cat("Loading pbmc3k...\n")
data(pbmc3k, package = "RcppML")
tmp <- tempfile(fileext = ".spz"); writeBin(pbmc3k, tmp)
pbmc <- st_read(tmp)
pbmc_norm <- log1p(pbmc %*% Diagonal(x = 1e4 / Matrix::colSums(pbmc)))
pbmc_norm <- as(pbmc_norm, "dgCMatrix")
cat(sprintf("pbmc3k: %d x %d, nnz = %d\n", nrow(pbmc_norm), ncol(pbmc_norm), length(pbmc_norm@x)))

cat("", file = pf)
results <- list()

# 5) IRLS KL on 10K, 10 outer iterations, irls_max_iter=5, 1 rep
log_p("--- 5. IRLS KL: hcabm40k 10K k=20 ---")
results[[1]] <- bench("IRLS KL NMF (k=20, 10K cells)",
  quote(nmf(hca_10k, k = 20, loss = "kl", resource = "cpu", seed = 42, maxit = 10, tol = 0, irls_max_iter = 5)),
  quote(nmf(hca_10k, k = 20, loss = "kl", resource = "gpu", seed = 42, maxit = 10, tol = 0, irls_max_iter = 5)))

# 6) IRLS KL on pbmc3k, irls_max_iter=5, 2 reps
log_p("--- 6. IRLS KL: pbmc3k k=16 ---")
results[[2]] <- bench("IRLS KL NMF (k=16, pbmc3k)",
  quote(nmf(pbmc_norm, k = 16, loss = "kl", resource = "cpu", seed = 42, maxit = 50, tol = 0, irls_max_iter = 5)),
  quote(nmf(pbmc_norm, k = 16, loss = "kl", resource = "gpu", seed = 42, maxit = 50, tol = 0, irls_max_iter = 5)),
  reps = 2)

# 7) mask_zeros NMF on 10K, 1 rep
log_p("--- 7. mask_zeros NMF: hcabm40k 10K k=20 ---")
results[[3]] <- bench("Sparse-mask NMF (k=20, 10K cells)",
  quote(nmf(hca_10k, k = 20, mask_zeros = TRUE, resource = "cpu", seed = 42, maxit = 20, tol = 0)),
  quote(nmf(hca_10k, k = 20, mask_zeros = TRUE, resource = "gpu", seed = 42, maxit = 20, tol = 0)))

# 8-10) SVD on full 40K, 2 reps each
for (method in c("deflation", "lanczos", "krylov")) {
  idx <- switch(method, deflation = 4, lanczos = 5, krylov = 6)
  log_p(sprintf("--- %d. SVD %s: hcabm40k k=30 ---", idx + 4, method))
  results[[idx]] <- bench(sprintf("SVD %s (k=30, 40K cells)", method),
    bquote(svd(hca_norm, k = 30, method = .(method), resource = "cpu", seed = 42)),
    bquote(svd(hca_norm, k = 30, method = .(method), resource = "gpu", seed = 42)),
    reps = 2)
}

# 11) Bipartition on full 40K
log_p("--- 11. Bipartition: hcabm40k ---")
results[[7]] <- bench("Bipartition (40K cells)",
  quote(bipartition(hca_norm, seed = 42, resource = "cpu")),
  quote(bipartition(hca_norm, seed = 42, resource = "gpu")),
  reps = 2)

# 12) NMF pbmc3k k=20
log_p("--- 12. NMF: pbmc3k k=20 ---")
results[[8]] <- bench("NMF (k=20, pbmc3k)",
  quote(nmf(pbmc_norm, k = 20, resource = "cpu", seed = 42, maxit = 50, tol = 0)),
  quote(nmf(pbmc_norm, k = 20, resource = "gpu", seed = 42, maxit = 50, tol = 0)),
  reps = 2)

# ─── Combine with ops 1-4 ───
new_results <- do.call(rbind, results)
prev <- readRDS("tools/gpu_bench_vignette_results.rds")
all_results <- rbind(prev, new_results)
all_results <- all_results[order(all_results$Speedup, decreasing = TRUE), ]

cat("\n============================\n")
cat("FULL SUMMARY\n")
cat("============================\n")
print(all_results, row.names = FALSE)

saveRDS(all_results, "tools/gpu_bench_all_results.rds")
log_p("=== ALL DONE ===")
cat("\nSaved to tools/gpu_bench_all_results.rds\n")
