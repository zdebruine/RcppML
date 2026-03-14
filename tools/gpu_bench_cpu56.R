#!/usr/bin/env Rscript
# Re-run ALL 10 benchmark operations CPU-only with full thread count.
# GPU numbers stay the same — only CPU baselines change.
suppressMessages(library(RcppML))
suppressMessages(library(Matrix))
suppressMessages(library(Seurat))
suppressMessages(library(SeuratData))

threads <- as.integer(Sys.getenv("OMP_NUM_THREADS", "56"))
cat("OMP_NUM_THREADS:", threads, "\n")
cat("GPU available:", gpu_available(), "\n")

pf <- "tools/gpu_bench_cpu56_progress.txt"
lp <- function(...) { m <- paste0(...); cat(m, "\n"); cat(m, "\n", file = pf, append = TRUE) }

bench_cpu <- function(label, expr) {
  lp(sprintf("=== %s ===", label))
  gc(verbose = FALSE)
  t <- system.time(eval(expr))[["elapsed"]]
  lp(sprintf("  CPU: %.2f s", t))
  t
}

# ── Load hcabm40k ──
cat("Loading hcabm40k...\n")
data("hcabm40k"); hca <- UpdateSeuratObject(hcabm40k); rm(hcabm40k)
rna <- hca[["RNA"]]$counts; rm(hca); gc(verbose = FALSE)
gv <- Matrix::rowMeans(rna^2) - Matrix::rowMeans(rna)^2
top5k <- head(order(gv, decreasing = TRUE), 5000)
hca5k <- rna[top5k, ]; ls <- Matrix::colSums(hca5k)
hca_norm <- as(log1p(hca5k %*% Diagonal(x = 1e4 / ls)), "dgCMatrix")
set.seed(42); hca_10k <- as(hca_norm[, sample(ncol(hca_norm), 10000)], "dgCMatrix")
rm(rna, hca5k); gc(verbose = FALSE)
cat(sprintf("40K: %d x %d, 10K: %d x %d\n", nrow(hca_norm), ncol(hca_norm), nrow(hca_10k), ncol(hca_10k)))

# ── Load pbmc3k ──
data(pbmc3k, package = "RcppML")
tmp <- tempfile(fileext = ".spz")
writeBin(pbmc3k, tmp)
pbmc <- st_read(tmp)
pbmc_norm <- as(log1p(pbmc %*% Diagonal(x = 1e4 / Matrix::colSums(pbmc))), "dgCMatrix")
cat(sprintf("pbmc3k: %d x %d\n", nrow(pbmc_norm), ncol(pbmc_norm)))

cat("", file = pf)
results <- numeric(10)

# 1. NMF (k=20, 40K cells)
lp("--- 1 ---")
results[1] <- bench_cpu("NMF (k=20, 40K cells)",
  quote(nmf(hca_norm, k = 20, resource = "cpu", seed = 42, maxit = 20, tol = 0)))

# 2. NMF (k=64, 10K cells)
lp("--- 2 ---")
results[2] <- bench_cpu("NMF (k=64, 10K cells)",
  quote(nmf(hca_10k, k = 64, resource = "cpu", seed = 42, maxit = 20, tol = 0)))

# 3. CV NMF (k=64, 10K cells)
lp("--- 3 ---")
results[3] <- bench_cpu("CV NMF (k=64, 10K cells)",
  quote(nmf(hca_10k, k = 64, resource = "cpu", seed = 42, maxit = 20, tol = 0,
            test_fraction = 0.05)))

# 4. CV NMF (k=16, pbmc3k)
lp("--- 4 ---")
results[4] <- bench_cpu("CV NMF (k=16, pbmc3k)",
  quote(nmf(pbmc_norm, k = 16, resource = "cpu", seed = 42, maxit = 20, tol = 0,
            test_fraction = 0.05)))

# 5. KL-divergence NMF (k=16, pbmc3k)
lp("--- 5 ---")
results[5] <- bench_cpu("KL NMF (k=16, pbmc3k)",
  quote(nmf(pbmc_norm, k = 16, loss = "kl", resource = "cpu", seed = 42, maxit = 20, tol = 0)))

# 6. Sparse-mask NMF (k=20, 10K cells)
lp("--- 6 ---")
results[6] <- bench_cpu("Sparse-mask NMF (k=20, 10K cells)",
  quote(nmf(hca_10k, k = 20, mask_zeros = TRUE, resource = "cpu", seed = 42, maxit = 20, tol = 0)))

# 7. SVD Lanczos (k=10, 40K cells)
lp("--- 7 ---")
results[7] <- bench_cpu("SVD Lanczos (k=10, 40K cells)",
  quote(svd(hca_norm, k = 10, method = "lanczos", resource = "cpu", seed = 42)))

# 8. SVD randomized (k=10, 40K cells)
lp("--- 8 ---")
results[8] <- bench_cpu("SVD randomized (k=10, 40K cells)",
  quote(svd(hca_norm, k = 10, method = "randomized", resource = "cpu", seed = 42)))

# 9. SVD IRLBA (k=10, 40K cells)
lp("--- 9 ---")
results[9] <- bench_cpu("SVD IRLBA (k=10, 40K cells)",
  quote(svd(hca_norm, k = 10, method = "irlba", resource = "cpu", seed = 42)))

# 10. NMF (k=20, pbmc3k)
lp("--- 10 ---")
results[10] <- bench_cpu("NMF (k=20, pbmc3k)",
  quote(nmf(pbmc_norm, k = 20, resource = "cpu", seed = 42, maxit = 20, tol = 0)))

names(results) <- c(
  "NMF (k=20, 40K cells)", "NMF (k=64, 10K cells)",
  "CV NMF (k=64, 10K cells)", "CV NMF (k=16, pbmc3k)",
  "KL NMF (k=16, pbmc3k)", "Sparse-mask NMF (k=20, 10K cells)",
  "SVD Lanczos (k=10, 40K)", "SVD randomized (k=10, 40K)",
  "SVD IRLBA (k=10, 40K)", "NMF (k=20, pbmc3k)")

saveRDS(results, "tools/gpu_bench_cpu56_results.rds")
lp("=== ALL DONE ===")
cat("\n"); print(results)
