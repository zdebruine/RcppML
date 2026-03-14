#!/usr/bin/env Rscript
# Run hcabm40k-based ops: 7 (mask_zeros), 8-10 (SVD), 11 (bipartition)
# All 1 rep for speed. Results saved to tools/gpu_bench_hca_ops.rds
suppressMessages(library(RcppML))
suppressMessages(library(Matrix))
suppressMessages(library(Seurat))
suppressMessages(library(SeuratData))

pf <- "tools/gpu_bench_hca_progress.txt"
lp <- function(...) { m <- paste0(...); cat(m, "\n"); cat(m, "\n", file = pf, append = TRUE) }

bench1 <- function(label, cpu_expr, gpu_expr, reps = 1) {
  lp(sprintf("=== %s ===", label))
  tryCatch(eval(gpu_expr), error = function(e) lp("  GPU warmup error: ", e$message))
  lp("  warmup done")
  ct <- gt <- numeric(reps)
  for (r in seq_len(reps)) {
    gc(verbose = FALSE); ct[r] <- system.time(eval(cpu_expr))[["elapsed"]]
    lp(sprintf("  rep %d CPU: %.2f s", r, ct[r]))
    gc(verbose = FALSE); gt[r] <- system.time(eval(gpu_expr))[["elapsed"]]
    lp(sprintf("  rep %d GPU: %.2f s", r, gt[r]))
  }
  cm <- median(ct); gm <- median(gt); sp <- cm / gm
  lp(sprintf("  RESULT => CPU: %.2f s  GPU: %.2f s  Speedup: %.1fx", cm, gm, sp))
  data.frame(Operation = label, CPU_s = cm, GPU_s = gm, Speedup = sp, stringsAsFactors = FALSE)
}

cat("GPU available:", gpu_available(), "\n")
cat("OMP:", Sys.getenv("OMP_NUM_THREADS", "8"), "\n")

# Load hcabm40k
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

cat("", file = pf)
results <- list()

# 7) mask_zeros NMF on 10K
lp("--- 7. mask_zeros NMF ---")
results[[1]] <- bench1("Sparse-mask NMF (k=20, 10K cells)",
  quote(nmf(hca_10k, k = 20, mask_zeros = TRUE, resource = "cpu", seed = 42, maxit = 20, tol = 0)),
  quote(nmf(hca_10k, k = 20, mask_zeros = TRUE, resource = "gpu", seed = 42, maxit = 20, tol = 0)))

# 8) SVD deflation
lp("--- 8. SVD deflation ---")
results[[2]] <- bench1("SVD deflation (k=30, 40K cells)",
  quote(svd(hca_norm, k = 30, method = "deflation", resource = "cpu", seed = 42)),
  quote(svd(hca_norm, k = 30, method = "deflation", resource = "gpu", seed = 42)))

# 9) SVD lanczos
lp("--- 9. SVD lanczos ---")
results[[3]] <- bench1("SVD Lanczos (k=30, 40K cells)",
  quote(svd(hca_norm, k = 30, method = "lanczos", resource = "cpu", seed = 42)),
  quote(svd(hca_norm, k = 30, method = "lanczos", resource = "gpu", seed = 42)))

# 10) SVD krylov
lp("--- 10. SVD krylov ---")
results[[4]] <- bench1("SVD Krylov (k=30, 40K cells)",
  quote(svd(hca_norm, k = 30, method = "krylov", resource = "cpu", seed = 42)),
  quote(svd(hca_norm, k = 30, method = "krylov", resource = "gpu", seed = 42)))

# 11) Bipartition
lp("--- 11. Bipartition ---")
results[[5]] <- bench1("Bipartition (40K cells)",
  quote(bipartition(hca_norm, seed = 42, resource = "cpu")),
  quote(bipartition(hca_norm, seed = 42, resource = "gpu")))

all_new <- do.call(rbind, results)
saveRDS(all_new, "tools/gpu_bench_hca_ops.rds")
lp("=== ALL HCA OPS DONE ===")
cat("\nSaved to tools/gpu_bench_hca_ops.rds\n")
print(all_new, row.names = FALSE)
