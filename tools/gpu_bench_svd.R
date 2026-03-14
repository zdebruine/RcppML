#!/usr/bin/env Rscript
# Fast SVD + bipartition benchmarks. 1 rep each, k=10 for SVD.
suppressMessages(library(RcppML))
suppressMessages(library(Matrix))
suppressMessages(library(Seurat))
suppressMessages(library(SeuratData))

pf <- "tools/gpu_bench_svd_progress.txt"
lp <- function(...) { m <- paste0(...); cat(m, "\n"); cat(m, "\n", file = pf, append = TRUE) }

bench1 <- function(label, cpu_expr, gpu_expr) {
  lp(sprintf("=== %s ===", label))
  tryCatch(eval(gpu_expr), error = function(e) lp("  GPU warmup error: ", e$message))
  lp("  warmup done")
  gc(verbose = FALSE); ct <- system.time(eval(cpu_expr))[["elapsed"]]
  lp(sprintf("  CPU: %.2f s", ct))
  gc(verbose = FALSE); gt <- system.time(eval(gpu_expr))[["elapsed"]]
  lp(sprintf("  GPU: %.2f s", gt))
  sp <- ct / gt
  lp(sprintf("  RESULT => CPU: %.2f s  GPU: %.2f s  Speedup: %.1fx", ct, gt, sp))
  data.frame(Operation = label, CPU_s = ct, GPU_s = gt, Speedup = sp, stringsAsFactors = FALSE)
}

cat("GPU available:", gpu_available(), "\n")

# Load hcabm40k
cat("Loading hcabm40k...\n")
data("hcabm40k"); hca <- UpdateSeuratObject(hcabm40k); rm(hcabm40k)
rna <- hca[["RNA"]]$counts; rm(hca); gc(verbose = FALSE)
gv <- Matrix::rowMeans(rna^2) - Matrix::rowMeans(rna)^2
top5k <- head(order(gv, decreasing = TRUE), 5000)
hca5k <- rna[top5k, ]; ls <- Matrix::colSums(hca5k)
hca_norm <- as(log1p(hca5k %*% Diagonal(x = 1e4 / ls)), "dgCMatrix")
rm(rna, hca5k); gc(verbose = FALSE)
cat(sprintf("hca_norm: %d x %d\n", nrow(hca_norm), ncol(hca_norm)))

cat("", file = pf)
results <- list()

# SVD Lanczos (fast, unconstrained)
lp("--- SVD Lanczos ---")
results[[1]] <- bench1("SVD Lanczos (k=10, 40K cells)",
  quote(svd(hca_norm, k = 10, method = "lanczos", resource = "cpu", seed = 42)),
  quote(svd(hca_norm, k = 10, method = "lanczos", resource = "gpu", seed = 42)))

# SVD Randomized
lp("--- SVD Randomized ---")
results[[2]] <- bench1("SVD randomized (k=10, 40K cells)",
  quote(svd(hca_norm, k = 10, method = "randomized", resource = "cpu", seed = 42)),
  quote(svd(hca_norm, k = 10, method = "randomized", resource = "gpu", seed = 42)))

# SVD IRLBA
lp("--- SVD IRLBA ---")
results[[3]] <- bench1("SVD IRLBA (k=10, 40K cells)",
  quote(svd(hca_norm, k = 10, method = "irlba", resource = "cpu", seed = 42)),
  quote(svd(hca_norm, k = 10, method = "irlba", resource = "gpu", seed = 42)))

# Bipartition
lp("--- Bipartition ---")
results[[4]] <- bench1("Bipartition (40K cells)",
  quote(bipartition(hca_norm, seed = 42, resource = "cpu")),
  quote(bipartition(hca_norm, seed = 42, resource = "gpu")))

all_res <- do.call(rbind, results)
saveRDS(all_res, "tools/gpu_bench_svd_results.rds")
lp("=== ALL DONE ===")
cat("\n")
print(all_res, row.names = FALSE)
