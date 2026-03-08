#!/usr/bin/env Rscript
# full_profile.R — Comprehensive streaming vs in-memory profiling
#
# Uses pbmc3k (2700 cells) and synthetic scRNA-seq-like data (40k cells)
# to profile streaming NMF by section and compare to in-memory.

.libPaths(c("/tmp/rcppml_agent7", .libPaths()))
suppressPackageStartupMessages({
  library(RcppML)
  library(Matrix)
})

cat("=== Streaming NMF Profiling Investigation ===\n")
cat("Date:", format(Sys.time()), "\n")
cat("R:", R.version.string, "\n")
cat("OMP_NUM_THREADS:", Sys.getenv("OMP_NUM_THREADS", "unset"), "\n\n")

# ---- 1. Prepare datasets ----
datasets <- list()

# pbmc3k from SeuratData (real data, small)
tryCatch({
  suppressPackageStartupMessages(library(SeuratData))
  data("pbmc3k")
  mat <- pbmc3k@assays$RNA@counts
  mat <- mat[rowSums(mat) > 0, ]
  datasets$pbmc3k <- as(mat, "dgCMatrix")
  cat(sprintf("pbmc3k: %d x %d, nnz=%d (%.2f%% dense)\n",
      nrow(mat), ncol(mat), length(datasets$pbmc3k@x),
      100 * length(datasets$pbmc3k@x) / (as.double(nrow(mat)) * ncol(mat))))
  rm(pbmc3k, mat); gc(verbose=FALSE)
}, error = function(e) cat("Failed to load pbmc3k:", e$message, "\n"))

# Synthetic scRNA-seq-like: 10k genes x 20k cells, ~5% density
cat("Generating synthetic scRNA-seq (10k x 20k, ~5% dense)...\n")
set.seed(123)
m_syn <- 10000; n_syn <- 20000; density_syn <- 0.05
A_syn <- rsparsematrix(m_syn, n_syn, density = density_syn)
A_syn@x <- abs(A_syn@x) * 10
datasets$synth_20k <- A_syn
cat(sprintf("synth_20k: %d x %d, nnz=%d (%.2f%% dense)\n",
    nrow(A_syn), ncol(A_syn), length(A_syn@x),
    100 * length(A_syn@x) / (as.double(m_syn) * n_syn)))
rm(A_syn); gc(verbose=FALSE)

# Medium synthetic: 5k x 10k, ~3% density
set.seed(456)
m_med <- 5000; n_med <- 10000; density_med <- 0.03
A_med <- rsparsematrix(m_med, n_med, density = density_med)
A_med@x <- abs(A_med@x) * 5
datasets$synth_10k <- A_med
cat(sprintf("synth_10k: %d x %d, nnz=%d (%.2f%% dense)\n",
    nrow(A_med), ncol(A_med), length(A_med@x),
    100 * length(A_med@x) / (as.double(m_med) * n_med)))
rm(A_med); gc(verbose=FALSE)

# ---- 2. Write SPZ files ----
spz_dir <- "/tmp/stream_profile_data"
dir.create(spz_dir, recursive = TRUE, showWarnings = FALSE)

spz_files <- list()
for (nm in names(datasets)) {
  spz_path <- file.path(spz_dir, paste0(nm, ".spz"))
  if (!file.exists(spz_path)) {
    cat(sprintf("Writing %s to SPZ... ", nm))
    t0 <- proc.time()["elapsed"]
    sp_write(datasets[[nm]], spz_path, include_transpose = TRUE)
    cat(sprintf("%.1fs\n", proc.time()["elapsed"] - t0))
  } else {
    cat(sprintf("SPZ already exists: %s\n", spz_path))
  }
  spz_files[[nm]] <- spz_path
}

# ---- 3. Profiling functions ----
profile_streaming <- function(spz_path, k, maxit = 10, reps = 3) {
  options(RcppML.streaming = TRUE)
  on.exit(options(RcppML.streaming = FALSE))
  
  profiles <- list()
  times <- numeric(reps)
  
  for (r in 1:reps) {
    t0 <- proc.time()["elapsed"]
    res <- nmf(spz_path, k = k, maxit = maxit, tol = 1e-10,
               seed = 42 + r, profile = TRUE, verbose = FALSE)
    t1 <- proc.time()["elapsed"]
    times[r] <- t1 - t0
    if (!is.null(res@misc$profile)) profiles[[r]] <- res@misc$profile
  }
  
  avg_prof <- NULL
  if (length(profiles) > 0) {
    all_names <- unique(unlist(lapply(profiles, names)))
    avg_prof <- sapply(all_names, function(nm) {
      mean(sapply(profiles, function(p) ifelse(nm %in% names(p), p[nm], 0)))
    })
  }
  
  list(avg_profile = avg_prof, total_times = times,
       avg_total = mean(times), iterations = res@misc$iter, loss = res@misc$loss)
}

profile_inmemory <- function(mat, k, maxit = 10, reps = 3) {
  times <- numeric(reps)
  profiles <- list()
  
  for (r in 1:reps) {
    t0 <- proc.time()["elapsed"]
    res <- nmf(mat, k = k, maxit = maxit, tol = 1e-10,
               seed = 42 + r, profile = TRUE, verbose = FALSE)
    t1 <- proc.time()["elapsed"]
    times[r] <- t1 - t0
    if (!is.null(res@misc$profile)) profiles[[r]] <- res@misc$profile
  }
  
  avg_prof <- NULL
  if (length(profiles) > 0) {
    all_names <- unique(unlist(lapply(profiles, names)))
    avg_prof <- sapply(all_names, function(nm) {
      mean(sapply(profiles, function(p) ifelse(nm %in% names(p), p[nm], 0)))
    })
  }
  
  list(avg_profile = avg_prof, total_times = times,
       avg_total = mean(times), iterations = res@misc$iter, loss = res@misc$loss)
}

# ---- 4. Run Profiling ----
ranks <- c(8, 16, 32)
maxit <- 10
reps <- 3

results <- list()

for (ds_name in names(datasets)) {
  cat(sprintf("\n========== Dataset: %s ==========\n", ds_name))
  mat <- datasets[[ds_name]]
  spz_path <- spz_files[[ds_name]]
  
  for (k in ranks) {
    tag <- paste0(ds_name, "_k", k)
    cat(sprintf("\n--- k=%d, maxit=%d, reps=%d ---\n", k, maxit, reps))
    
    cat("  In-memory NMF... ")
    im <- profile_inmemory(mat, k = k, maxit = maxit, reps = reps)
    cat(sprintf("%.2fs (avg)\n", im$avg_total))
    
    cat("  Streaming NMF... ")
    st <- profile_streaming(spz_path, k = k, maxit = maxit, reps = reps)
    cat(sprintf("%.2fs (avg)\n", st$avg_total))
    
    results[[tag]] <- list(
      dataset = ds_name, k = k,
      inmemory = im, streaming = st,
      overhead_ratio = st$avg_total / im$avg_total
    )
    
    if (!is.null(st$avg_profile)) {
      cat("  Streaming profile (ms, cumulative over all iterations):\n")
      prof <- sort(st$avg_profile, decreasing = TRUE)
      total_ms <- if ("total_iter" %in% names(prof)) prof["total_iter"] else sum(prof)
      for (nm in names(prof)) {
        pct <- 100 * prof[nm] / total_ms
        cat(sprintf("    %-20s %10.1f ms  (%5.1f%%)\n", nm, prof[nm], pct))
      }
    }
    
    if (!is.null(im$avg_profile)) {
      cat("  In-memory profile (ms):\n")
      prof <- sort(im$avg_profile, decreasing = TRUE)
      total_ms <- sum(prof)
      for (nm in names(prof)) {
        pct <- 100 * prof[nm] / total_ms
        cat(sprintf("    %-20s %10.1f ms  (%5.1f%%)\n", nm, prof[nm], pct))
      }
    }
    
    cat(sprintf("  Overhead: streaming is %.2fx in-memory time\n",
                results[[tag]]$overhead_ratio))
  }
}

# ---- 5. Summary Table ----
cat("\n\n========== SUMMARY ==========\n")
cat(sprintf("%-15s %4s %10s %10s %10s %12s\n",
    "Dataset", "k", "InMem(s)", "Stream(s)", "Ratio", "IO_pct"))
cat(paste(rep("-", 70), collapse=""), "\n")

for (tag in names(results)) {
  r <- results[[tag]]
  io_ms <- 0
  total_ms <- 1
  if (!is.null(r$streaming$avg_profile)) {
    p <- r$streaming$avg_profile
    io_ms <- sum(p[grep("read", names(p))])
    total_ms <- if ("total_iter" %in% names(p)) p["total_iter"] else sum(p)
  }
  io_pct <- 100 * io_ms / total_ms
  cat(sprintf("%-15s %4d %10.2f %10.2f %10.2fx %10.1f%%\n",
      r$dataset, r$k, r$inmemory$avg_total, r$streaming$avg_total,
      r$overhead_ratio, io_pct))
}

save(results, file = file.path(spz_dir, "profile_results.RData"))
cat(sprintf("\nResults saved to %s/profile_results.RData\n", spz_dir))
cat("Done.\n")
