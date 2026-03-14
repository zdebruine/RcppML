#!/usr/bin/env Rscript
# bench_large_spz.R —
# Cbind 100 GEO .spz files, write as (1) .spz with 8-MB chunks and
# (2) raw Eigen::SparseMatrix<float>-layout binary, then benchmark reads
# with detailed IO vs decode timing.
# Files are deleted at the end; matrices are freed after each use.

suppressPackageStartupMessages({ library(RcppML); library(Matrix) })

SPZ_PATH    <- "/tmp/bench_combined.spz"
RAW_PATH    <- "/tmp/bench_combined.csc"
CHUNK_BYTES <- 8e6      # 8 MB → ~52 cols/chunk at 38k rows

hr  <- function(lbl = "") {
  cat(sprintf("\n%s%s\n", if (nchar(lbl)) paste0("── ", lbl, " "), strrep("─", max(0, 65 - nchar(lbl)))))
}
fmb <- function(n)   sprintf("%.1f MB", n / 1e6)
fs  <- function(t)   sprintf("%.4f s",  t)

# ═══════════════════════════════════════════════════════════════════════════
hr("Phase 1: Load 100 SPZ files and cbind")
# ═══════════════════════════════════════════════════════════════════════════

base  <- "/mnt/projects/debruinz_project/cellarium/harmonized/data/homo_sapiens"
gses  <- list.dirs(base, recursive = FALSE)
files <- character(0)
for (g in gses) {
  hits  <- list.files(g, "^counts\\.spz$", recursive = TRUE, full.names = TRUE)
  files <- c(files, hits)
  if (length(files) >= 100) break
}
files     <- sort(files)[1:100]
spz_sizes <- file.size(files)
cat(sprintf("  Found %d files, total %s compressed on NFS\n",
            length(files), fmb(sum(spz_sizes))))

# Load all 100; 1 thread each (avoids intra-file OMP overhead for tiny files)
t_load <- proc.time()[3]
mats   <- lapply(files, st_read, threads = 1L)
t_load <- as.numeric(proc.time()[3] - t_load)

# Align to modal row count (all should match; this is just a safety filter)
nrows      <- vapply(mats, nrow, integer(1L))
modal_nrow <- as.integer(names(which.max(table(nrows))))
if (any(nrows != modal_nrow)) {
  kept  <- sum(nrows == modal_nrow)
  mats  <- mats[nrows == modal_nrow]
  files <- files[nrows == modal_nrow]
  cat(sprintf("  [filtered to %d files with nrow = %d]\n", kept, modal_nrow))
}

t_cbind  <- proc.time()[3]
combined <- do.call(cbind, mats)
t_cbind  <- as.numeric(proc.time()[3] - t_cbind)
rm(mats); gc(verbose = FALSE)  # free individual matrices immediately

total_nnz <- nnzero(combined)
ncols     <- ncol(combined)
nrows_r   <- nrow(combined)
mem_bytes <- as.numeric(object.size(combined))

cat(sprintf("  Serial load:  %.3f s (%d files, 1T each)\n", t_load, length(files)))
cat(sprintf("  cbind:        %.3f s\n", t_cbind))
cat(sprintf("  Combined:     %d rows × %d cols  |  %s NNZ  (%.3f%% density)\n",
            nrows_r, ncols,
            formatC(total_nnz, format = "d", big.mark = ","),
            100 * total_nnz / (as.numeric(nrows_r) * ncols)))
cat(sprintf("  In-memory:    %s  (R dgCMatrix, float64 values + int32 indices)\n",
            fmb(mem_bytes)))

# ═══════════════════════════════════════════════════════════════════════════
hr("Phase 2: Write .spz  (chunk_bytes = 8 MB)")
# ═══════════════════════════════════════════════════════════════════════════

bytes_per_col  <- nrows_r * 4L
exp_chunk_cols <- max(1L, as.integer(floor(CHUNK_BYTES / bytes_per_col)))
exp_n_chunks   <- ceiling(ncols / exp_chunk_cols)
cat(sprintf("  Predicted: chunk_cols ≈ %d  →  n_chunks ≈ %d  (chunks >> 40 cores)\n",
            exp_chunk_cols, exp_n_chunks))

t_write_spz <- proc.time()[3]
st_write(combined, SPZ_PATH,
         chunk_bytes       = CHUNK_BYTES,
         include_transpose = FALSE,
         verbose           = FALSE)
t_write_spz <- as.numeric(proc.time()[3] - t_write_spz)

spz_bytes <- file.size(SPZ_PATH)
info      <- st_info(SPZ_PATH)
cat(sprintf("  Written:    %s in %.3f s  (%.0f MB/s)\n",
            fmb(spz_bytes), t_write_spz, spz_bytes / 1e6 / t_write_spz))
cat(sprintf("  Confirmed:  chunk_cols = %d  →  n_chunks = %d\n",
            info$chunk_cols, info$num_chunks))
cat(sprintf("  Ratio vs in-memory fp64: %.2fx\n", mem_bytes / spz_bytes))

# ═══════════════════════════════════════════════════════════════════════════
hr("Phase 3: Write raw CSC binary  (Eigen::SparseMatrix<float> layout)")
# ═══════════════════════════════════════════════════════════════════════════
# Format:  [m:u32][n:u32][nnz:u32] | [p:(n+1)×i32] | [i:nnz×i32] | [x:nnz×f32]
# Exactly what Eigen would mmap() for a SparseMatrix<float> stored to disk.

t_write_raw <- proc.time()[3]
con <- file(RAW_PATH, "wb")
writeBin(as.integer(c(nrows_r, ncols, total_nnz)), con, size = 4L)
writeBin(as.integer(combined@p), con, size = 4L)   # colptr (n+1 × int32)
writeBin(as.integer(combined@i), con, size = 4L)   # rowidx  (0-based int32)
writeBin(as.numeric(combined@x), con, size = 4L)   # values  (truncated to float32)
close(con)
t_write_raw <- as.numeric(proc.time()[3] - t_write_raw)

raw_bytes <- file.size(RAW_PATH)
cat(sprintf("  Written:   %s in %.3f s  (%.0f MB/s)\n",
            fmb(raw_bytes), t_write_raw, raw_bytes / 1e6 / t_write_raw))
cat(sprintf("  Layout:    header 12 B  |  p=%s  |  i=%s  |  x=%s\n",
            fmb((ncols + 1L) * 4),
            fmb(total_nnz * 4),
            fmb(total_nnz * 4)))
cat(sprintf("  Raw/SPZ:   %.2fx  (raw is that much larger on disk)\n",
            raw_bytes / spz_bytes))

rm(combined); gc(verbose = FALSE)   # combined no longer needed

# ═══════════════════════════════════════════════════════════════════════════
hr("Phase 4: Warm page cache, then timed reads")
# ═══════════════════════════════════════════════════════════════════════════

# Use dd (reads sequentially, drops to /dev/null) for large-file IO timing;
# avoids readBin integer-size overflow on files > 2.1 GB.
sys_io <- function(path) {
  as.numeric(system.time(
    system(sprintf("dd if='%s' of=/dev/null bs=64M 2>/dev/null", path),
           ignore.stdout = TRUE)
  )["elapsed"])
}

cat("  Warming OS page cache for both files (dd to /dev/null) ...\n")
system(sprintf("dd if='%s' of=/dev/null bs=64M 2>/dev/null", SPZ_PATH))
system(sprintf("dd if='%s' of=/dev/null bs=64M 2>/dev/null", RAW_PATH))
gc(verbose = FALSE)

# ── 4a: IO-only baselines ──────────────────────────────────────────────────
cat("\n[ IO-only: bytes off disk, no decompression or Matrix construction ]\n")

t_io_spz <- sys_io(SPZ_PATH); gc(verbose = FALSE)
t_io_raw <- sys_io(RAW_PATH); gc(verbose = FALSE)

cat(sprintf("  SPZ IO only:     %s  (%s @ %.0f MB/s)\n",
            fs(t_io_spz), fmb(spz_bytes), spz_bytes / 1e6 / t_io_spz))
cat(sprintf("  Raw CSC IO only: %s  (%s @ %.0f MB/s)\n",
            fs(t_io_raw), fmb(raw_bytes), raw_bytes / 1e6 / t_io_raw))
cat(sprintf("  Raw has %.2fx more bytes → even at equal IO rates raw takes %.2fx longer\n",
            raw_bytes / spz_bytes, raw_bytes / spz_bytes))

# ── 4b: Raw CSC full read (IO + sparseMatrix construction) ─────────────────
cat("\n[ Raw CSC binary: full pipeline  (fread → data vectors → sparseMatrix()) ]\n")

# i and x are each ~554M elements × 4 bytes = 2.2 GB.
# readBin() n-argument must be < .Machine$integer.max (~2.1e9); read in two chunks.
read_large_int <- function(con, n) {
  CHUNK <- 500L * 1000L * 1000L   # 500 M elements safely within int range
  out   <- integer(n)
  done  <- 0L
  while (done < n) {
    blk        <- min(CHUNK, n - done)
    out[(done + 1L):(done + blk)] <- readBin(con, integer(), n = blk, size = 4L)
    done       <- done + blk
  }
  out
}
read_large_dbl <- function(con, n) {
  CHUNK <- 500L * 1000L * 1000L
  out   <- numeric(n)
  done  <- 0L
  while (done < n) {
    blk        <- min(CHUNK, n - done)
    out[(done + 1L):(done + blk)] <- readBin(con, numeric(), n = blk, size = 4L)
    done       <- done + blk
  }
  out
}

t0   <- proc.time()[3]
con  <- file(RAW_PATH, "rb")
hdr  <- readBin(con, integer(), n = 3L, size = 4L)   # m, n, nnz
p_r  <- readBin(con, integer(), n = hdr[2] + 1L, size = 4L)  # colptr (fits in int)
i_r  <- read_large_int(con, hdr[3])   # rowidx  (554 M entries, read in 2 chunks)
x_r  <- read_large_dbl(con, hdr[3])   # values  (554 M entries)
close(con)
t_raw_io  <- as.numeric(proc.time()[3] - t0)

mat_raw    <- sparseMatrix(i = i_r + 1L, p = p_r, x = x_r,
                           dims = c(hdr[1], hdr[2]), repr = "C")
t_raw_full <- as.numeric(proc.time()[3] - t0)
t_raw_cst  <- t_raw_full - t_raw_io
rm(mat_raw, p_r, i_r, x_r); gc(verbose = FALSE)

cat(sprintf("  IO portion:    %s  (%.0f MB/s over %s)\n",
            fs(t_raw_io), raw_bytes / 1e6 / t_raw_io, fmb(raw_bytes)))
cat(sprintf("  sparseMatrix() %s\n", fs(t_raw_cst)))
cat(sprintf("  TOTAL:         %s  ← this is the target to beat\n", fs(t_raw_full)))

# ── 4c: SPZ at 1 → 40 threads ──────────────────────────────────────────────
cat(sprintf("\n[ SPZ st_read() at varying thread counts  |  %d chunks ]\n",
            info$num_chunks))
cat(sprintf("  %-6s  %-10s  %-10s  %-14s  %-10s  %-10s  %s\n",
            "Thrds", "Total", "MB/s(cmp)", "Decode-only", "Scale/1T",
            "vs raw", "Decode MB/s"))
cat(sprintf("  %s\n", strrep("-", 80)))

t_spz_1t     <- NA_real_
t_decode_1t  <- NA_real_

for (thr in c(1L, 2L, 4L, 8L, 12L, 16L, 24L, 32L, 40L)) {
  if (thr > info$num_chunks) {
    cat(sprintf("  %-6d  [skipped — only %d chunks]\n", thr, info$num_chunks))
    next
  }
  gc(verbose = FALSE)
  t0    <- proc.time()[3]
  mat_s <- st_read(SPZ_PATH, threads = thr)
  t_spz <- as.numeric(proc.time()[3] - t0)
  rm(mat_s); gc(verbose = FALSE)   # free immediately to reclaim RAM

  if (is.na(t_spz_1t)) { t_spz_1t <- t_spz; t_decode_1t <- max(0, t_spz - t_io_spz) }

  t_dec   <- max(0, t_spz - t_io_spz)
  dec_bw  <- spz_bytes / 1e6 / max(t_dec, 1e-4)
  sc_1t   <- t_spz_1t / t_spz
  sc_raw  <- t_raw_full / t_spz
  win     <- if (t_spz < t_raw_full) "<<< SPZ wins" else "(raw faster)"

  cat(sprintf("  %-6d  %-10s  %-10.0f  %-14s  %-10.2f  %-10s  %.0f MB/s  %s\n",
              thr, fs(t_spz), spz_bytes / 1e6 / t_spz, fs(t_dec),
              sc_1t, sprintf("%.2fx", sc_raw), dec_bw, win))
}
cat(sprintf("  %-6s  %-10s  %-10.0f  %-14s  (IO-only floor — infinite-core limit)\n",
            "IO flr", fs(t_io_spz), spz_bytes / 1e6 / t_io_spz, "0.0000 s"))

# ═══════════════════════════════════════════════════════════════════════════
hr("Phase 5: Theoretical vs actual")
# ═══════════════════════════════════════════════════════════════════════════

dec_rate_1t <- spz_bytes / 1e6 / max(t_decode_1t, 1e-4)   # MB/s compressed, 1 thread
total_dec_s  <- spz_bytes / 1e6 / dec_rate_1t

cat(sprintf("  1T decode rate:  %.0f MB/s (compressed input)\n", dec_rate_1t))
cat(sprintf("  IO bandwidth:    %.0f MB/s (SPZ)  /  %.0f MB/s (raw)\n",
            spz_bytes / 1e6 / t_io_spz, raw_bytes / 1e6 / t_io_raw))
cat(sprintf("  Raw CSC total:   %s  =  %s IO + %s construct\n\n",
            fs(t_raw_full), fs(t_raw_io), fs(t_raw_cst)))

cat(sprintf("  %-8s  %-14s  %-12s  %s\n",
            "Cores", "Theoretical", "Budget vs raw", "Comment"))
cat(sprintf("  %s\n", strrep("-", 60)))
for (N in c(1, 2, 4, 8, 12, 16, 24, 32, 40)) {
  t_theory <- t_io_spz + total_dec_s / N
  margin   <- t_raw_full - t_theory
  flag     <- if (margin > 0) "  SPZ wins" else "  raw wins"
  cat(sprintf("  %-8d  %-14s  %+.4f s%s\n", N, fs(t_theory), margin, flag))
}
cat(sprintf("\n  Crossover at N = %.1f cores (perfect parallelism assumed)\n",
            total_dec_s / max(t_raw_full - t_io_spz, 1e-4)))

# ═══════════════════════════════════════════════════════════════════════════
hr("Cleanup")
# ═══════════════════════════════════════════════════════════════════════════
file.remove(SPZ_PATH); file.remove(RAW_PATH)
cat(sprintf("  Removed %s\n  Removed %s\n", SPZ_PATH, RAW_PATH))
cat("Done.\n")
