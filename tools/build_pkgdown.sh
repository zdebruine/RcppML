#!/bin/bash
# Build pkgdown site with timing per article
cd /mnt/home/debruinz/RcppML-2
module load r/4.5.2
export OMP_NUM_THREADS=22

echo "=== pkgdown build started at $(date) ==="
echo "Node: $(hostname), OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo ""

R --no-save <<'RSCRIPT'
options(warn = 1)
t0 <- proc.time()

cat("=== Building pkgdown site ===\n")
tryCatch({
  pkgdown::build_site(lazy = FALSE)
  cat("\n=== BUILD SUCCEEDED ===\n")
}, error = function(e) {
  cat("\n=== BUILD FAILED ===\n")
  cat("Error:", conditionMessage(e), "\n")
})

elapsed <- (proc.time() - t0)[3]
cat(sprintf("\nTotal elapsed: %.1f seconds (%.1f minutes)\n", elapsed, elapsed/60))
RSCRIPT

echo "=== pkgdown build finished at $(date) ==="
