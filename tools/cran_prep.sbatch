#!/bin/bash
#SBATCH --job-name=cran_prep
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=tools/cran_prep_%j.log
#SBATCH --error=tools/cran_prep_%j.log

# =============================================================================
# CRAN Submission Preparation
# =============================================================================
# This script:
#   1. Fixes Rcpp bug + installs the dev package
#   2. Runs devtools::document() + re-fixes Rcpp bug
#   3. Reinstalls with updated NAMESPACE/man
#   4. Verifies data("pbmc3k") loads from spz
#   5. Runs R CMD check --as-cran
# =============================================================================

set -euo pipefail
cd /mnt/home/debruinz/RcppML-2

module load r/4.5.2
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "=== $(date) === Starting CRAN prep on $(hostname) ==="
echo "Threads: $OMP_NUM_THREADS"

# ---- Step 0: Clean up data/pbmc3k.R (incompatible with LazyData:true) ----
# The .R loader can't work during LazyData DB creation at install time.
# We regenerate pbmc3k.rda from the spz file in Step 1b.
if [ -f data/pbmc3k.R ]; then
  echo "--- Step 0: Removing data/pbmc3k.R (will regenerate as .rda) ---"
  rm -f data/pbmc3k.R
fi

# ---- Step 1: compileAttributes + fix bug + install ----
echo "--- Step 1: Compile attributes + fix + install ---"
# Clean stale object files to force full recompilation
rm -f src/*.o src/*.so
Rscript -e 'Rcpp::compileAttributes(".")'
bash tools/fix_rcpp_info_bug.sh
R CMD INSTALL --no-multiarch . 2>&1 | tail -20

# ---- Step 1b: Generate pbmc3k.rda from spz ----
echo "--- Step 1b: Generate pbmc3k.rda ---"
Rscript -e '
library(RcppML)
spz_path <- "inst/extdata/pbmc3k.spz"
meta_path <- "inst/extdata/pbmc3k_meta.rds"
mat <- sp_read(spz_path)
meta <- readRDS(meta_path)
rownames(mat) <- meta$rownames
colnames(mat) <- meta$colnames
attr(mat, "cell_type") <- meta$cell_type
pbmc3k <- mat
save(pbmc3k, file = "data/pbmc3k.rda", compress = "xz")
cat(sprintf("Generated data/pbmc3k.rda: %d x %d, nnz=%d, %.1f KB\n",
            nrow(pbmc3k), ncol(pbmc3k), length(pbmc3k@x),
            file.size("data/pbmc3k.rda") / 1024))
'

# ---- Step 2: devtools::document() ----
echo "--- Step 2: devtools::document() ---"
Rscript -e 'devtools::document()' 2>&1 | tail -10
bash tools/fix_rcpp_info_bug.sh

# ---- Step 3: Reinstall with updated docs ----
echo "--- Step 3: Reinstall ---"
R CMD INSTALL --no-multiarch . 2>&1 | tail -20

# ---- Step 4: Verify data("pbmc3k") ----
echo "--- Step 4: Verify pbmc3k loading ---"
Rscript -e '
library(RcppML)
data("pbmc3k")
cat(sprintf("pbmc3k: %d x %d, nnz=%d\n", nrow(pbmc3k), ncol(pbmc3k), length(pbmc3k@x)))
stopifnot(inherits(pbmc3k, "dgCMatrix"))
stopifnot(nrow(pbmc3k) == 13714)
stopifnot(ncol(pbmc3k) == 2700)
stopifnot(!is.null(rownames(pbmc3k)))
stopifnot(!is.null(attr(pbmc3k, "cell_type")))
cat("pbmc3k loading: PASS\n")

# Verify other datasets
data("aml"); stopifnot(nrow(aml) == 824)
data("hawaiibirds"); stopifnot(nrow(hawaiibirds) == 183)
cat("All datasets: PASS\n")
'

# ---- Step 5: Check data/ size ----
echo "--- Step 5: Data size check ---"
du -sh data/
DATA_SIZE=$(du -sb data/ | awk '{print $1}')
echo "data/ directory: $DATA_SIZE bytes"
if [ "$DATA_SIZE" -gt 5242880 ]; then
  echo "WARNING: data/ exceeds 5 MB ($DATA_SIZE bytes)"
else
  echo "OK: data/ is under 5 MB"
fi

# ---- Step 6: Run tests ----
echo "--- Step 6: Quick test suite ---"
Rscript -e '
library(testthat)
library(RcppML)
test_results <- test_dir("tests/testthat", reporter = "summary")
cat(sprintf("Tests: %d passed, %d failed, %d skipped\n",
    sum(as.data.frame(test_results)$passed),
    sum(as.data.frame(test_results)$failed),
    sum(as.data.frame(test_results)$skipped)))
if (any(as.data.frame(test_results)$failed > 0)) {
  stop("Test failures detected!")
}
'

# ---- Step 7: R CMD build + check --as-cran ----
echo "--- Step 7: R CMD check --as-cran ---"
BUILDDIR=$(mktemp -d)
cd "$BUILDDIR"
R CMD build /mnt/home/debruinz/RcppML-2 --no-build-vignettes --resave-data=best
TARBALL=$(ls -t RcppML_*.tar.gz | head -1)
echo "Built: $TARBALL ($(du -h $TARBALL | awk '{print $1}'))"

export _R_CHECK_FORCE_SUGGESTS_=false
R CMD check --as-cran --no-manual --no-vignettes "$TARBALL" 2>&1 | tee /mnt/home/debruinz/RcppML-2/tools/rcmd_check.log

# ---- Summary ----
echo ""
echo "=== $(date) === CRAN prep complete ==="
echo "Check results in: $BUILDDIR/RcppML.Rcheck/"

# Count and display issues
echo ""
CHECK_LOG="$BUILDDIR/RcppML.Rcheck/00check.log"
if [ -f "$CHECK_LOG" ]; then
  echo "=== Check Summary ==="
  grep -cE "^.*WARNING" "$CHECK_LOG" && echo " WARNING(s)" || echo "0 WARNINGs"
  grep -cE "^.*ERROR" "$CHECK_LOG" && echo " ERROR(s)" || echo "0 ERRORs"
  grep -cE "^.*NOTE" "$CHECK_LOG" && echo " NOTE(s)" || echo "0 NOTEs"
  echo ""
  echo "=== Issues found ==="
  grep -E "WARNING|ERROR|NOTE" "$CHECK_LOG" || echo "None"
fi

# Copy log
cp "$CHECK_LOG" /mnt/home/debruinz/RcppML-2/tools/00check.log 2>/dev/null || true
