#!/bin/bash
#SBATCH --job-name=revdep_check
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=4:00:00
#SBATCH --output=tools/revdep_check_%j.log
#SBATCH --error=tools/revdep_check_%j.log

# =============================================================================
# Reverse Dependency Check
# =============================================================================
# Tests the 6 known reverse dependencies against dev RcppML to detect breakage.
# Run AFTER tools/cran_prep.sbatch completes successfully.
# =============================================================================

set -euo pipefail
cd /mnt/home/debruinz/RcppML-2

module load r/4.5.2
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "=== $(date) === Reverse dependency check on $(hostname) ==="

# Install dev RcppML first
echo "--- Installing dev RcppML ---"
Rscript -e 'install.packages(".", repos = NULL, type = "source", INSTALL_opts = "--no-multiarch")'

# Check each reverse dependency
Rscript -e '
# Known reverse dependencies of RcppML 0.3.7
revdeps <- c("GeneNMF", "phytoclass", "flashier")

# Note: CARDspa, scater, miloR are Bioconductor packages with complex dep trees.
# We check the CRAN ones first; Bioc packages can be tested separately.

cat("=== Reverse Dependency Check ===\n")
results <- list()

for (pkg in revdeps) {
  cat(sprintf("\n--- Checking %s ---\n", pkg))

  # Try to install the reverse dep (skip if deps unavailable)
  tryCatch({
    if (!requireNamespace(pkg, quietly = TRUE)) {
      install.packages(pkg, dependencies = TRUE)
    }

    # Run its tests
    res <- tryCatch({
      tools::testInstalledPackage(pkg, types = "tests")
    }, error = function(e) {
      cat(sprintf("  TEST ERROR: %s\n", e$message))
      1L  # failure
    })

    results[[pkg]] <- if (identical(res, 0L)) "PASS" else "FAIL"
    cat(sprintf("  Result: %s\n", results[[pkg]]))

  }, error = function(e) {
    cat(sprintf("  INSTALL ERROR: %s\n", e$message))
    results[[pkg]] <<- "INSTALL_FAIL"
  })
}

cat("\n=== Summary ===\n")
for (pkg in names(results)) {
  cat(sprintf("  %-15s %s\n", pkg, results[[pkg]]))
}

n_fail <- sum(results != "PASS")
if (n_fail > 0) {
  cat(sprintf("\n%d package(s) failed. Review logs above.\n", n_fail))
  quit(status = 1)
} else {
  cat("\nAll reverse dependencies PASS.\n")
}
'

echo "=== $(date) === Reverse dependency check complete ==="
