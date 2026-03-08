#!/bin/bash
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/home/debruinz/RcppML-2/tools/render_vignettes_%j.log
#SBATCH --job-name=render_vig

set -euo pipefail

module load r/4.5.2
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NOT_CRAN=true

cd /mnt/home/debruinz/RcppML-2

# Clean stale compiled objects and reinstall dev package
echo "=== Installing dev RcppML ==="
R CMD INSTALL --preclean --no-multiarch . 2>&1 | tail -5

# Fix Rcpp bug if needed
if [ -f tools/fix_rcpp_info_bug.sh ]; then
  bash tools/fix_rcpp_info_bug.sh 2>/dev/null || true
fi

# Create inst/doc if it doesn't exist
mkdir -p inst/doc

echo ""
echo "=== Rendering Vignettes ==="

Rscript -e '
library(rmarkdown)

vignette_dir <- "vignettes"
out_dir <- "inst/doc"
Sys.setenv(NOT_CRAN = "true")

rmd_files <- list.files(vignette_dir, pattern = "\\.Rmd$", full.names = TRUE)
cat(sprintf("Found %d vignettes to render\n\n", length(rmd_files)))

results <- data.frame(vignette = character(), status = character(), 
                      time_sec = numeric(), stringsAsFactors = FALSE)

for (rmd in rmd_files) {
  name <- tools::file_path_sans_ext(basename(rmd))
  cat(sprintf("--- Rendering: %s ---\n", name))
  
  t0 <- proc.time()["elapsed"]
  
  tryCatch({
    # Render to HTML
    out <- rmarkdown::render(
      rmd,
      output_dir = out_dir,
      output_format = rmarkdown::html_vignette(),
      quiet = TRUE,
      envir = new.env(parent = globalenv())
    )
    
    # Also copy the Rmd and create the .R tangled file for CRAN
    file.copy(rmd, file.path(out_dir, basename(rmd)), overwrite = TRUE)
    knitr::purl(rmd, output = file.path(out_dir, paste0(name, ".R")), quiet = TRUE)
    
    elapsed <- proc.time()["elapsed"] - t0
    cat(sprintf("  OK (%.1f sec)\n", elapsed))
    results <- rbind(results, data.frame(vignette = name, status = "OK", 
                                         time_sec = elapsed))
  }, error = function(e) {
    elapsed <- proc.time()["elapsed"] - t0
    cat(sprintf("  FAILED: %s (%.1f sec)\n", conditionMessage(e), elapsed))
    results <- rbind(results, data.frame(vignette = name, status = paste("FAILED:", conditionMessage(e)),
                                         time_sec = elapsed))
  })
  cat("\n")
}

cat("=== Summary ===\n")
print(results, right = FALSE, row.names = FALSE)
cat(sprintf("\nTotal: %d OK, %d failed\n", sum(results$status == "OK"), sum(results$status != "OK")))
cat(sprintf("Total time: %.1f sec\n", sum(results$time_sec)))

# List output files
cat("\nFiles in inst/doc:\n")
cat(paste(list.files(out_dir), collapse = "\n"), "\n")
'

echo ""
echo "=== Done ==="
