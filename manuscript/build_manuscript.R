#!/usr/bin/env Rscript
# =============================================================================
# Build manuscript: generate figures, tables, and compile LaTeX to PDF
# Run from project root: Rscript manuscript/build_manuscript.R
# =============================================================================

setwd("/mnt/home/debruinz/RcppML-2")

cat("=== Step 1: Generate figures and tables ===\n")
source("manuscript/nn2net_plots.R")

cat("\n=== Step 2: Compile LaTeX ===\n")

# Check for tinytex
if (!requireNamespace("tinytex", quietly = TRUE)) {
  stop("tinytex package not available")
}

# Install tinytex if not installed
if (!tinytex::is_tinytex()) {
  cat("Installing TinyTeX...\n")
  tinytex::install_tinytex()
}

# Install needed LaTeX packages
tryCatch({
  tinytex::tlmgr_install(c("booktabs", "multirow", "natbib",
                             "algorithms", "algorithmicx", "microtype",
                             "caption", "subcaption", "hyperref",
                             "xcolor", "geometry"))
}, error = function(e) {
  cat("Note: some LaTeX packages may already be installed:", e$message, "\n")
})

# Compile
cat("Compiling nn2net_generalization.tex...\n")
setwd("manuscript")
tryCatch({
  tinytex::pdflatex("nn2net_generalization.tex")
  cat("\n=== SUCCESS: manuscript/nn2net_generalization.pdf generated ===\n")
}, error = function(e) {
  cat("LaTeX compilation error:", e$message, "\n")
  cat("Trying with system pdflatex...\n")
  system("pdflatex -interaction=nonstopmode nn2net_generalization.tex")
  system("pdflatex -interaction=nonstopmode nn2net_generalization.tex")  # 2nd pass for refs
  if (file.exists("nn2net_generalization.pdf")) {
    cat("\n=== SUCCESS: manuscript/nn2net_generalization.pdf generated ===\n")
  } else {
    cat("\n=== FAILED: Check .log file for details ===\n")
  }
})
