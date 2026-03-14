#!/usr/bin/env Rscript
# Render critical vignettes to HTML for visual review
# Run on CPU node with sufficient memory

library(rmarkdown)

vignettes_dir <- "/mnt/home/debruinz/RcppML-2/vignettes"
output_dir <- "/tmp/rendered_vignettes"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Critical vignettes to render (the ones with visual content changes)
to_render <- c(
  "factor-graphs.Rmd",
  "nmf-fundamentals.Rmd",
  "regularization.Rmd",
  "svd-pca.Rmd",
  "clustering.Rmd",
  "gpu-acceleration.Rmd"
)

for (vf in to_render) {
  cat("=== Rendering:", vf, "===\n")
  input_file <- file.path(vignettes_dir, vf)
  output_file <- file.path(output_dir, sub("\\.Rmd$", ".html", vf))
  tryCatch({
    rmarkdown::render(
      input_file,
      output_file = output_file,
      output_format = rmarkdown::html_document(self_contained = TRUE),
      quiet = FALSE,
      envir = new.env()
    )
    cat("  SUCCESS:", output_file, "\n")
  }, error = function(e) {
    cat("  ERROR:", e$message, "\n")
  })
  cat("\n")
}

cat("=== RENDERING COMPLETE ===\n")
cat("Files in", output_dir, ":\n")
print(list.files(output_dir, pattern = "\\.html$"))
