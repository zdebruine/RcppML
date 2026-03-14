#!/usr/bin/env Rscript
# Quick-render a single vignette to check for runtime errors.
# Usage: Rscript tools/render_vignette.R vignettes/factor-graphs.Rmd
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Usage: Rscript render_vignette.R <vignette.Rmd>")

vignette <- args[1]
cat(sprintf("Rendering: %s\n", vignette))

outfile <- tempfile(fileext = ".html")
tryCatch({
  rmarkdown::render(vignette, output_file = outfile, quiet = FALSE,
                    envir = new.env(parent = globalenv()))
  cat(sprintf("\nSUCCESS: output at %s\n", outfile))
  cat(sprintf("Output size: %s bytes\n", file.info(outfile)$size))
}, error = function(e) {
  cat(sprintf("\nFAILED: %s\n", conditionMessage(e)))
  quit(status = 1)
})
