#!/usr/bin/env Rscript
# Validate that all modified vignettes can have R code extracted (purl)
# and that the R code parses cleanly.

cat("=== Vignette Validation ===\n\n")

library(knitr)

vignettes <- c(
  "vignettes/factor-graphs.Rmd",
  "vignettes/gpu-acceleration.Rmd",
  "vignettes/nmf-fundamentals.Rmd",
  "vignettes/regularization.Rmd",
  "vignettes/svd-pca.Rmd",
  "vignettes/clustering.Rmd",
  "vignettes/getting-started.Rmd"
)

results <- list()

for (v in vignettes) {
  cat(sprintf("--- %s ---\n", v))
  
  # Check file exists
  if (!file.exists(v)) {
    cat("  SKIP: file not found\n")
    results[[v]] <- "not found"
    next
  }
  
  # Try purl (extract R code)
  tmp <- tempfile(fileext = ".R")
  tryCatch({
    knitr::purl(v, output = tmp, quiet = TRUE)
    code <- readLines(tmp)
    cat(sprintf("  PURL OK: %d lines of R code extracted\n", length(code)))
    
    # Try parsing
    tryCatch({
      parse(file = tmp)
      cat("  PARSE OK: R code parses without errors\n")
      results[[v]] <- "OK"
    }, error = function(e) {
      cat(sprintf("  PARSE ERROR: %s\n", conditionMessage(e)))
      results[[v]] <<- paste("parse error:", conditionMessage(e))
    })
  }, error = function(e) {
    cat(sprintf("  PURL ERROR: %s\n", conditionMessage(e)))
    results[[v]] <- paste("purl error:", conditionMessage(e))
  })
  
  unlink(tmp)
  cat("\n")
}

cat("=== Summary ===\n")
for (v in names(results)) {
  cat(sprintf("  %-40s %s\n", basename(v), results[[v]]))
}

n_ok <- sum(results == "OK")
cat(sprintf("\n%d/%d vignettes validated successfully.\n", n_ok, length(results)))

if (n_ok < length(results)) {
  cat("\nFAILED vignettes need attention!\n")
  quit(status = 1)
} else {
  cat("\nAll vignettes OK!\n")
}
