#!/usr/bin/env Rscript
# Render all vignettes and report status

vignettes <- c(
  "getting-started",
  "nmf-fundamentals",
  "svd-pca",
  "cross-validation",
  "distributions",
  "regularization",
  "clustering",
  "factor-graphs",
  "image-decomposition",
  "recommendation-systems",
  "streampress",
  "gpu-acceleration"
)

out_dir <- "/mnt/home/debruinz/RcppML-2/vignettes"
results <- data.frame(vignette = character(), status = character(), time_sec = numeric(),
                      stringsAsFactors = FALSE)

for (v in vignettes) {
  rmd <- file.path(out_dir, paste0(v, ".Rmd"))
  if (!file.exists(rmd)) {
    cat("SKIP:", v, "(file not found)\n")
    results <- rbind(results, data.frame(vignette = v, status = "SKIP", time_sec = 0))
    next
  }
  cat("Rendering:", v, "... ")
  t0 <- proc.time()[3]
  tryCatch({
    rmarkdown::render(rmd, output_dir = out_dir, quiet = TRUE)
    elapsed <- round(proc.time()[3] - t0, 1)
    cat("OK (", elapsed, "s)\n")
    results <- rbind(results, data.frame(vignette = v, status = "OK", time_sec = elapsed))
  }, error = function(e) {
    elapsed <- round(proc.time()[3] - t0, 1)
    cat("FAIL (", elapsed, "s):", conditionMessage(e), "\n")
    results <<- rbind(results, data.frame(vignette = v, status = paste("FAIL:", conditionMessage(e)),
                                           time_sec = elapsed))
  })
}

cat("\n=== RENDER SUMMARY ===\n")
print(results)
cat("\nTotal:", sum(results$time_sec), "seconds\n")
cat("Passed:", sum(results$status == "OK"), "/", nrow(results), "\n")
