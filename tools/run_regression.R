.libPaths(c("/tmp/debruinz_agent3_rlib", .libPaths()))
library(RcppML)
library(testthat)
library(Matrix)

test_files <- c("test_streaming.R", "test_svd.R", "test_nmf.R",
                "test_bipartiteMatch.R", "test_dclust_expanded.R",
                "test_streaming_svd_cv.R")

for (tf in test_files) {
  path <- file.path("tests/testthat", tf)
  if (!file.exists(path)) {
    cat("--- SKIP", tf, "(not found) ---\n")
    next
  }
  cat("---", tf, "---\n")
  r <- tryCatch(
    testthat::test_file(path, reporter = "silent"),
    error = function(e) {
      cat("  ERROR:", conditionMessage(e), "\n")
      NULL
    }
  )
  if (!is.null(r)) {
    df <- as.data.frame(r)
    cat("  Pass:", sum(df$passed), " Fail:", sum(df$failed), "\n")
  }
}
