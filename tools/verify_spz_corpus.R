#!/usr/bin/env Rscript
# Verifies that all .spz files in the GEO reprocessed corpus are readable
# after the StreamPress rename.
#
# Usage: Rscript tools/verify_spz_corpus.R [corpus_dir] [n_sample]

library(RcppML)

args <- commandArgs(trailingOnly = TRUE)
corpus_dir <- if (length(args) >= 1) args[1] else {
  stop("Usage: Rscript tools/verify_spz_corpus.R <corpus_dir> [n_sample]")
}
n_sample <- if (length(args) >= 2) as.integer(args[2]) else 100L

files <- list.files(corpus_dir, pattern = "\\.spz$", full.names = TRUE, recursive = TRUE)
cat(sprintf("Found %d .spz files in %s\n", length(files), corpus_dir))

sample_files <- sample(files, min(n_sample, length(files)))
n_ok <- 0L
n_fail <- 0L
for (f in sample_files) {
  tryCatch({
    info <- st_info(f)
    stopifnot(info$version == 2L, info$rows > 0, info$cols > 0, info$nnz > 0)
    n_ok <- n_ok + 1L
  }, error = function(e) {
    cat(sprintf("FAIL: %s — %s\n", f, conditionMessage(e)))
    n_fail <<- n_fail + 1L
  })
}
cat(sprintf("Passed: %d / %d sampled files\n", n_ok, n_ok + n_fail))
if (n_fail > 0) stop(sprintf("%d files failed compatibility check!", n_fail))
cat("All sampled files are compatible.\n")
