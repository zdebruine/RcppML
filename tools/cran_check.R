#!/usr/bin/env Rscript
# Build and run R CMD check --as-cran
# Usage: Rscript tools/cran_check.R

setwd("/mnt/home/debruinz/RcppML-2")
Sys.setenv(`_R_CHECK_FORCE_SUGGESTS_` = "false")

cat("=== Building tarball ===\n")
unlink("RcppML_1.0.1.tar.gz")
system2("R", c("CMD", "build", "."), stdout = "/tmp/build.log", stderr = "/tmp/build.log")
if (!file.exists("RcppML_1.0.1.tar.gz")) {
  cat("BUILD FAILED. Last 20 lines of log:\n")
  writeLines(tail(readLines("/tmp/build.log"), 20))
  quit(status = 1)
}
cat("Build succeeded.\n")

cat("\n=== Running R CMD check --as-cran ===\n")
system2("R", c("CMD", "check", "--as-cran", "--no-manual", "RcppML_1.0.1.tar.gz"),
        stdout = "/tmp/check.log", stderr = "/tmp/check.log")

# Parse results
log <- readLines("/tmp/check.log")
cat("\n=== Summary (ERROR/WARNING/NOTE lines) ===\n")
idx <- grep("ERROR|WARNING|NOTE|Status:", log, ignore.case = FALSE)
writeLines(log[idx])

cat("\n=== Full check status ===\n")
status_lines <- grep("^Status:", log)
if (length(status_lines) > 0) writeLines(log[status_lines])

cat("\n=== Last 10 lines ===\n")
writeLines(tail(log, 10))
