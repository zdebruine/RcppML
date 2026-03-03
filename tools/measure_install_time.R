#!/usr/bin/env Rscript
# Measure R CMD INSTALL compilation time for RcppML
#
# Run this on a COMPUTE NODE (not login node) with:
#   ssh <node>
#   module load r/4.5.2
#   cd /mnt/home/debruinz/RcppML-2
#   Rscript tools/measure_install_time.R
#
# Results are written to tools/install_timing_<timestamp>.txt

pkg_dir <- normalizePath(".")
stopifnot(file.exists(file.path(pkg_dir, "DESCRIPTION")))

# Optionally run devtools::document() first to regenerate RcppExports.cpp
# with the current include set (reflects changes to RcppML.h).
if (!requireNamespace("devtools", quietly = TRUE)) {
  message("devtools not available, skipping compileAttributes step")
} else {
  message("Regenerating RcppExports.cpp and RcppExports.R ...")
  Rcpp::compileAttributes(pkg_dir)
}

# Force clean build by removing existing .so and .o files
obj_files <- c(
  list.files(file.path(pkg_dir, "src"), pattern = "\\.(o|so)$", full.names = TRUE),
  list.files(file.path(pkg_dir, "src"), pattern = "\\.dll$", full.names = TRUE)
)
if (length(obj_files) > 0) {
  message(sprintf("Removing %d cached object files ...", length(obj_files)))
  file.remove(obj_files)
}

message("Starting timed R CMD INSTALL ...")
t_start <- proc.time()

exit_code <- system(
  paste("R CMD INSTALL --no-multiarch --no-test-load", shQuote(pkg_dir)),
  ignore.stdout = FALSE,
  ignore.stderr = FALSE
)

t_elapsed <- proc.time() - t_start
elapsed_sec <- t_elapsed["elapsed"]

result <- sprintf(
  "[%s] R CMD INSTALL: %.1f sec (%.1f min)  exit=%d\n",
  format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  elapsed_sec,
  elapsed_sec / 60,
  exit_code
)

message(result)

# Append to log
log_file <- file.path(pkg_dir, "tools", "install_timing.txt")
cat(result, file = log_file, append = TRUE)
message("Result appended to ", log_file)
