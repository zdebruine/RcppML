#!/usr/bin/env Rscript
# GPU build script - run on a GPU node
cat("=== GPU Build ===\n")
cat("Hostname:", system("hostname", intern=TRUE), "\n")

# Build GPU binary
ret <- system("cd /mnt/home/debruinz/RcppML-2/src && make -f Makefile.gpu clean 2>/dev/null && make -f Makefile.gpu install 2>&1 | tail -3")
if (ret != 0) {
  cat("GPU BUILD FAILED with code", ret, "\n")
  # Show errors
  system("cd /mnt/home/debruinz/RcppML-2/src && make -f Makefile.gpu install 2>&1 | grep -i error | head -20")
} else {
  cat("GPU BUILD SUCCESS\n")
}

# Install R package (CPU side)
cat("\n=== R Package Install ===\n")
ret2 <- system("cd /mnt/home/debruinz/RcppML-2 && R CMD INSTALL . 2>&1 | tail -3")
if (ret2 != 0) {
  cat("R INSTALL FAILED\n")
} else {
  cat("R INSTALL SUCCESS\n")
}
