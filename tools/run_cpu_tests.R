devtools::load_all()
library(testthat)
gpu_files <- list.files("tests/testthat", pattern="test_gpu", full.names=TRUE)
all_files <- list.files("tests/testthat", pattern="^test_.*\\.R$", full.names=TRUE)
cpu_files <- setdiff(all_files, gpu_files)
cat("CPU test files:", length(cpu_files), "\n")
results <- list()
for (f in cpu_files) {
  res <- tryCatch(test_file(f, reporter="summary"), error=function(e) e)
  results[[basename(f)]] <- res
}
totP <- sum(sapply(results, function(r) if(inherits(r,"error")) 0 else sum(as.data.frame(r)$passed)))
totF <- sum(sapply(results, function(r) if(inherits(r,"error")) 0 else sum(as.data.frame(r)$failed)))
totS <- sum(sapply(results, function(r) if(inherits(r,"error")) 0 else sum(as.data.frame(r)$skipped)))
totE <- sum(sapply(results, function(r) inherits(r,"error")))
cat(sprintf("\nCPU TOTAL: PASS=%d FAIL=%d SKIP=%d ERROR=%d\n", totP, totF, totS, totE))
