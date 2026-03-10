#!/bin/bash
module load r/4.5.2
export OMP_NUM_THREADS=4
cd /mnt/home/debruinz/RcppML-2
mkdir -p logs
Rscript -e '
  files <- list.files("tests/testthat", pattern = "^test_.*\\.R$", full.names = FALSE)  
  files <- files[!grepl("gpu|factor_net", files)]
  # Extract test names from filenames
  test_names <- sub("^test_", "", sub("\\.R$", "", files))
  cat("Running", length(test_names), "test files (excluding gpu, factor_net)\n")
  res <- devtools::test(filter = paste(test_names, collapse = "|"))
  df <- as.data.frame(res)
  cat(sprintf("\nSUMMARY: %d passed, %d failed, %d skipped\n",
    sum(df$passed), sum(df$failed), sum(df$skipped)))
' > logs/cpu_test_fp32_v2.log 2>&1
echo "TESTS_DONE" >> logs/cpu_test_fp32_v2.log
