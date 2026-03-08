#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --gpus=2
#SBATCH --job-name=gpu-test
#SBATCH --output=tools/gpu_test_%j.log
#SBATCH --error=tools/gpu_test_%j.log

set -eo pipefail

module load r/4.5.2
module load cuda/12.8.1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-$(nproc)}

echo "=== Hostname: $(hostname) ==="
echo "=== Date: $(date) ==="

cd /mnt/home/debruinz/RcppML-2

echo ""
echo "========================================"
echo "=== Running GPU NMF tests ==="
echo "========================================"
Rscript -e '
library(testthat)
library(RcppML)
cat("GPU available:", gpu_available(), "\n")
test_file("tests/testthat/test_gpu_features.R")
' 2>&1

echo ""
echo "========================================"
echo "=== Running all GPU-related test files ==="
echo "========================================"
Rscript -e '
library(testthat)
library(RcppML)
gpu_test_files <- list.files("tests/testthat", pattern = "gpu", full.names = TRUE)
cat("GPU test files found:", length(gpu_test_files), "\n")
for (f in gpu_test_files) {
  cat("\n--- Running:", basename(f), "---\n")
  tryCatch(test_file(f), error = function(e) cat("ERROR:", conditionMessage(e), "\n"))
}
' 2>&1

echo ""
echo "=== Done: $(date) ==="
