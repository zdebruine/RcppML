#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --gpus=2
#SBATCH --job-name=gpu-build-test
#SBATCH --output=tools/gpu_build_test_%j.log
#SBATCH --error=tools/gpu_build_test_%j.log

set -eo pipefail

module load r/4.5.2
module load cuda/12.8.1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-$(nproc)}

echo "=== Hostname: $(hostname) ==="
echo "=== Date: $(date) ==="
echo "=== CUDA: $(nvcc --version | tail -1) ==="

cd /mnt/home/debruinz/RcppML-2

# --- Step 1: Build GPU library ---
echo ""
echo "========================================"
echo "=== Building GPU library ==="
echo "========================================"
cd src
make -f Makefile.gpu clean
make -f Makefile.gpu 2>&1
echo "GPU build exit code: $?"
cd ..

# --- Step 2: Install R package ---
echo ""
echo "========================================"
echo "=== Installing R package ==="
echo "========================================"
R CMD INSTALL . --no-test-load 2>&1
echo "R install exit code: $?"

# Copy fresh GPU .so to installed package location
RLIB=$(Rscript --no-save --no-restore -e 'cat(find.package("RcppML"))')
mkdir -p "$RLIB/lib"
cp -v src/RcppML_gpu.so "$RLIB/lib/"
echo "GPU .so installed to: $RLIB/lib/"

# --- Step 3: Run GPU tests ---
echo ""
echo "========================================"
echo "=== Running GPU NMF tests ==="
echo "========================================"
Rscript -e '
library(testthat)
library(RcppML)
cat("GPU available:", gpu_available(), "\n")
if (gpu_available()) {
  info <- gpu_info()
  cat("GPU:", info$name, "\n")
}
test_file("tests/testthat/test_gpu_features.R")
' 2>&1

echo ""
echo "========================================"
echo "=== Running all GPU tests ==="
echo "========================================"
Rscript -e '
library(testthat)
library(RcppML)
gpu_test_files <- list.files("tests/testthat", pattern = "gpu", full.names = TRUE)
cat("GPU test files:", gpu_test_files, "\n")
for (f in gpu_test_files) {
  cat("\n--- Running:", f, "---\n")
  tryCatch(test_file(f), error = function(e) cat("ERROR:", conditionMessage(e), "\n"))
}
' 2>&1

echo ""
echo "=== Done: $(date) ==="
