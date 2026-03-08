#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH --job-name=rcppml-build
#SBATCH --output=tools/build_test_%j.log

set -e

module load r/4.5.2
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /mnt/home/debruinz/RcppML-2

echo "=== Hostname: $(hostname) ==="
echo "=== Regenerating Rcpp exports ==="
Rscript -e 'Rcpp::compileAttributes(".")'
bash tools/fix_rcpp_info_bug.sh

echo "=== Installing package ==="
R CMD INSTALL --no-multiarch . 2>&1

echo "=== Smoke test: all solver/init combos ==="
Rscript -e '
library(RcppML)
library(Matrix)
data(pbmc3k)
A <- pbmc3k[1:500, 1:200]  # small subset for quick test

cat("--- CD + random ---\n")
m1 <- nmf(A, k = 5, maxit = 20, tol = 1e-10, seed = 1, solver = "cd", init = "random")
cat(sprintf("  Loss: %.6f, iter: %d\n", m1@misc$loss, m1@misc$iter))

cat("--- CD + lanczos ---\n")
m2 <- nmf(A, k = 5, maxit = 20, tol = 1e-10, seed = 1, solver = "cd", init = "lanczos")
cat(sprintf("  Loss: %.6f, iter: %d\n", m2@misc$loss, m2@misc$iter))

cat("--- Cholesky + random ---\n")
m3 <- nmf(A, k = 5, maxit = 20, tol = 1e-10, seed = 1, solver = "cholesky", init = "random")
cat(sprintf("  Loss: %.6f, iter: %d\n", m3@misc$loss, m3@misc$iter))

cat("--- Cholesky + lanczos ---\n")
m4 <- nmf(A, k = 5, maxit = 20, tol = 1e-10, seed = 1, solver = "cholesky", init = "lanczos")
cat(sprintf("  Loss: %.6f, iter: %d\n", m4@misc$loss, m4@misc$iter))

cat("--- Hierarchical Cholesky + random ---\n")
m5 <- nmf(A, k = 5, maxit = 20, tol = 1e-10, seed = 1, solver = "hierarchical_cholesky", init = "random")
cat(sprintf("  Loss: %.6f, iter: %d\n", m5@misc$loss, m5@misc$iter))

cat("--- Hierarchical Cholesky + lanczos ---\n")
m6 <- nmf(A, k = 5, maxit = 20, tol = 1e-10, seed = 1, solver = "hierarchical_cholesky", init = "lanczos")
cat(sprintf("  Loss: %.6f, iter: %d\n", m6@misc$loss, m6@misc$iter))

cat("\n=== All 6 combinations passed ===\n")
'

echo "=== BUILD AND TEST COMPLETE ==="
