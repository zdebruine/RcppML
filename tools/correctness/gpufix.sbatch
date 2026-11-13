#!/bin/bash
#SBATCH --job-name=rcppml-gpufix
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:tesla_v100s:2
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# GPU verification of the build in $RW/rlib/fix: builds the GPU library with install_gpu() from the
# installed package, then runs the GPU tests, the issue #75 reproduction and the correctness sweep on GPU.
set -uo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/jobs/rcppml/env.sh
module load cuda/12.9 || exit 1
export R_LIBS=$RW/rlib/fix:$RW/rlib/4.5.2 R_USER_CACHE_DIR=$RW/rcache SWEEP_TAG=fix NOT_CRAN=true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== gpu_status before install_gpu ==="
Rscript -e 'library(RcppML); cat("available:", gpu_available(), "\n"); print(gpu_status())' 2>&1 | tail -25
echo "=== install_gpu ==="
/usr/bin/time -v Rscript -e 'library(RcppML); print(install_gpu())' > $RW/out/fix_install_gpu.log 2>&1; echo "install_gpu exit: $?"
grep -E " error|Error|Elapsed" $RW/out/fix_install_gpu.log | head -40
echo "gpu build warnings: $(grep -c 'warning' $RW/out/fix_install_gpu.log)"
grep -B2 -A8 -E " error: | error:" $RW/out/fix_install_gpu.log | head -120
tail -6 $RW/out/fix_install_gpu.log
echo "=== gpu_status after ==="
Rscript -e 'library(RcppML); cat("available:", gpu_available(), "\n"); print(gpu_status())' 2>&1 | tail -25
Rscript -e 'library(RcppML); quit(status = if (gpu_available()) 0 else 1)' || { echo GPU_NOT_AVAILABLE; exit 1; }
echo "=== issue 75 ==="; Rscript $WORK/jobs/rcppml/r75.R 2>&1 | tail -20
echo "=== GPU vs CPU per iteration ==="; Rscript $WORK/jobs/rcppml/gpudiag.R 2>&1 | grep -E "it=|timing|rror" | cut -c1-200
echo "=== GPU tests ==="
cp -r $WORK/src/RcppML/fix/tests $TMPDIR/tests && cd $TMPDIR/tests/testthat && find . -name "*.R" -exec sed -i 's/\r$//' {} +
Rscript $WORK/jobs/rcppml/run_tests.R gpu tests_fix_gpu.csv 2>&1 | grep -vE "^Loading|^Attaching|masked|^ *svd$|^$" | tail -260
for s in A B C D E; do echo "######## SWEEP $s (gpu)"; Rscript $WORK/jobs/rcppml/sweep.R $s gpu 2>&1 | grep -E "^(FAIL|NOOP|ERR |WRON|PART|CRASH|Error)" | sed -E 's/ +\|/|/g' | cut -c1-250 | head -70; done
echo GPUFIX_DONE
