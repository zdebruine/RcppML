#!/bin/bash
#SBATCH --job-name=rcppml-gpu
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:tesla_v100s:2
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -uo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/jobs/rcppml/env.sh
module load cuda/12.9 || exit 1
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv; nvcc --version | tail -2
BASE=$RW/rlib/4.5.2
export R_LIBS_USER=$RW/rlib/gpu-4.5.2; mkdir -p $R_LIBS_USER; export R_LIBS=$R_LIBS_USER:$BASE
cp -r $SRC $TMPDIR/RcppML && cd $TMPDIR/RcppML && chmod +x configure cleanup
echo "=== Issue #76: install with CUDA on PATH, no separate GPU build ==="
R CMD INSTALL --library=$R_LIBS_USER . > $RW/out/install_gpu_r.log 2>&1; echo "install exit: $?"; grep -A8 "Configuring" $RW/out/install_gpu_r.log
Rscript $WORK/jobs/rcppml/gpu_probe.R 2>&1 | tail -20
echo "=== GPU library build ==="
( cd src && make -f Makefile.gpu clean >/dev/null 2>&1; /usr/bin/time -v make -f Makefile.gpu > $RW/out/gpu_make.log 2>&1; echo "gpu make exit: $?"; make -f Makefile.gpu install 2>&1 | tail -3 )
echo "gpu build warnings: $(grep -ciE 'warning' $RW/out/gpu_make.log)"; grep -iE "error|Elapsed" $RW/out/gpu_make.log | head -20; ls -la src/*.so inst/lib 2>/dev/null
R CMD INSTALL --library=$R_LIBS_USER . > $RW/out/install_gpu_r2.log 2>&1; echo "reinstall exit: $?"
Rscript $WORK/jobs/rcppml/gpu_probe.R 2>&1 | tail -20
echo "=== Issue #75 ==="; Rscript $WORK/jobs/rcppml/r75.R 2>&1 | tail -40
echo "=== GPU tests ==="
export NOT_CRAN=true
cd tests/testthat
Rscript $WORK/jobs/rcppml/run_tests.R gpu tests_gpu.csv 2>&1 | tail -300
echo GPU_DONE
