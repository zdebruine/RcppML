#!/bin/bash
#SBATCH --job-name=rcppml-verify
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-6
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Verifies the build in $RW/rlib/fix: tasks 0-5 run the correctness sweep sections A-F, task 6 the test suite.
set -uo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/jobs/rcppml/env.sh
export R_LIBS=$RW/rlib/fix:$RW/rlib/4.5.2
export SWEEP_TAG=fix
SECS=(A B C D E F)
if [ $SLURM_ARRAY_TASK_ID -le 5 ]; then
  Rscript $WORK/jobs/rcppml/sweep.R ${SECS[$SLURM_ARRAY_TASK_ID]} cpu 2>&1
else
  export NOT_CRAN=true
  cp -r $WORK/src/RcppML/fix/tests $TMPDIR/tests && cd $TMPDIR/tests/testthat
  find . -name "*.R" -exec sed -i 's/\r$//' {} +
  Rscript $WORK/jobs/rcppml/run_tests.R "" tests_fix_cpu.csv 2>&1 | tail -600
fi
echo VERIFY_DONE
