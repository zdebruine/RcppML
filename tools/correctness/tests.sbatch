#!/bin/bash
#SBATCH --job-name=rcppml-tests
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -uo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/jobs/rcppml/env.sh
export NOT_CRAN=true RCPPML_SKIP_GPU=true
cp -r $SRC/tests $TMPDIR/tests && cd $TMPDIR/tests/testthat
Rscript $WORK/jobs/rcppml/run_tests.R "" tests_cpu.csv 2>&1 | tail -400
echo "=== C++ unit tests ==="
ls $SRC/tests/cpp
cp -r $SRC $TMPDIR/full && ( cd $TMPDIR/full/tests/cpp && timeout 1800 make run 2>&1 | tail -40 )
echo TESTS_DONE
