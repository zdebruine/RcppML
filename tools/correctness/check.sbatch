#!/bin/bash
#SBATCH --job-name=rcppml-check
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -uo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/jobs/rcppml/env.sh
cp -r $SRC $TMPDIR/RcppML && cd $TMPDIR && chmod +x RcppML/configure RcppML/cleanup
/usr/bin/time -v R CMD build RcppML 2>&1 | tail -60; echo "build exit: ${PIPESTATUS[0]}"
ls -la *.tar.gz
export _R_CHECK_FORCE_SUGGESTS_=false
/usr/bin/time -v R CMD check --as-cran --no-manual RcppML_*.tar.gz 2>&1 | tail -150; echo "check exit: ${PIPESTATUS[0]}"
rm -rf $RW/out/Rcheck; mkdir -p $RW/out/Rcheck
cp RcppML.Rcheck/00check.log RcppML.Rcheck/00install.out $RW/out/Rcheck/ 2>/dev/null
cp RcppML.Rcheck/tests/testthat.Rout* $RW/out/Rcheck/ 2>/dev/null
du -sh RcppML.Rcheck/RcppML/* 2>/dev/null
echo CHECK_DONE
