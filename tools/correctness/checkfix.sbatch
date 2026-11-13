#!/bin/bash
#SBATCH --job-name=rcppml-checkfix
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# R CMD check --as-cran of the working tree under CRAN-like limits (two threads, NOT_CRAN unset for the check).
set -uo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/jobs/rcppml/env.sh
export OMP_NUM_THREADS=4 MAKEFLAGS=-j4
FIX=$WORK/src/RcppML/fix
rm -rf $TMPDIR/RcppML && cp -r $FIX $TMPDIR/RcppML && cd $TMPDIR
find RcppML -type f \( -name "*.sh" -o -name "*.R" -o -name "*.Rmd" -o -name "*.cpp" -o -name "*.h" -o -name "*.hpp" -o -name "*.cu" -o -name "*.cuh" -o -name "Makevars*" -o -name "Makefile*" -o -name DESCRIPTION -o -name NAMESPACE -o -name "*.Rd" -o -name "*.md" -o -name "*.yaml" -o -name ".Rbuildignore" -o -name "CITATION" -o -name "COPYRIGHTS" \) -exec sed -i 's/\r$//' {} +
if [ -d $RW/out/generated ]; then cp $RW/out/generated/NAMESPACE RcppML/; rm -f RcppML/man/*.Rd; cp $RW/out/generated/man/*.Rd RcppML/man/; cp $RW/out/generated/R/RcppExports.R RcppML/R/; cp $RW/out/generated/src/RcppExports.cpp RcppML/src/; fi
echo "=== build (NOT_CRAN=true so vignettes carry output) ==="
NOT_CRAN=true /usr/bin/time -v R CMD build RcppML > $RW/out/fix_build.log 2>&1; echo "build exit: $?"; grep -E "Elapsed|WARNING|ERROR|Error" $RW/out/fix_build.log | head; ls -la *.tar.gz ls -la *.tar.gz
echo '--- largest files in the tarball ---'; tar tvzf RcppML_*.tar.gz | grep -v '^d' | sort -k4 -n -r | head -18 | cut -c1-150
echo "=== check --as-cran, two threads ==="
unset NOT_CRAN; export _R_CHECK_LIMIT_CORES_=true OMP_THREAD_LIMIT=2 _R_CHECK_FORCE_SUGGESTS_=false
/usr/bin/time -v R CMD check --as-cran --no-manual RcppML_*.tar.gz > $RW/out/fix_check.log 2>&1; echo "check exit: $?"
grep -E "^Status|Elapsed" $RW/out/fix_check.log
grep -nE "\.\.\. (WARNING|NOTE|ERROR)|^\* checking (tests|examples|re-building|whether package|installed package size|CRAN incoming)|\[[0-9]+s/[0-9]+s\]" $RW/out/fix_check.log | cut -c1-160
grep -n -A14 -E "\.\.\. (WARNING|NOTE|ERROR)$" $RW/out/fix_check.log | cut -c1-200 | head -150
rm -rf $RW/out/Rcheck_fix; mkdir -p $RW/out/Rcheck_fix; cp RcppML.Rcheck/00check.log RcppML.Rcheck/00install.out $RW/out/Rcheck_fix/ 2>/dev/null; cp RcppML.Rcheck/tests/testthat.Rout* $RW/out/Rcheck_fix/ 2>/dev/null; cp RcppML.Rcheck/RcppML-Ex.timings $RW/out/Rcheck_fix/ 2>/dev/null
[ -f RcppML.Rcheck/RcppML-Ex.timings ] && sort -t$'\t' -k4 -n -r RcppML.Rcheck/RcppML-Ex.timings | head -12
echo CHECKFIX_DONE
