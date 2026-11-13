#!/bin/bash
#SBATCH --job-name=rcppml-buildfix
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Builds the working tree that was pushed to $WORK/src/RcppML/fix into $RW/rlib/fix.
# Regenerates RcppExports, NAMESPACE and man/; copies the regenerated files to $RW/out/generated.
set -uo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/jobs/rcppml/env.sh
FIX=$WORK/src/RcppML/fix; mkdir -p $RW/rlib/fix
rm -rf $TMPDIR/pkg && cp -r $FIX $TMPDIR/pkg && cd $TMPDIR/pkg && rm -rf docs src/*.o src/*.so
find . -type f \( -name "*.sh" -o -name "*.R" -o -name "*.Rmd" -o -name "*.cpp" -o -name "*.h" -o -name "*.hpp" -o -name "*.cu" -o -name "*.cuh" -o -name "Makevars*" -o -name "Makefile*" -o -name DESCRIPTION -o -name NAMESPACE -o -name "*.Rd" -o -name "*.md" -o -name "*.yaml" -o -name configure -o -name cleanup \) -exec sed -i 's/\r$//' {} +
[ -f configure ] && chmod +x configure cleanup
Rscript -e 'for (p in c("roxygen2")) if (!requireNamespace(p, quietly=TRUE)) install.packages(p, lib=Sys.getenv("R_LIBS_USER"), repos="https://cloud.r-project.org", Ncpus=16)' > $RW/out/fix_deps.log 2>&1
echo "=== compileAttributes ==="
Rscript -e 'Rcpp::compileAttributes(".")' 2>&1 | tail -3
[ -f tools/fix_rcpp_info_bug.sh ] && bash tools/fix_rcpp_info_bug.sh | tail -2
echo "=== install ==="
/usr/bin/time -v R CMD INSTALL --library=$RW/rlib/fix . > $RW/out/fix_install.log 2>&1; rc=$?; echo "install exit: $rc"
grep -E "Elapsed" $RW/out/fix_install.log
if [ $rc -ne 0 ]; then
  grep -E " error: |^Error|ERROR" $RW/out/fix_install.log | sed -E 's/^.*inst\/include\///' | sort | uniq -c | sort -rn | head -70
  echo "--- first errors with context ---"
  grep -n -A7 -E " error: " $RW/out/fix_install.log | grep -v "ignoring attributes\|Wignored" | head -120
  tail -15 $RW/out/fix_install.log
  echo BUILD_FAILED; exit 1
fi
echo "=== document (against the installed build) ==="
R_LIBS=$RW/rlib/fix:$R_LIBS Rscript -e 'roxygen2::roxygenise(".", load_code = roxygen2::load_installed)' > $RW/out/fix_document.log 2>&1; echo "document exit: $?"
grep -iE "^Error|^Warning|NAMESPACE|\[.*\] @" $RW/out/fix_document.log | head -40
R CMD INSTALL --library=$RW/rlib/fix . > $RW/out/fix_install2.log 2>&1; echo "reinstall exit: $?"; tail -3 $RW/out/fix_install2.log
echo "own-code warnings: $(grep -E 'warning:' $RW/out/fix_install.log | grep -vc 'Wignored-attributes')"
grep -E "warning:" $RW/out/fix_install.log | grep -v "Wignored-attributes" | sed -E 's/^.*inst\/include\///' | sort | uniq -c | sort -rn | head -20
rm -rf $RW/out/generated && mkdir -p $RW/out/generated/R $RW/out/generated/src && cp NAMESPACE $RW/out/generated/ && cp -r man $RW/out/generated/ && cp R/RcppExports.R $RW/out/generated/R/ && cp src/RcppExports.cpp $RW/out/generated/src/
echo BUILD_OK
