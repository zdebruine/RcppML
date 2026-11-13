#!/bin/bash
#SBATCH --job-name=rcppml-setup
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -uo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/jobs/rcppml/env.sh
module avail 2>&1 | grep -oiE "\b(r|cuda|gcc|pandoc|hdf5)/[0-9][^ ]*" | sort -u | tr '\n' ' '; echo
mkdir -p $WORK/src/RcppML
if [ -d $SRC/.git ]; then git -C $SRC fetch origin && git -C $SRC reset --hard origin/main && git -C $SRC clean -fdx
else git clone https://github.com/zdebruine/RcppML $SRC; fi
git -C $SRC log --oneline -1
R --version | head -1; gcc --version | head -1; which pandoc qpdf || true
Rscript $WORK/jobs/rcppml/deps.R
echo "=== Issue #78: plain-checkout install (file modes as committed) ==="
ls -l $SRC/configure $SRC/cleanup
mkdir -p $TMPDIR/lib78
( cd $SRC && R CMD INSTALL --library=$TMPDIR/lib78 . 2>&1 | tail -5 ); echo "plain install exit: ${PIPESTATUS[0]}"
git -C $SRC status --short | head; git -C $SRC clean -fdxq; git -C $SRC checkout -q .
echo "=== install current main ==="
cp -r $SRC $TMPDIR/build && chmod +x $TMPDIR/build/configure $TMPDIR/build/cleanup
( cd $TMPDIR/build && /usr/bin/time -v R CMD INSTALL --library=$R_LIBS_USER . > $RW/out/install_main.log 2>&1 ); echo "main install exit: $?"
echo "compiler warnings: $(grep -ciE 'warning' $RW/out/install_main.log)"; grep -E "Elapsed|Maximum resident" $RW/out/install_main.log; tail -5 $RW/out/install_main.log
echo "=== old versions for #79 ==="
Rscript -e 'install.packages("RcppML", lib=file.path(Sys.getenv("RW"),"rlib/old037"), repos="https://cloud.r-project.org")' > $RW/out/install_037.log 2>&1; echo "CRAN release exit: $?"
git clone -q --branch v0.5.6 --depth 1 https://github.com/zdebruine/RcppML $TMPDIR/v056
chmod +x $TMPDIR/v056/configure $TMPDIR/v056/cleanup 2>/dev/null
( cd $TMPDIR/v056 && R CMD INSTALL --library=$RW/rlib/old056 . > $RW/out/install_056.log 2>&1 ); echo "0.5.6 exit: $?"; tail -3 $RW/out/install_056.log
for l in 4.5.2 old037 old056; do echo "$l: $(grep -m1 '^Version' $RW/rlib/$l/RcppML/DESCRIPTION 2>/dev/null)"; done
test -d $R_LIBS_USER/RcppML || { echo SETUP_FAILED; exit 1; }
echo SETUP_DONE
