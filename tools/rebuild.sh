#!/bin/bash
cd /mnt/home/debruinz/RcppML-2
module load r/4.5.2
export OMP_NUM_THREADS=4
Rscript -e 'devtools::document()' 2>&1
bash tools/fix_rcpp_info_bug.sh 2>&1
Rscript -e 'devtools::install(quick=TRUE)' 2>&1
echo "REBUILD_DONE"
