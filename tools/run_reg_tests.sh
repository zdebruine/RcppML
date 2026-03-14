#!/bin/bash
cd /mnt/home/debruinz/RcppML-2
module load r/4.5.2
export OMP_NUM_THREADS=4
Rscript -e 'devtools::test(filter="regularization")' 2>&1
echo "TESTS_DONE"
