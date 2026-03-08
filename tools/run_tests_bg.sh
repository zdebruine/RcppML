#!/bin/bash
# Run full test suite - intended to be run via: ssh c001 'nohup bash tools/run_tests_bg.sh &'
cd /mnt/home/debruinz/RcppML-2
module load r/4.5.2
export OMP_NUM_THREADS=4
Rscript tools/run_full_tests.R > test_output.log 2>&1
echo "DONE" >> test_output.log
