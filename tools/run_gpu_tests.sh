#!/bin/bash
module load r/4.5.2 cuda/12.8.1
export OMP_NUM_THREADS=4
cd /mnt/home/debruinz/RcppML-2
mkdir -p logs
Rscript -e 'devtools::test(filter = "gpu")' > logs/gpu_test_fp32_v5.log 2>&1
echo "TESTS_DONE" >> logs/gpu_test_fp32_v5.log
