#!/bin/bash
module load r/4.5.2 cuda/12.8.1 2>/dev/null
cd /mnt/home/debruinz/RcppML-2/src
make -f Makefile.gpu install > /mnt/home/debruinz/RcppML-2/logs/build_fix.log 2>&1
echo "Exit code: $?" >> /mnt/home/debruinz/RcppML-2/logs/build_fix.log
echo "DONE" >> /mnt/home/debruinz/RcppML-2/logs/build_fix.log
