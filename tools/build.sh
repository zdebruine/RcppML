#!/bin/bash
set -e
source /etc/profile.d/modules.sh 2>/dev/null || true
module load r/4.5.2
cd /mnt/home/debruinz/RcppML-2/src
rm -f *.o *.so
cd /mnt/home/debruinz/RcppML-2
R CMD INSTALL . 2>&1
echo "BUILD_SUCCESS"
