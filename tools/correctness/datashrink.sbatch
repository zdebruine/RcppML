#!/bin/bash
#SBATCH --job-name=rcppml-data
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -uo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/jobs/rcppml/env.sh
export R_LIBS=$RW/rlib/fix:$RW/rlib/4.5.2
rm -rf $TMPDIR/pkg && mkdir -p $TMPDIR/pkg && cp -r $WORK/src/RcppML/fix/data $TMPDIR/pkg/ && cd $TMPDIR/pkg
Rscript $WORK/jobs/rcppml/datashrink.R 2>&1 | grep -vE "^Loading|^Attaching|masked|^ *svd$|^$"
rm -rf $RW/out/data_new && cp -r data $RW/out/data_new && ls -la $RW/out/data_new
echo DATA_DONE
