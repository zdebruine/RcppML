#!/bin/bash
#SBATCH --job-name=rcppml-bench
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclusive=user
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Usage: sbatch bench.sbatch "lib1:label1 lib2:label2 ..."   (libs relative to $RW/rlib)
set -uo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/jobs/rcppml/env.sh
export OMP_PROC_BIND=true OMP_PLACES=cores
lscpu | grep -E "Model name|^CPU\(s\)"
PAIRS="${1:-old037:v0.3.7 old056:v0.5.6 4.5.2:v1.0.0}"
for pair in $PAIRS; do
  lib=${pair%%:*}; label=${pair##*:}
  echo "######## $label ($lib)"
  R_LIBS=$RW/rlib/$lib:$RW/rlib/4.5.2 Rscript $WORK/jobs/rcppml/bench.R $RW/rlib/$lib $label 2>&1 | grep -E "^(v|fix|ERR)[^ ]* +[a-z]"
done
echo BENCH_DONE
