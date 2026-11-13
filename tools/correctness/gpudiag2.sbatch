#!/bin/bash
#SBATCH --job-name=rcppml-gpudiag
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:tesla_v100s:1
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -uo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/jobs/rcppml/env.sh
module load cuda/12.9 || exit 1
export R_LIBS=$RW/rlib/fix:$RW/rlib/4.5.2 R_USER_CACHE_DIR=$RW/rcache
Rscript $WORK/jobs/rcppml/gpudiag2.R 2>&1 | grep -E "it1=|rror|exact"
