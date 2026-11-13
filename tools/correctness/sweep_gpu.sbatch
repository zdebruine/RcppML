#!/bin/bash
#SBATCH --job-name=rcppml-sweep-gpu
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:tesla_v100s:1
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -uo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/jobs/rcppml/env.sh
module load cuda/12.9 || exit 1
export R_LIBS_USER=$RW/rlib/gpu-4.5.2; export R_LIBS=$R_LIBS_USER:$RW/rlib/4.5.2
Rscript -e 'library(RcppML); cat("gpu:", gpu_available(), "\n")'
for s in A B C D E; do echo "######## SECTION $s (gpu)"; Rscript $WORK/jobs/rcppml/sweep.R $s gpu 2>&1; done
echo SWEEP_GPU_DONE
