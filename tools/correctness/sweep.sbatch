#!/bin/bash
#SBATCH --job-name=rcppml-sweep
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
set -uo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/jobs/rcppml/env.sh
SECS=(A B C D E F)
Rscript $WORK/jobs/rcppml/sweep.R ${SECS[$SLURM_ARRAY_TASK_ID]} cpu 2>&1
echo SWEEP_DONE
