#!/bin/bash
#SBATCH --job-name=rcppml-repro
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -uo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/jobs/rcppml/env.sh
J=$WORK/jobs/rcppml
for s in r81 r82; do echo "=================== $s"; Rscript $J/$s.R 2>&1 | tail -80; done
echo "=================== r79 prep"; Rscript $J/r79_prep.R 2>&1 | tail
for l in old037 old056 4.5.2; do echo "=================== r79 with $l"; R_LIBS=$RW/rlib/$l:$R_LIBS_USER Rscript $J/r79_run.R $RW/rlib/$l 2>&1 | tail -40; done
echo REPRO_DONE
