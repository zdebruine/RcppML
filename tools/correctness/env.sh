# common environment for RcppML jobs (sourced inside Slurm jobs only)
WORK=/mnt/projects/debruinz_project/debruinz/singlet
RW=$WORK/rcppml; export RW
SRC=$WORK/src/RcppML/main
source /etc/profile.d/lmod.sh
module load r/4.5.3 || exit 1
export R_LIBS_USER=$RW/rlib/4.5.2
export R_LIBS=$R_LIBS_USER
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MAKEFLAGS=-j${SLURM_CPUS_PER_TASK:-8}
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR; trap 'rm -rf $TMPDIR' EXIT
mkdir -p $RW/rlib/4.5.2 $RW/rlib/old037 $RW/rlib/old056 $RW/out
