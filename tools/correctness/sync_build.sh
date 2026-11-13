#!/bin/bash
# Push the working tree to Clipper (no git needed), build it there in a Slurm job, print the log.
# Usage (Git Bash, from the repo root): bash tools/correctness/sync_build.sh
set -u
C="python $HOME/.claude/skills/clipper/clipper.py"
W=/mnt/projects/debruinz_project/debruinz/singlet
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
$C run "rm -rf $W/src/RcppML/fix" >/dev/null
$C push "$(cygpath -w "$ROOT")" $W/src/RcppML/fix >/dev/null || { echo "push failed"; exit 2; }
for f in buildfix.sbatch env.sh run_tests.R sweep.R bench.R; do
  $C put "$(cygpath -w "$ROOT/tools/correctness/$f")" $W/jobs/rcppml/$f >/dev/null
done
J=$($C run "cd $W/jobs/rcppml && sbatch --parsable buildfix.sbatch")
echo "build job $J"
for i in $(seq 1 30); do sleep 30; st=$($C run "squeue -h -j $J -o %T" 2>/dev/null); [ -z "$st" ] && break; done
$C run "cat $W/logs/rcppml-buildfix-$J.out" | cut -c1-300
