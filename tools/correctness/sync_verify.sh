#!/bin/bash
# Run the test suite and the correctness sweep against the build in rlib/fix and print a compact report.
# Usage: bash tools/correctness/sync_verify.sh [array-spec, default 0-6]
set -u
C="python $HOME/.claude/skills/clipper/clipper.py"
W=/mnt/projects/debruinz_project/debruinz/singlet
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
for f in verify.sbatch sweep.R run_tests.R; do $C put "$(cygpath -w "$ROOT/tools/correctness/$f")" $W/jobs/rcppml/$f >/dev/null; done
J=$($C run "cd $W/jobs/rcppml && sbatch --parsable --array=${1:-0-6} verify.sbatch")
echo "verify job $J"
for i in $(seq 1 60); do sleep 30; st=$($C run "squeue -h -j $J -o %T" 2>/dev/null); [ -z "$st" ] && break; done
$C run "cd $W/logs; sacct -j $J -X -n -o JobID%14,State,Elapsed | tr '\n' ';'; echo
for i in 0 1 2 3 4 5; do f=rcppml-verify-${J}_\$i.out; [ -f \$f ] || continue
  echo \"######## sweep \$i: \$(grep -cE '^PASS' \$f) pass, \$(grep -cE '^(FAIL|NOOP|ERR |WRON|PART)' \$f) not; \$(grep -c CRASHED \$f) crash\"
  grep -E '^(FAIL|NOOP|ERR |WRON|PART|CRASH|Error)' \$f | sed -E 's/ +\|/|/g' | cut -c1-${2:-250} | head -${3:-60}
done
f=rcppml-verify-${J}_6.out; if [ -f \$f ]; then echo '######## tests'; grep -E '^\[ FAIL|^TOTAL' \$f | tail -2
  awk '/Failed tests/{p=1} p' \$f | grep -E '^(── |Error|Expected|\`actual|\`expected|Actual)' | cut -c1-240 | head -${4:-160}; fi"
