#!/usr/bin/env bash
# Remove run output so the suite is shippable again.
#
# compile/correctness/performance write build/*.json, python writes __pycache__,
# the GEAK harness regenerates ut/_cand_overlay and ut/_baseline_random.pt on
# every measurement, and the callable UTs write a ledger copy under reports/.
# All of it is regenerable, and the arena validator flags it as shipped junk, so
# strip it before handing the suite over or committing it.
#
# This script does NOT swallow failures any more. A GPU sweep run with
# `docker run -u 0` leaves root-owned files on the NFS export that a normal user
# cannot delete; that used to be reported as success while validate_suite.py went
# on to FAIL 12 of 15 tasks. Now it exits 1 and names the survivors.
set -uo pipefail
SUITE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

survey() {
  find "$SUITE" \( -type d \( -name build -o -name __pycache__ -o -name .pytest_cache \
       -o -name _cand_overlay -o -name reports -o -name .rocprofv3 \) -o -type f \( -name '*.pyc' \
       -o -name '_baseline_random.pt' \) \) 2>/dev/null
}

before=$(survey | wc -l)

find "$SUITE/tasks" -maxdepth 3 -type d -name build -prune -exec rm -rf {} + 2>/dev/null
find "$SUITE" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null
find "$SUITE" -type d -name .pytest_cache -prune -exec rm -rf {} + 2>/dev/null
find "$SUITE" -type d -name .rocprofv3 -prune -exec rm -rf {} + 2>/dev/null
find "$SUITE/tasks" -type d -name _cand_overlay -prune -exec rm -rf {} + 2>/dev/null
# The callable UTs write <task>/ut/reports/ledger/<case>.json.
find "$SUITE/tasks" -maxdepth 4 -type d -name reports -prune -exec rm -rf {} + 2>/dev/null
find "$SUITE/tasks" -type f -name '_baseline_random.pt' -delete 2>/dev/null
find "$SUITE" -type f -name '*.pyc' -delete 2>/dev/null

left="$(survey)"
echo "cleaned: $SUITE  ($before target(s) found, $(echo -n "$left" | grep -c . || true) left)"

if [[ -n "$left" ]]; then
  cat >&2 <<EOF
ERROR: the following could not be removed. They are almost certainly root-owned
       residue from a 'docker run -u 0' GPU sweep against the NFS export. Re-run
       this script as root from a compute node:

         docker run --rm -u 0 -v /shared_nfs:/shared_nfs --entrypoint /bin/bash \\
           <any image> -c 'bash $SUITE/tools/clean.sh'

EOF
  echo "$left" | sed 's/^/       /' >&2
  exit 1
fi
