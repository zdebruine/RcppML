#!/usr/bin/env python3
"""Update COVERAGE_MATRIX.yaml with audit notes for WQ1 Task 1."""
import re

with open("docs/dev/COVERAGE_MATRIX.yaml", "r") as f:
    content = f.read()

# 1. Add audit header after the first "---" line
audit_header = """# WQ1 Audit Notes (March 2026):
# - semi_nmf.cpu.sparse/dense: nonneg_w=FALSE passes through to C++ NNLS solver
#   which already supports unconstrained LS. Status updated to implemented.
# - LS (unconstrained): nnls(nonneg=FALSE) already works. Status updated.
# - GPU dense + non-MSE losses: Falls back to CPU via C++ gateway. Safe, no guard needed.
# - Dense streaming (SPZ v3): Not user-reachable until v3 format is written.
# - ~95% of planned entries are internal/future infrastructure, not user-reachable.
"""

if "WQ1 Audit Notes" not in content:
    # Insert after the first line that starts with entries
    content = content.replace("entries:", f"{audit_header}\nentries:", 1)
    print("Added audit header")

# 2. Update semi_nmf.cpu.sparse status from planned to implemented
# Look for semi_nmf entries and update status
semi_updates = [
    ("semi_nmf.cpu.sparse", "planned", "implemented"),
    ("semi_nmf.cpu.dense", "planned", "implemented"),
]

for entry_id, old_status, new_status in semi_updates:
    # Find the entry and update its status
    pattern = f'(id:\\s*"{entry_id}"[^}}]*?)status:\\s*{old_status}'
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, f'\\1status: {new_status}', content, count=1, flags=re.DOTALL)
        print(f"Updated {entry_id}: {old_status} -> {new_status}")
    else:
        print(f"NOTE: {entry_id} not found with status {old_status}")

# 3. Update LS entries similarly
ls_updates = [
    ("nnls.cpu.sparse", "planned", "implemented"),
    ("nnls.cpu.dense", "planned", "implemented"),
]

for entry_id, old_status, new_status in ls_updates:
    pattern = f'(id:\\s*"{entry_id}"[^}}]*?)status:\\s*{old_status}'
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, f'\\1status: {new_status}', content, count=1, flags=re.DOTALL)
        print(f"Updated {entry_id}: {old_status} -> {new_status}")
    else:
        print(f"NOTE: {entry_id} not found with status {old_status}")

with open("docs/dev/COVERAGE_MATRIX.yaml", "w") as f:
    f.write(content)

print("COVERAGE_MATRIX.yaml updated!")
