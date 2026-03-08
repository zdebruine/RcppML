#!/usr/bin/env python3
"""Make Rcpp_sp_write always use v2 format."""

BRIDGE = "/mnt/home/debruinz/RcppML-2/src/sparsepress_bridge.cpp"

with open(BRIDGE) as f:
    content = f.read()

old = '''    // Use v2 if any v2-specific features are requested
    bool use_v2 = (precision != "auto" && precision != "fp64") ||
                  row_sort || include_transpose ||
                  !rownames.empty() || !colnames.empty();'''

new = '''    // Always use v2 format — v1 is legacy
    bool use_v2 = true;'''

if old in content:
    content = content.replace(old, new)
    with open(BRIDGE, 'w') as f:
        f.write(content)
    print("[OK] Changed to always use v2 format")
else:
    print("[SKIP] Pattern not found or already changed")
