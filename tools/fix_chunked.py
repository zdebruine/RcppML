#!/usr/bin/env python3
"""Fix broken FACTORNET_LOG_NMF macro calls in fit_chunked.hpp.

The debug XYZZY strings were partially removed, leaving dangling
FACTORNET_LOG_NMF(... verbose, lines with no format string.
This script removes those broken lines entirely.
"""
import re

path = "inst/include/FactorNet/nmf/fit_chunked.hpp"
with open(path, "r") as f:
    lines = f.readlines()

# Find lines that have FACTORNET_LOG_NMF(...verbose, but no closing );
# Pattern: line ends with "verbose," and the next line does NOT start with
# a string literal (format string). These are the broken ones.
to_remove = set()
for i, line in enumerate(lines):
    stripped = line.rstrip()
    if stripped.endswith("verbose,"):
        # Check if next line starts with a string literal (format string)
        if i + 1 < len(lines):
            next_stripped = lines[i + 1].strip()
            if not next_stripped.startswith('"'):
                to_remove.add(i)
                print(f"Removing broken macro at line {i+1}: {stripped}")

if not to_remove:
    print("No broken macro calls found.")
else:
    new_lines = [l for i, l in enumerate(lines) if i not in to_remove]
    with open(path, "w") as f:
        f.writelines(new_lines)
    print(f"Removed {len(to_remove)} broken lines. File has {len(new_lines)} lines.")
