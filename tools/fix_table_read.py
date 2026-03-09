#!/usr/bin/env python3
"""Fix the total_bytes calculation in read_table_at_offset."""
path = "/mnt/home/debruinz/RcppML-2/src/sparsepress_bridge.cpp"
with open(path, "r") as f:
    src = f.read()

# The bug: total_bytes = hdr_plus_desc + max_end
# But max_end already includes hdr_plus_desc since data_offset is absolute
old = "size_t total_bytes = hdr_plus_desc + static_cast<size_t>(max_end);"
new = "size_t total_bytes = std::max(hdr_plus_desc, static_cast<size_t>(max_end));"

if old in src:
    src = src.replace(old, new, 1)
    print("Fixed total_bytes calculation in read_table_at_offset")
else:
    print("WARNING: Could not find total_bytes pattern")

with open(path, "w") as f:
    f.write(src)
print("Done.")
