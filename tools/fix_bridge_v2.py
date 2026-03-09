#!/usr/bin/env python3
"""Fix sparsepress_bridge.cpp: add obs_buf/var_buf population and has_obs/has_var to v2 return."""
import re

path = "/mnt/home/debruinz/RcppML-2/src/sparsepress_bridge.cpp"
with open(path, "r") as f:
    src = f.read()

changes = 0

# Change 1: After cfg2.chunk_cols line, add obs_buf/var_buf population
old_cfg2 = "cfg2.chunk_cols = static_cast<uint32_t>(chunk_cols);"
new_cfg2 = """cfg2.chunk_cols = static_cast<uint32_t>(chunk_cols);

        // Populate obs/var buffers from R raw vectors
        if (obs_raw.isNotNull()) {
            RawVector obs_rv(obs_raw.get());
            cfg2.obs_buf.assign(obs_rv.begin(), obs_rv.end());
        }
        if (var_raw.isNotNull()) {
            RawVector var_rv(var_raw.get());
            cfg2.var_buf.assign(var_rv.begin(), var_rv.end());
        }"""
if old_cfg2 in src:
    src = src.replace(old_cfg2, new_cfg2, 1)
    changes += 1
    print("1. Added obs_buf/var_buf population after cfg2.chunk_cols")
else:
    print("1. WARNING: Could not find cfg2.chunk_cols line")

# Change 2: Add has_obs/has_var to v2 return block
# The current v2 return has has_transpose as the last Named field before ");"
old_return = '            Named("has_transpose") = include_transpose\n        );'
new_return = """            Named("has_transpose") = include_transpose,
            Named("has_obs") = obs_raw.isNotNull(),
            Named("has_var") = var_raw.isNotNull()
        );"""
if old_return in src:
    src = src.replace(old_return, new_return, 1)
    changes += 1
    print("2. Added has_obs/has_var to v2 return block")
else:
    print("2. WARNING: Could not find v2 return block")

with open(path, "w") as f:
    f.write(src)

print(f"Done. Applied {changes} changes.")
