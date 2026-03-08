#!/usr/bin/env python3
"""Fix namespace in obs_var_table.hpp from streampress to streampress::v2."""

filepath = "/mnt/home/debruinz/RcppML-2/inst/include/streampress/format/obs_var_table.hpp"

with open(filepath, 'r') as f:
    content = f.read()

content = content.replace(
    "namespace streampress {",
    "namespace streampress {\nnamespace v2 {",
    1  # only first occurrence
)
content = content.replace(
    "}  // namespace streampress",
    "}  // namespace v2\n}  // namespace streampress"
)

with open(filepath, 'w') as f:
    f.write(content)

print("Fixed namespace to streampress::v2")
