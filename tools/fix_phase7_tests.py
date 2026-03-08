#!/usr/bin/env python3
"""Fix test_streampress_compat.R to use correct field names from st_info()"""

path = "tests/testthat/test_streampress_compat.R"
with open(path, "r") as f:
    content = f.read()

# Fix st_info field names: m -> rows, n -> cols
content = content.replace("info$m", "info$rows")
content = content.replace("info$n", "info$cols")

with open(path, "w") as f:
    f.write(content)

print("OK: Fixed field names in test_streampress_compat.R")
