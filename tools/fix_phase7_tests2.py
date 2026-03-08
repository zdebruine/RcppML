#!/usr/bin/env python3
"""Fix the mode_info$rowsode bug caused by overly broad replacement"""

path = "tests/testthat/test_streampress_compat.R"
with open(path, "r") as f:
    content = f.read()

# Fix: mode_info$rowsode -> mode_info$mode (the global m->rows replacement broke this)
content = content.replace("mode_info$rowsode", "mode_info$mode")

with open(path, "w") as f:
    f.write(content)

print("OK: Fixed mode_info$mode in test_streampress_compat.R")
