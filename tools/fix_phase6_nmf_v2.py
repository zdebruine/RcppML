#!/usr/bin/env python3
"""Phase 6c v2: Wire auto-dispatch into nmf_thin.R using line-based insertion"""

path = "R/nmf_thin.R"
with open(path, "r") as f:
    lines = f.readlines()

# Find the line with "# --- Streaming SPZ shortcut ---"
target_line = None
for i, line in enumerate(lines):
    if "# --- Streaming SPZ shortcut ---" in line:
        target_line = i
        break

if target_line is None:
    print("ERROR: Could not find '# --- Streaming SPZ shortcut ---'")
    exit(1)

# Insert auto-dispatch block BEFORE the streaming shortcut section
dispatch_block = [
    "  # --- StreamPress auto-dispatch ---\n",
    "  # If data is a .spz file path, auto-detect the optimal execution mode\n",
    "  # based on available RAM/VRAM and estimated data size.\n",
    "  if (is.character(data) && length(data) == 1 &&\n",
    '      grepl("\\\\.spz$", data, ignore.case = TRUE)) {\n',
    '    if (is.null(dispatch) || identical(dispatch, "auto")) {\n',
    "      mode_info <- .st_dispatch(data, k, resource = resource)\n",
    '      if (verbose) message("StreamPress auto-dispatch: ", mode_info$mode)\n',
    "      resource <- mode_info$resource\n",
    "      streaming <- mode_info$streaming\n",
    "    } else {\n",
    "      message(\n",
    '        "WARNING: `dispatch` is set manually to \'", dispatch, "\'.\\\n",\n',
    '        "  Auto-dispatch ensures sufficient RAM is available before loading.\\\n",\n',
    '        "  Manual dispatch may cause out-of-memory errors or crashes.\\\n",\n',
    '        "  Remove `dispatch=` to restore safe automatic mode."\n',
    "      )\n",
    "    }\n",
    "  }\n",
    "\n",
]

lines[target_line:target_line] = dispatch_block

with open(path, "w") as f:
    f.writelines(lines)

print(f"OK: Inserted auto-dispatch block at line {target_line + 1}")
