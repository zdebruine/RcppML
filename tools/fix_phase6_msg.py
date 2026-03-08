#!/usr/bin/env python3
"""Phase 6c v3: Fix the message() call in nmf_thin.R to use proper \\n escapes"""

path = "R/nmf_thin.R"
with open(path, "r") as f:
    content = f.read()

# Replace the broken message block with correct one
old_msg = '''      message(
        "WARNING: `dispatch` is set manually to '", dispatch, "'.\\
",
        "  Auto-dispatch ensures sufficient RAM is available before loading.\\
",
        "  Manual dispatch may cause out-of-memory errors or crashes.\\
",
        "  Remove `dispatch=` to restore safe automatic mode."
      )'''

new_msg = '''      message(
        "WARNING: `dispatch` is set manually to '", dispatch, "'.\\n",
        "  Auto-dispatch ensures sufficient RAM is available before loading.\\n",
        "  Manual dispatch may cause out-of-memory errors or crashes.\\n",
        "  Remove `dispatch=` to restore safe automatic mode."
      )'''

if old_msg in content:
    content = content.replace(old_msg, new_msg, 1)
    with open(path, "w") as f:
        f.write(content)
    print("OK: Fixed message() newline escapes")
else:
    print("WARNING: Could not find the broken message block, trying alternate fix")
    # Just do a line-by-line fix
    lines = content.split('\n')
    fixed = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Fix lines ending with .\  followed by empty ", on next line
        if line.rstrip().endswith("'.\\") or line.rstrip().endswith("loading.\\") or line.rstrip().endswith("crashes.\\"):
            # Merge with next line
            next_line = lines[i+1] if i+1 < len(lines) else ""
            if next_line.strip().startswith('",'):
                fixed.append(line.rstrip().rstrip('\\') + '\\n",')
                i += 2
                continue
        fixed.append(line)
        i += 1
    content = '\n'.join(fixed)
    with open(path, "w") as f:
        f.write(content)
    print("OK: Fixed message() via line merge")
