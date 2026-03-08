#!/usr/bin/env python3
"""Phase 6c: Wire auto-dispatch into nmf_thin.R
1. Add dispatch=NULL parameter
2. Add auto-dispatch logic before streaming SPZ section
"""

path = "R/nmf_thin.R"
with open(path, "r") as f:
    content = f.read()

# 1. Add dispatch=NULL parameter after profile=FALSE
old_params = """                profile = FALSE,
                ...)"""
new_params = """                profile = FALSE,
                dispatch = NULL,
                ...)"""
content = content.replace(old_params, new_params, 1)

# 2. Replace the streaming dispatch logic
old_streaming = """  # --- Streaming SPZ shortcut ---
  # If data is a .spz file path AND streaming is enabled,
  # use streaming NMF directly (no decompression into memory).
  # This requires a v2 .spz file with include_transpose=TRUE.
  # Supports standard NMF, NA masking, and cross-validation (CV).
  # Streaming is enabled when: streaming=TRUE, or streaming="auto" with SPZ input,
  # or the global option RcppML.streaming is TRUE.
  use_streaming <- isTRUE(streaming) ||
    (identical(streaming, "auto") && is.character(data) && length(data) == 1 &&
     grepl("\\.spz$", data)) ||
    isTRUE(getOption("RcppML.streaming", FALSE))
  if (is.character(data) && length(data) == 1 && grepl("\\.spz$", data) &&
      use_streaming) {"""

new_streaming = """  # --- StreamPress auto-dispatch ---
  # If data is a .spz file path, auto-detect the optimal execution mode
  # based on available RAM/VRAM and estimated data size.
  if (is.character(data) && length(data) == 1 &&
      grepl("\\.spz$", data, ignore.case = TRUE)) {
    if (is.null(dispatch) || identical(dispatch, "auto")) {
      mode_info <- .st_dispatch(data, k, resource = resource)
      if (verbose) message("StreamPress auto-dispatch: ", mode_info$mode)
      resource <- mode_info$resource
      streaming <- mode_info$streaming
    } else {
      message(
        "WARNING: `dispatch` is set manually to '", dispatch, "'.\n",
        "  Auto-dispatch ensures sufficient RAM is available before loading.\n",
        "  Manual dispatch may cause out-of-memory errors or crashes.\n",
        "  Remove `dispatch=` to restore safe automatic mode."
      )
    }
  }

  # --- Streaming SPZ shortcut ---
  # If data is a .spz file path AND streaming is enabled,
  # use streaming NMF directly (no decompression into memory).
  # This requires a v2 .spz file with include_transpose=TRUE.
  use_streaming <- isTRUE(streaming) ||
    (identical(streaming, "auto") && is.character(data) && length(data) == 1 &&
     grepl("\\.spz$", data)) ||
    isTRUE(getOption("RcppML.streaming", FALSE))
  if (is.character(data) && length(data) == 1 && grepl("\\.spz$", data) &&
      use_streaming) {"""

content = content.replace(old_streaming, new_streaming, 1)

with open(path, "w") as f:
    f.write(content)

print("OK: Wired auto-dispatch into nmf_thin.R")
