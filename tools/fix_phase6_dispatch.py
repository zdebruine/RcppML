#!/usr/bin/env python3
"""Phase 6b: Add .st_dispatch(), .get_available_ram_bytes(), .get_available_vram_bytes() to R/streampress.R"""

path = "R/streampress.R"
with open(path, "r") as f:
    content = f.read()

dispatch_code = '''
# =============================================================================
# Internal helpers for auto-dispatch
# =============================================================================

#' @keywords internal
.get_available_ram_bytes <- function() {
  Rcpp_get_available_ram_mb() * 1024 * 1024
}

#' @keywords internal
.get_available_vram_bytes <- function() {
  if (!gpu_available()) return(0)
  info <- gpu_info()
  if (is.null(info) || nrow(info) == 0) return(0)
  max(info$free_mb) * 1024 * 1024
}

#' Auto-dispatch for StreamPress NMF
#'
#' Determines the optimal execution mode (in-core vs streaming, CPU vs GPU)
#' based on available system resources and estimated decompressed data size.
#'
#' @param path Path to a .spz file
#' @param k Factorization rank (unused currently, reserved for future memory estimation)
#' @param resource One of \\code{"auto"}, \\code{"cpu"}, \\code{"gpu"}
#' @return A list with components \\code{mode}, \\code{resource}, \\code{streaming}
#' @keywords internal
.st_dispatch <- function(path, k, resource = "auto") {
  info <- st_info(path)
  file_size <- file.info(path)$size

  # Estimate decompressed size based on file version and compression
  decomp_factor <- if (info$version == 2L) 8.0 else 1.0
  est_decomp_bytes <- file_size * decomp_factor

  # Available RAM
  avail_ram <- .get_available_ram_bytes()

  # Available VRAM (0 if no GPU or GPU not requested)
  avail_vram <- if (resource %in% c("gpu", "auto")) {
    .get_available_vram_bytes()
  } else {
    0
  }

  safety <- 0.70

  if (avail_vram > 0 && est_decomp_bytes < avail_vram * safety) {
    list(mode = "IN_CORE_GPU", resource = "gpu", streaming = FALSE)
  } else if (avail_vram > 0 && est_decomp_bytes < avail_ram * safety) {
    list(mode = "CPU_TO_GPU", resource = "gpu", streaming = FALSE)
  } else if (avail_vram > 0) {
    list(mode = "STREAMING_GPU", resource = "gpu", streaming = TRUE)
  } else if (est_decomp_bytes < avail_ram * safety) {
    list(mode = "IN_CORE_CPU", resource = "cpu", streaming = FALSE)
  } else {
    list(mode = "STREAMING_CPU", resource = "cpu", streaming = TRUE)
  }
}
'''

# Append before the st_add_transpose section
marker = "\n#' Add Transpose Section to an Existing .spz File"
content = content.replace(marker, dispatch_code + marker, 1)

with open(path, "w") as f:
    f.write(content)

print("OK: Added .st_dispatch() and helpers to R/streampress.R")
