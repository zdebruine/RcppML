#!/usr/bin/env python3
"""Fix R/sparsepress.R to add codec/delta params to sp_write_dense."""

path = "R/sparsepress.R"
with open(path, "r") as f:
    content = f.read()

# Update function signature
content = content.replace(
    """sp_write_dense <- function(x, path, include_transpose = FALSE,
                           chunk_cols = 256L, verbose = FALSE) {""",
    """sp_write_dense <- function(x, path, include_transpose = FALSE,
                           chunk_cols = 256L, codec = "raw",
                           delta = FALSE, verbose = FALSE) {""")

# Add codec map after validation
content = content.replace(
    """  if (!is.matrix(x) || !is.numeric(x)) {
    stop("'x' must be a numeric matrix")
  }
  path <- normalizePath(path, mustWork = FALSE)
  stats <- Rcpp_sp_write_dense(x, path,
                               include_transpose = include_transpose,
                               chunk_cols = as.integer(chunk_cols))""",
    """  if (!is.matrix(x) || !is.numeric(x)) {
    stop("'x' must be a numeric matrix")
  }
  codec_map <- c(raw = 0L, fp16 = 1L, quant8 = 2L,
                 fp16_rans = 3L, fp32_rans = 4L)
  codec_int <- codec_map[match.arg(codec, names(codec_map))]
  path <- normalizePath(path, mustWork = FALSE)
  stats <- Rcpp_sp_write_dense(x, path,
                               include_transpose = include_transpose,
                               chunk_cols = as.integer(chunk_cols),
                               codec = codec_int,
                               delta = delta)""")

# Update verbose message
content = content.replace(
    """  if (verbose) {
    message(sprintf("SPZ v3 dense: %d x %d -> %s (%d chunks%s)",
                    stats$rows, stats$cols,
                    .format_bytes(stats$file_bytes),
                    stats$num_chunks,
                    if (include_transpose) ", with transpose" else ""))
  }""",
    """  if (verbose) {
    codec_label <- if (codec_int == 0L) "" else paste0(", codec=", codec)
    ratio <- if (stats$raw_bytes > 0) stats$file_bytes / stats$raw_bytes else NA
    message(sprintf("SPZ v3 dense: %d x %d -> %s (%d chunks%s%s, ratio=%.2f)",
                    stats$rows, stats$cols,
                    .format_bytes(stats$file_bytes),
                    stats$num_chunks,
                    if (include_transpose) ", with transpose" else "",
                    codec_label, ratio))
  }""")

# Update @param docs
content = content.replace(
    """#' @param chunk_cols Integer; columns per chunk. Default 256.
#' @param verbose Logical; print write statistics. Default \\code{FALSE}.""",
    """#' @param chunk_cols Integer; columns per chunk. Default 256.
#' @param codec Compression codec. One of \\code{"raw"} (uncompressed float32),
#'   \\code{"fp16"} (half precision), \\code{"quant8"} (8-bit quantization),
#'   \\code{"fp16_rans"} (fp16 + rANS entropy coding),
#'   \\code{"fp32_rans"} (fp32 + rANS). Default \\code{"raw"}.
#' @param delta Logical; apply XOR-delta encoding before entropy coding.
#'   Only used with rANS codecs. Default \\code{FALSE}.
#' @param verbose Logical; print write statistics. Default \\code{FALSE}.""")

with open(path, "w") as f:
    f.write(content)

count = content.count("codec")
print(f"R/sparsepress.R updated. 'codec' appears {count} times.")
