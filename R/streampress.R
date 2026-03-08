#' @title StreamPress I/O: Read, Write, and Inspect Compressed Matrices
#'
#' @description
#' Functions for reading and writing matrices in StreamPress (.spz) format.
#' StreamPress (formerly SparsePress) achieves 10-20x compression on typical
#' scRNA-seq sparse matrices using rANS entropy coding, and supports
#' dense v3 format with multiple compression codecs.
#'
#' @details
#' StreamPress (.spz) supports two format versions:
#' \itemize{
#'   \item \strong{v2 (sparse)}: Column-oriented compressed CSC format.
#'     Lossless, 10-20x compression. Self-describing header.
#'   \item \strong{v3 (dense)}: Column-major dense panels with optional
#'     FP16/QUANT8/rANS compression. For streaming NMF on dense data.
#' }
#'
#' @name streampress
#' @seealso \code{\link{st_write}}, \code{\link{st_read}}, \code{\link{st_info}},
#'   \code{\link{st_write_dense}}, \code{\link{st_read_dense}}
NULL

#' Write a sparse matrix to a StreamPress file
#'
#' @param x A sparse matrix (\code{dgCMatrix}) or object coercible to one.
#' @param path Output file path. Extension \code{.spz} is recommended.
#' @param delta Logical; use density-based delta prediction for structure.
#'   Default \code{TRUE}.
#' @param value_pred Logical; use value prediction for integer-valued data.
#'   Default \code{FALSE}.
#' @param verbose Logical; print compression statistics. Default \code{FALSE}.
#' @param precision Value precision: \code{"auto"} (default), \code{"fp32"},
#'   \code{"fp16"}, \code{"quant8"}, \code{"fp64"}.
#' @param row_sort Logical; sort rows by nnz for better compression.
#' @param include_transpose Logical; store CSC(A^T) in the file.
#' @param chunk_cols Integer; columns per chunk. Default 2048.
#'
#' @return Invisibly returns a list with compression statistics.
#'
#' @examples
#' \donttest{
#' library(Matrix)
#' A <- rsparsematrix(1000, 500, 0.05)
#' f <- tempfile(fileext = ".spz")
#' st_write(A, f)
#' B <- st_read(f)
#' all.equal(A, B)  # TRUE
#' unlink(f)
#' }
#'
#' @seealso \code{\link{st_read}}, \code{\link{st_info}}
#' @export
st_write <- function(x, path, delta = TRUE, value_pred = FALSE, verbose = FALSE,
                     precision = "auto", row_sort = FALSE,
                     include_transpose = FALSE,
                     chunk_cols = 2048L) {
  if (!inherits(x, "dgCMatrix")) {
    if (inherits(x, "Matrix") || is.matrix(x)) {
      x <- .to_dgCMatrix(x)
    } else {
      stop("'x' must be a dgCMatrix or coercible to one")
    }
  }

  path <- normalizePath(path, mustWork = FALSE)
  stats <- Rcpp_sp_write(x, path,
                         use_delta = delta,
                         use_value_pred = value_pred,
                         verbose = verbose,
                         precision = precision,
                         row_sort = row_sort,
                         include_transpose = include_transpose,
                         chunk_cols = as.integer(chunk_cols))

  if (verbose) {
    message(sprintf("StreamPress: %s -> %s (%.1fx compression, %.1f ms)",
                    .format_bytes(stats$raw_bytes),
                    .format_bytes(stats$compressed_bytes),
                    stats$ratio,
                    stats$compress_ms))
  }

  invisible(stats)
}

#' Read a StreamPress file into a dgCMatrix
#'
#' @param path Path to a \code{.spz} file.
#' @param cols Optional integer vector of column indices to read (1-indexed).
#' @param reorder Logical; if \code{TRUE} (default), undo any row permutation.
#'
#' @return A \code{dgCMatrix} sparse matrix.
#'
#' @examples
#' \donttest{
#' library(Matrix)
#' A <- rsparsematrix(100, 50, 0.1)
#' f <- tempfile(fileext = ".spz")
#' st_write(A, f)
#' B <- st_read(f)
#' all.equal(A, B)
#' unlink(f)
#' }
#'
#' @seealso \code{\link{st_write}}, \code{\link{st_info}}
#' @export
st_read <- function(path, cols = NULL, reorder = TRUE) {
  path <- normalizePath(path, mustWork = TRUE)
  Rcpp_sp_read(path, cols = cols, reorder = reorder)
}

#' Read Pre-Stored Transpose from a StreamPress File
#'
#' @param path Path to a \code{.spz} file.
#' @return A \code{dgCMatrix} representing CSC(A^T).
#'
#' @seealso \code{\link{st_write}}, \code{\link{st_read}}
#' @keywords internal
st_read_transpose <- function(path) {
  path <- normalizePath(path, mustWork = TRUE)
  Rcpp_sp_read_transpose(path)
}

#' Get metadata from a StreamPress file
#'
#' Reads only the header — no decompression is performed.
#'
#' @param path Path to a \code{.spz} file.
#'
#' @return A list with fields including \code{rows}, \code{cols}, \code{nnz},
#'   \code{density_pct}, \code{file_bytes}, \code{raw_bytes}, \code{ratio},
#'   \code{version}.
#'
#' @examples
#' \donttest{
#' library(Matrix)
#' A <- rsparsematrix(100, 50, 0.1)
#' f <- tempfile(fileext = ".spz")
#' st_write(A, f)
#' info <- st_info(f)
#' cat(sprintf("Matrix: %d x %d, nnz=%d\n", info$rows, info$cols, info$nnz))
#' unlink(f)
#' }
#'
#' @seealso \code{\link{st_read}}, \code{\link{st_write}}
#' @export
st_info <- function(path) {
  path <- normalizePath(path, mustWork = TRUE)
  Rcpp_sp_metadata(path)
}

#' Write a Dense Matrix to StreamPress v3 Format
#'
#' @param x A numeric matrix.
#' @param path Output file path. Extension \code{.spz} is recommended.
#' @param include_transpose Logical; store transposed panels. Default \code{FALSE}.
#' @param chunk_cols Integer; columns per chunk. Default 2048.
#' @param codec Compression codec: \code{"raw"}, \code{"fp16"}, \code{"quant8"},
#'   \code{"fp16_rans"}, \code{"fp32_rans"}. Default \code{"raw"}.
#' @param delta Logical; apply XOR-delta encoding. Default \code{FALSE}.
#' @param verbose Logical; print write statistics. Default \code{FALSE}.
#'
#' @return Invisibly returns a list with write statistics.
#'
#' @examples
#' \donttest{
#' A <- matrix(rnorm(1000), 50, 20)
#' f <- tempfile(fileext = ".spz")
#' st_write_dense(A, f, include_transpose = TRUE)
#' B <- st_read_dense(f)
#' max(abs(A - B))
#' unlink(f)
#' }
#'
#' @seealso \code{\link{st_read_dense}}, \code{\link{st_write}}
#' @export
st_write_dense <- function(x, path, include_transpose = FALSE,
                           chunk_cols = 2048L, codec = "raw",
                           delta = FALSE, verbose = FALSE) {
  if (!is.matrix(x) || !is.numeric(x)) {
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
                               delta = delta)
  if (verbose) {
    codec_label <- if (codec_int == 0L) "" else paste0(", codec=", codec)
    ratio <- if (stats$raw_bytes > 0) stats$file_bytes / stats$raw_bytes else NA
    message(sprintf("StreamPress v3 dense: %d x %d -> %s (%d chunks%s%s, ratio=%.2f)",
                    stats$rows, stats$cols,
                    .format_bytes(stats$file_bytes),
                    stats$num_chunks,
                    if (include_transpose) ", with transpose" else "",
                    codec_label, ratio))
  }
  invisible(stats)
}

#' Read a Dense Matrix from StreamPress v3 Format
#'
#' @param path Path to a StreamPress v3 \code{.spz} file.
#'
#' @return A numeric matrix.
#'
#' @examples
#' \donttest{
#' A <- matrix(rnorm(1000), 50, 20)
#' f <- tempfile(fileext = ".spz")
#' st_write_dense(A, f)
#' B <- st_read_dense(f)
#' max(abs(A - B))
#' unlink(f)
#' }
#'
#' @seealso \code{\link{st_write_dense}}, \code{\link{st_read}}
#' @export
st_read_dense <- function(path) {
  path <- normalizePath(path, mustWork = TRUE)
  Rcpp_sp_read_dense(path)
}

#' Convert Any Supported Format to StreamPress (.spz)
#'
#' One-time migration tool: read a matrix from any supported format
#' and write it as a \code{.spz} v2 file.
#'
#' @param input Path to input file or an in-memory matrix.
#' @param output Path for the output \code{.spz} file.
#' @param precision Value precision: \code{"auto"}, \code{"fp32"},
#'   \code{"fp16"}, \code{"quant8"}, \code{"fp64"}.
#' @param include_transpose Logical; store CSC(A^T).
#' @param row_sort Logical; sort rows by nnz. Default \code{TRUE}.
#' @param verbose Integer verbosity level (0=silent, 1=summary, 2=detailed).
#'
#' @return Invisibly returns a list with conversion statistics.
#'
#' @seealso \code{\link{st_read}}, \code{\link{st_write}}, \code{\link{st_info}}
#' @keywords internal
st_convert <- function(input, output, precision = "auto",
                       include_transpose = FALSE,
                       row_sort = TRUE,
                       verbose = 1L) {
  verbose <- as.integer(verbose)

  input_format <- "in-memory"
  if (is.character(input) && length(input) == 1) {
    if (!file.exists(input)) stop("Input file not found: ", input)
    ext <- tolower(tools::file_ext(input))
    if (ext == "gz") {
      base_ext <- tolower(tools::file_ext(sub("\\.[^.]+$", "", input)))
      ext <- paste0(base_ext, ".gz")
    }
    input_format <- ext
    if (verbose >= 1L) {
      message(sprintf("st_convert: reading %s (format: %s)", basename(input), ext))
    }
  }

  x <- validate_data(input)$data

  if (verbose >= 1L) {
    is_sparse <- inherits(x, "dgCMatrix")
    nnz <- if (is_sparse) length(x@x) else sum(x != 0)
    message(sprintf("st_convert: %d x %d matrix, nnz = %d (%.1f%% dense)",
                    nrow(x), ncol(x), nnz,
                    nnz / (as.double(nrow(x)) * ncol(x)) * 100))
  }

  if (!inherits(x, "dgCMatrix")) {
    x <- .to_dgCMatrix(x)
  }

  output <- normalizePath(output, mustWork = FALSE)
  stats <- st_write(x, output,
                    precision = precision,
                    row_sort = row_sort,
                    verbose = (verbose >= 2L))

  if (verbose >= 1L) {
    message(sprintf("st_convert: written to %s (%.1fx compression)",
                    basename(output), stats$ratio))
  }

  if (input_format == "h5ad" && is.character(input)) {
    tryCatch({
      meta <- .extract_h5ad_metadata(input)
      if (length(meta) > 0) {
        meta_path <- paste0(output, ".meta")
        .write_sidecar(meta_path, meta)
        if (verbose >= 1L) {
          message(sprintf("st_convert: sidecar metadata written to %s", basename(meta_path)))
        }
      }
    }, error = function(e) {
      if (verbose >= 1L) {
        message("st_convert: note: could not extract h5ad metadata: ", conditionMessage(e))
      }
    })
  }

  invisible(list(
    input_format = input_format,
    output_path = output,
    rows = nrow(x),
    cols = ncol(x),
    nnz = length(x@x),
    compress_stats = stats
  ))
}
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
#' @param resource One of \code{"auto"}, \code{"cpu"}, \code{"gpu"}
#' @return A list with components \code{mode}, \code{resource}, \code{streaming}
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

#' Add Transpose Section to an Existing .spz File
#'
#' Reads an existing v2 \code{.spz} file without a transpose section,
#' builds CSC(A^T), and rewrites the file with the transpose section included.
#' If the file already has a transpose section, this is a no-op.
#'
#' @param path Path to an existing \code{.spz} v2 file.
#' @param verbose Logical; print progress messages. Default \code{TRUE}.
#'
#' @return Invisibly returns the path.
#'
#' @examples
#' \dontrun{
#' # Requires a v2 .spz file
#' st_add_transpose("data.spz")
#' }

#'
#' @seealso \code{\link{st_write}}, \code{\link{st_read}}
#' @export
st_add_transpose <- function(path, verbose = TRUE) {
  path <- normalizePath(path, mustWork = TRUE)
  Rcpp_st_add_transpose(path, verbose = verbose)
  invisible(path)
}
