#' @title SparsePress I/O: Read, Write, and Inspect Compressed Sparse Matrices
#'
#' @description
#' Functions for reading and writing sparse matrices in SparsePress (.spz) format.
#' SparsePress achieves 10-20x compression on typical scRNA-seq sparse matrices
#' using rANS entropy coding, enabling fast loading from disk or network.
#'
#' @details
#' SparsePress (.spz) is a column-oriented compressed format for CSC sparse
#' matrices. Key properties:
#' \itemize{
#'   \item Lossless compression: exact round-trip fidelity
#'   \item 10-20x compression ratio on typical scRNA-seq data
#'   \item Fast decompression (usually faster than reading uncompressed .rds)
#'   \item Self-describing header: metadata readable without decompression
#'   \item Column-level structure: preserves CSC layout for efficient access
#' }
#'
#' @name sparsepress
#' @seealso \code{\link{sp_write}}, \code{\link{sp_read}}, \code{\link{sp_info}},
#'   \code{\link{sp_write_dense}}, \code{\link{sp_read_dense}},
#'   \code{\link{sp_read_gpu}}
NULL

#' Write a sparse matrix to a SparsePress file
#'
#' @param x A sparse matrix (\code{dgCMatrix}) or object coercible to one.
#' @param path Output file path. Extension \code{.spz} is recommended.
#' @param delta Logical; use density-based delta prediction for structure.
#'   Default \code{TRUE}. Improves compression for structured sparsity patterns.
#' @param value_pred Logical; use value prediction for integer-valued data.
#'   Default \code{FALSE}. Only effective if all values are non-negative integers.
#' @param verbose Logical; print compression statistics. Default \code{FALSE}.
#' @param precision Value precision for v2 format: \code{"auto"} (default),
#'   \code{"fp32"}, \code{"fp16"}, \code{"quant8"}, \code{"fp64"}.
#' @param row_sort Logical; sort rows by nnz for better compression (v2 only).
#' @param include_transpose Logical; store CSC(A^T) in the file (v2 only).
#'   Approximately doubles file size but eliminates runtime transpose cost.
#'
#' @return Invisibly returns a list with compression statistics:
#'   \code{raw_bytes}, \code{compressed_bytes}, \code{ratio}, \code{compress_ms}.
#'
#' @examples
#' \donttest{
#' library(Matrix)
#' A <- rsparsematrix(1000, 500, 0.05)
#' f <- tempfile(fileext = ".spz")
#' sp_write(A, f)
#' B <- sp_read(f)
#' all.equal(A, B)  # TRUE
#' unlink(f)
#' }
#'
#' @seealso \code{\link{sp_read}}, \code{\link{sp_info}}
#' @export
sp_write <- function(x, path, delta = TRUE, value_pred = FALSE, verbose = FALSE,
                     precision = "auto", row_sort = FALSE,
                     include_transpose = FALSE) {
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
                         include_transpose = include_transpose)

  if (verbose) {
    message(sprintf("SparsePress: %s -> %s (%.1fx compression, %.1f ms)",
                    .format_bytes(stats$raw_bytes),
                    .format_bytes(stats$compressed_bytes),
                    stats$ratio,
                    stats$compress_ms))
  }

  invisible(stats)
}

#' Read a SparsePress file into a dgCMatrix
#'
#' @param path Path to a \code{.spz} file.
#' @param cols Optional integer vector of column indices to read (1-indexed).
#'   For v2 files, enables partial reads by only decoding the necessary chunks.
#'   Should be a contiguous range, e.g. \code{1:500}.
#' @param reorder Logical; if \code{TRUE} (default), undo any row permutation
#'   that was applied during compression (v2 only).
#'
#' @return A \code{dgCMatrix} sparse matrix.
#'
#' @examples
#' \donttest{
#' library(Matrix)
#' A <- rsparsematrix(100, 50, 0.1)
#' f <- tempfile(fileext = ".spz")
#' sp_write(A, f)
#' B <- sp_read(f)
#' all.equal(A, B)
#' unlink(f)
#' }
#'
#' @seealso \code{\link{sp_write}}, \code{\link{sp_info}}
#' @export
sp_read <- function(path, cols = NULL, reorder = TRUE) {
  path <- normalizePath(path, mustWork = TRUE)
  Rcpp_sp_read(path, cols = cols, reorder = reorder)
}

#' Read Pre-Stored Transpose from a SparsePress File
#'
#' Reads the CSC(A^T) section from a \code{.spz} v2 file that was
#' written with \code{include_transpose = TRUE}.
#'
#' @param path Path to a \code{.spz} file.
#' @return A \code{dgCMatrix} representing CSC(A^T).
#'
#' @examples
#' \dontrun{
#' library(Matrix)
#' A <- rsparsematrix(100, 50, 0.1)
#' f <- tempfile(fileext = ".spz")
#' sp_write(A, f, include_transpose = TRUE)
#' At <- sp_read_transpose(f)
#' dim(At)  # 50 x 100
#' unlink(f)
#' }
#'
#' @seealso \code{\link{sp_write}}, \code{\link{sp_read}}
#' @keywords internal
sp_read_transpose <- function(path) {
  path <- normalizePath(path, mustWork = TRUE)
  Rcpp_sp_read_transpose(path)
}

#' Get metadata from a SparsePress file
#'
#' Reads only the 72-byte header — no decompression is performed.
#'
#' @param path Path to a \code{.spz} file.
#'
#' @return A list with fields:
#'   \describe{
#'     \item{rows}{Number of rows (features)}
#'     \item{cols}{Number of columns (samples)}
#'     \item{nnz}{Number of non-zero entries}
#'     \item{density_pct}{Density as a percentage}
#'     \item{file_bytes}{Compressed file size in bytes}
#'     \item{raw_bytes}{Uncompressed size in bytes}
#'     \item{ratio}{Compression ratio (raw / compressed)}
#'     \item{version}{SparsePress format version}
#'     \item{integer_values}{Whether values are stored as integers}
#'     \item{delta_prediction}{Whether delta prediction was used}
#'     \item{value_prediction}{Whether value prediction was used}
#'   }
#'
#' @examples
#' \donttest{
#' library(Matrix)
#' A <- rsparsematrix(100, 50, 0.1)
#' f <- tempfile(fileext = ".spz")
#' sp_write(A, f)
#' info <- sp_info(f)
#' cat(sprintf("Matrix: %d x %d, nnz=%d, %.1fx compressed\n",
#'             info$rows, info$cols, info$nnz, info$ratio))
#' unlink(f)
#' }
#'
#' @seealso \code{\link{sp_read}}, \code{\link{sp_write}}
#' @export
sp_info <- function(path) {
  path <- normalizePath(path, mustWork = TRUE)
  Rcpp_sp_metadata(path)
}

#' Convert a sparse matrix to/from SparsePress format in memory
#'
#' @param x For \code{sp_compress}: a \code{dgCMatrix}. For
#'   \code{sp_decompress}: a raw vector from \code{sp_compress}.
#' @param delta Logical; use delta prediction. Default \code{TRUE}.
#' @param value_pred Logical; use value prediction. Default \code{FALSE}.
#'
#' @return \code{sp_compress} returns a raw vector. \code{sp_decompress}
#'   returns a \code{dgCMatrix}.
#'
#' @examples
#' \dontrun{
#' library(Matrix)
#' A <- rsparsematrix(100, 50, 0.1)
#' blob <- sp_compress(A)
#' B <- sp_decompress(blob)
#' all.equal(A, B)
#' }
#'
#' @seealso \code{\link{sp_read}}, \code{\link{sp_write}}
#' @rdname sp_compress
#' @keywords internal
sp_compress <- function(x, delta = TRUE, value_pred = FALSE) {
  if (!inherits(x, "dgCMatrix")) {
    x <- .to_dgCMatrix(x)
  }
  Rcpp_sp_compress(x, use_delta = delta, use_value_pred = value_pred)
}

#' @rdname sp_compress
#' @keywords internal
sp_decompress <- function(x) {
  if (!is.raw(x)) stop("'x' must be a raw vector")
  Rcpp_sp_decompress(x)
}

#' Write a Dense Matrix to SPZ v3 Format
#'
#' Writes a dense (non-sparse) matrix to a SparsePress v3 file, storing
#' column-major panels for streaming NMF. This enables out-of-core NMF
#' on dense data that does not fit in memory.
#'
#' @param x A numeric matrix.
#' @param path Output file path. Extension \code{.spz} is recommended.
#' @param include_transpose Logical; store transposed panels for streaming NMF.
#'   Default \code{FALSE}. Set to \code{TRUE} for streaming NMF use.
#' @param chunk_cols Integer; columns per chunk. Default 256.
#' @param verbose Logical; print write statistics. Default \code{FALSE}.
#'
#' @return Invisibly returns a list with write statistics.
#'
#' @examples
#' \donttest{
#' A <- matrix(rnorm(1000), 50, 20)
#' f <- tempfile(fileext = ".spz")
#' sp_write_dense(A, f, include_transpose = TRUE)
#' B <- sp_read_dense(f)
#' max(abs(A - B))  # small (float32 precision)
#' unlink(f)
#' }
#'
#' @seealso \code{\link{sp_read_dense}}, \code{\link{sp_write}}
#' @export
sp_write_dense <- function(x, path, include_transpose = FALSE,
                           chunk_cols = 256L, verbose = FALSE) {
  if (!is.matrix(x) || !is.numeric(x)) {
    stop("'x' must be a numeric matrix")
  }
  path <- normalizePath(path, mustWork = FALSE)
  stats <- Rcpp_sp_write_dense(x, path,
                               include_transpose = include_transpose,
                               chunk_cols = as.integer(chunk_cols))
  if (verbose) {
    message(sprintf("SPZ v3 dense: %d x %d -> %s (%d chunks%s)",
                    stats$rows, stats$cols,
                    .format_bytes(stats$file_bytes),
                    stats$num_chunks,
                    if (include_transpose) ", with transpose" else ""))
  }
  invisible(stats)
}

#' Read a Dense Matrix from SPZ v3 Format
#'
#' Reads a SparsePress v3 (dense) file back into an R numeric matrix.
#'
#' @param path Path to an SPZ v3 \code{.spz} file.
#'
#' @return A numeric matrix.
#'
#' @examples
#' \donttest{
#' A <- matrix(rnorm(1000), 50, 20)
#' f <- tempfile(fileext = ".spz")
#' sp_write_dense(A, f)
#' B <- sp_read_dense(f)
#' max(abs(A - B))  # small (float32 precision)
#' unlink(f)
#' }
#'
#' @seealso \code{\link{sp_write_dense}}, \code{\link{sp_read}}
#' @export
sp_read_dense <- function(path) {
  path <- normalizePath(path, mustWork = TRUE)
  Rcpp_sp_read_dense(path)
}

# Internal helper: format bytes for display
.format_bytes <- function(bytes) {
  if (bytes < 1024) return(sprintf("%d B", bytes))
  if (bytes < 1024^2) return(sprintf("%.1f KB", bytes / 1024))
  if (bytes < 1024^3) return(sprintf("%.1f MB", bytes / 1024^2))
  sprintf("%.1f GB", bytes / 1024^3)
}

#' Convert Any Supported Format to SparsePress (.spz)
#'
#' One-time migration tool: read a matrix from any supported format
#' (.h5ad, .h5, .loom, .rds, .mtx, .csv, .tsv) and write it as a
#' \code{.spz} v2 file for high-performance I/O.
#'
#' @param input Path to input file (any supported format) or an in-memory matrix.
#' @param output Path for the output \code{.spz} file.
#' @param precision Value precision: \code{"auto"} (default), \code{"fp32"},
#'   \code{"fp16"}, \code{"quant8"}, \code{"fp64"}.
#' @param include_transpose Logical; if \code{TRUE}, also store CSC(A^T) for
#'   pre-computed transpose (doubles file size, eliminates runtime transpose cost).
#' @param row_sort Logical; sort rows by nnz for better compression.
#'   Default \code{TRUE}.
#' @param verbose Integer verbosity level (0=silent, 1=summary, 2=detailed).
#'
#' @return Invisibly returns a list with:
#'   \describe{
#'     \item{input_format}{Detected input format}
#'     \item{output_path}{Path to the written .spz file}
#'     \item{rows, cols, nnz}{Matrix dimensions}
#'     \item{compress_stats}{Compression statistics from \code{sp_write()}}
#'   }
#'
#' @details
#' If the input is an \code{.h5ad} file, AnnData metadata (obs, var, uns) is
#' extracted and saved as a sidecar JSON file (\code{<output>.meta}).
#'
#' @examples
#' \dontrun{
#' # Convert an h5ad file to .spz
#' sp_convert("data.h5ad", "data.spz")
#'
#' # Convert with lossy half-precision
#' sp_convert("counts.mtx", "counts.spz", precision = "fp16")
#'
#' # Convert in-memory matrix
#' library(Matrix)
#' A <- rsparsematrix(1000, 500, 0.05)
#' sp_convert(A, "matrix.spz")
#' }
#'
#' @seealso \code{\link{sp_read}}, \code{\link{sp_write}}, \code{\link{sp_info}}
#' @keywords internal
sp_convert <- function(input, output, precision = "auto",
                       include_transpose = FALSE,
                       row_sort = TRUE,
                       verbose = 1L) {
  verbose <- as.integer(verbose)

  # Determine input format
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
      message(sprintf("sp_convert: reading %s (format: %s)", basename(input), ext))
    }
  }

  # Load the data
  x <- validate_data(input)$data

  if (verbose >= 1L) {
    is_sparse <- inherits(x, "dgCMatrix")
    nnz <- if (is_sparse) length(x@x) else sum(x != 0)
    message(sprintf("sp_convert: %d x %d matrix, nnz = %d (%.1f%% dense)",
                    nrow(x), ncol(x), nnz,
                    nnz / (as.double(nrow(x)) * ncol(x)) * 100))
  }

  # Ensure dgCMatrix
  if (!inherits(x, "dgCMatrix")) {
    x <- .to_dgCMatrix(x)
  }

  # Write to .spz
  output <- normalizePath(output, mustWork = FALSE)
  stats <- sp_write(x, output,
                    precision = precision,
                    row_sort = row_sort,
                    verbose = (verbose >= 2L))

  if (verbose >= 1L) {
    message(sprintf("sp_convert: written to %s (%.1fx compression)",
                    basename(output), stats$ratio))
  }

  # Extract sidecar metadata from h5ad if applicable
  if (input_format == "h5ad" && is.character(input)) {
    tryCatch({
      meta <- .extract_h5ad_metadata(input)
      if (length(meta) > 0) {
        meta_path <- paste0(output, ".meta")
        .write_sidecar(meta_path, meta)
        if (verbose >= 1L) {
          message(sprintf("sp_convert: sidecar metadata written to %s", basename(meta_path)))
        }
      }
    }, error = function(e) {
      if (verbose >= 1L) {
        message("sp_convert: note: could not extract h5ad metadata: ", conditionMessage(e))
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


# ---- Sidecar Metadata Helpers ----

#' Extract metadata from an h5ad file
#'
#' Pulls obs, var, and uns DataFrames from an AnnData .h5ad file
#' into a plain R list suitable for JSON serialization.
#'
#' @param path Path to .h5ad file
#' @return Named list with \code{obs}, \code{var}, and \code{uns} elements
#' @keywords internal
.extract_h5ad_metadata <- function(path) {
  if (!requireNamespace("hdf5r", quietly = TRUE)) {
    stop("Package 'hdf5r' is required for h5ad metadata extraction.")
  }

  h5 <- hdf5r::H5File$new(path, mode = "r")
  on.exit(h5$close_all(), add = TRUE)

  meta <- list()

  # Extract obs (cell-level metadata)
  if (h5$exists("obs")) {
    obs_group <- h5[["obs"]]
    obs_names <- obs_group$ls()$name
    meta$obs <- lapply(stats::setNames(obs_names, obs_names), function(nm) {
      tryCatch({
        dat <- obs_group[[nm]]
        if (inherits(dat, "H5D")) {
          val <- dat$read()
          # Convert factors stored as categorical
          if (dat$attr_exists("categories")) {
            cats <- dat[["categories"]]$read()
            val <- cats[val + 1L]
          }
          val
        } else {
          NULL
        }
      }, error = function(e) NULL)
    })
    meta$obs <- Filter(Negate(is.null), meta$obs)
  }

  # Extract var (gene-level metadata)
  if (h5$exists("var")) {
    var_group <- h5[["var"]]
    var_names <- var_group$ls()$name
    meta$var <- lapply(stats::setNames(var_names, var_names), function(nm) {
      tryCatch({
        dat <- var_group[[nm]]
        if (inherits(dat, "H5D")) {
          val <- dat$read()
          if (dat$attr_exists("categories")) {
            cats <- dat[["categories"]]$read()
            val <- cats[val + 1L]
          }
          val
        } else {
          NULL
        }
      }, error = function(e) NULL)
    })
    meta$var <- Filter(Negate(is.null), meta$var)
  }

  # Extract uns (unstructured metadata) — only scalar/string values

  if (h5$exists("uns")) {
    uns_group <- h5[["uns"]]
    uns_names <- uns_group$ls()$name
    meta$uns <- lapply(stats::setNames(uns_names, uns_names), function(nm) {
      tryCatch({
        dat <- uns_group[[nm]]
        if (inherits(dat, "H5D")) {
          val <- dat$read()
          if (length(val) <= 100) val else NULL
        } else {
          NULL
        }
      }, error = function(e) NULL)
    })
    meta$uns <- Filter(Negate(is.null), meta$uns)
  }

  meta
}


#' Write Sidecar Metadata File
#'
#' Writes a JSON sidecar file for a .spz file, preserving
#' metadata from the original format.
#'
#' @param path Path for the sidecar file (typically \code{<spz_path>.meta})
#' @param metadata Named list to serialize as JSON
#' @keywords internal
.write_sidecar <- function(path, metadata) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Package 'jsonlite' is required for sidecar metadata.")
  }
  json <- jsonlite::toJSON(metadata, auto_unbox = TRUE, pretty = TRUE,
                           na = "null", null = "null")
  writeLines(json, path)
  invisible(path)
}


#' Read Sidecar Metadata File
#'
#' Reads a JSON sidecar file written by \code{.write_sidecar()}.
#'
#' @param path Path to the sidecar \code{.meta} file
#' @return Named list of metadata, or \code{NULL} if file does not exist
#' @keywords internal
.read_sidecar <- function(path) {
  if (!file.exists(path)) return(NULL)
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Package 'jsonlite' is required for sidecar metadata.")
  }
  jsonlite::fromJSON(path, simplifyVector = TRUE, simplifyDataFrame = FALSE)
}
