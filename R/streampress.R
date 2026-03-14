#' @title StreamPress I/O: Read, Write, and Inspect Compressed Matrices
#'
#' @description
#' Functions for reading and writing matrices in StreamPress (.spz) format.
#' StreamPress achieves 5-10x compression over raw float32 CSC binary on
#' typical scRNA-seq sparse matrices using rANS entropy coding.  Beyond storage
#' savings, SPZ is also faster to read than raw CSC binary at any thread count:
#' the bottleneck in reading large sparse matrices is sparse object construction
#' (sorting indices, allocating R/Eigen structures), which SPZ parallelises
#' across independent chunks while raw CSC must perform sequentially.
#'
#' @details
#' StreamPress (.spz) supports two format versions:
#' \itemize{
#'   \item \strong{v2 (sparse)}: Column-oriented compressed CSC format.
#'     Lossless, 5-10x compression over raw float32 CSC. Self-describing header.
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
#' @param obs Optional data.frame of observation (row/cell) metadata.
#'   Must have \code{nrow(obs) == nrow(x)}.
#' @param var Optional data.frame of variable (column/gene) metadata.
#'   Must have \code{nrow(var) == ncol(x)}.
#' @param delta Logical; use density-based delta prediction for structure.
#'   Default \code{TRUE}.
#' @param value_pred Logical; use value prediction for integer-valued data.
#'   Default \code{FALSE}.
#' @param verbose Logical; print compression statistics. Default \code{FALSE}.
#' @param precision Value precision: \code{"auto"} (default), \code{"fp32"},
#'   \code{"fp16"}, \code{"quant8"}, \code{"fp64"}.
#' @param row_sort Logical; sort rows by nnz for better compression.
#' @param include_transpose Logical; store CSC(A^T) in the file. Default \code{TRUE}.
#' @param chunk_cols Integer or NULL; columns per chunk. If NULL, computed from
#'   \code{chunk_bytes}.
#' @param chunk_bytes Numeric; target bytes per chunk when \code{chunk_cols}
#'   is NULL. Default 8 MB, which yields ~50 columns per chunk for typical
#'   scRNA-seq matrices (~38 k rows). Smaller chunks create more parallel work
#'   during reading; larger chunks compress slightly better.
#' @param transp_chunk_cols Integer or NULL; columns per transpose chunk.
#' @param transp_chunk_bytes Numeric or NULL; target bytes per transpose chunk.
#' @param threads Integer; number of threads for parallel compression
#'   (0 = all available). Default 0.
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
st_write <- function(x, path, obs = NULL, var = NULL,
                     delta = TRUE, value_pred = FALSE, verbose = FALSE,
                     precision = "auto", row_sort = FALSE,
                     include_transpose = TRUE,
                     chunk_cols = NULL,
                     chunk_bytes = 8e6,
                     transp_chunk_cols = NULL,
                     transp_chunk_bytes = NULL,
                     threads = 0L) {
  if (!inherits(x, "dgCMatrix")) {
    if (inherits(x, "Matrix") || is.matrix(x)) {
      x <- .to_dgCMatrix(x)
    } else {
      stop("'x' must be a dgCMatrix or coercible to one")
    }
  }

  # Memory-driven chunk sizing (Phase 3)
  if (is.null(chunk_cols)) {
    bytes_per_col <- nrow(x) * 4L  # float32 = 4 bytes worst case
    chunk_cols <- max(1L, as.integer(floor(chunk_bytes / bytes_per_col)))
    chunk_cols <- min(chunk_cols, ncol(x))
  }
  chunk_cols <- as.integer(chunk_cols)

  # Transpose chunk sizing
  if (include_transpose && is.null(transp_chunk_cols)) {
    tb <- if (!is.null(transp_chunk_bytes)) transp_chunk_bytes else chunk_bytes
    bytes_per_row <- ncol(x) * 4L
    transp_chunk_cols <- max(1L, as.integer(floor(tb / bytes_per_row)))
    transp_chunk_cols <- min(transp_chunk_cols, nrow(x))
  }

  # Serialize obs/var tables if provided
  obs_raw <- NULL
  var_raw <- NULL
  if (!is.null(obs)) {
    stopifnot(is.data.frame(obs), nrow(obs) == nrow(x))
    obs_raw <- Rcpp_st_serialize_table(obs)
  }
  if (!is.null(var)) {
    stopifnot(is.data.frame(var), nrow(var) == ncol(x))
    var_raw <- Rcpp_st_serialize_table(var)
  }

  path <- normalizePath(path, mustWork = FALSE)
  stats <- Rcpp_sp_write(x, path,
                         use_delta = delta,
                         use_value_pred = value_pred,
                         verbose = verbose,
                         precision = precision,
                         row_sort = row_sort,
                         include_transpose = include_transpose,
                         chunk_cols = chunk_cols,
                         obs_raw = obs_raw,
                         var_raw = var_raw)

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
#' @param threads Integer or NULL; threads for parallel decompression.
#'   \code{NULL} (default) enables automatic selection: 1 thread for files
#'   smaller than 50 MB (where threading overhead exceeds the benefit), and
#'   all available threads for larger files. Use \code{0} to always request
#'   all available threads, or a positive integer to fix the count.
#'
#' @return A \code{dgCMatrix} sparse matrix with dimnames if available.
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
st_read <- function(path, cols = NULL, reorder = TRUE, threads = NULL) {
  path <- normalizePath(path, mustWork = TRUE)
  if (is.null(threads)) {
    fsize <- file.info(path)$size
    threads <- if (!is.na(fsize) && fsize < 50e6) 1L else 0L
  }
  Rcpp_sp_read(path, cols = cols, reorder = reorder, threads = as.integer(threads))
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
#' Reads only the header -- no decompression is performed.
#'
#' @param path Path to a \code{.spz} file.
#'
#' @return A list with fields including \code{rows}, \code{cols}, \code{nnz},
#'   \code{density_pct}, \code{file_bytes}, \code{raw_bytes}, \code{ratio},
#'   \code{version}, \code{has_obs}, \code{has_var}, \code{has_transpose},
#'   \code{transp_chunk_cols}.
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

# Internal helpers for h5ad metadata sidecar (requires hdf5r)
.extract_h5ad_metadata <- function(path) {
  if (!requireNamespace("hdf5r", quietly = TRUE)) return(list())
  h5 <- hdf5r::H5File$new(path, mode = "r")
  on.exit(h5$close_all(), add = TRUE)
  meta <- list()
  if (h5$exists("obs"))
    meta$obs_names <- tryCatch(h5[["obs/_index"]]$read(), error = function(e) NULL)
  if (h5$exists("var"))
    meta$var_names <- tryCatch(h5[["var/_index"]]$read(), error = function(e) NULL)
  meta
}

.write_sidecar <- function(path, meta) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    saveRDS(meta, path)
  } else {
    writeLines(jsonlite::toJSON(meta, auto_unbox = TRUE), path)
  }
}

#' Add Transpose Section to an Existing StreamPress File
#'
#' @param path Path to a \code{.spz} file.
#' @param verbose Logical; print progress. Default TRUE.
#' @return Invisibly returns the path.
#' @seealso \code{\link{st_write}}, \code{\link{st_read}}
#'
#' @examples
#' \donttest{
#' library(Matrix)
#' m <- rsparsematrix(50, 20, 0.3)
#' tmp <- tempfile(fileext = ".spz")
#' st_write(m, tmp, include_transpose = FALSE)
#' st_add_transpose(tmp)
#' info <- st_info(tmp)
#' unlink(tmp)
#' }
#'
#' @export
st_add_transpose <- function(path, verbose = TRUE) {
  path <- normalizePath(path, mustWork = TRUE)
  Rcpp_st_add_transpose(path, verbose = verbose)
  invisible(path)
}

# =============================================================================
# Obs/Var Table Read
# =============================================================================

#' Read Observation (Row) Metadata from a StreamPress File
#'
#' Reads the embedded obs table from a v2 \code{.spz} file without
#' decompressing the matrix data. Returns an empty data.frame if no obs
#' table was stored.
#'
#' @param path Path to a \code{.spz} file.
#' @return A \code{data.frame} with observation metadata, or an empty
#'   data.frame if no obs table is present.
#'
#' @seealso \code{\link{st_read_var}}, \code{\link{st_write}}
#' @examples
#' \dontrun{
#' obs <- st_read_obs("data.spz")
#' head(obs)
#' }
#'
#' @export
st_read_obs <- function(path) {
  path <- normalizePath(path, mustWork = TRUE)
  Rcpp_st_read_obs(path)
}

#' Read Variable (Column) Metadata from a StreamPress File
#'
#' Reads the embedded var table from a v2 \code{.spz} file without
#' decompressing the matrix data. Returns an empty data.frame if no var
#' table was stored.
#'
#' @param path Path to a \code{.spz} file.
#' @return A \code{data.frame} with variable metadata, or an empty
#'   data.frame if no var table is present.
#'
#' @seealso \code{\link{st_read_obs}}, \code{\link{st_write}}
#' @examples
#' \dontrun{
#' var <- st_read_var("data.spz")
#' head(var)
#' }
#'
#' @export
st_read_var <- function(path) {
  path <- normalizePath(path, mustWork = TRUE)
  Rcpp_st_read_var(path)
}

# =============================================================================
# Column and Row Slicing
# =============================================================================

#' Slice Columns from a StreamPress File
#'
#' Read a subset of columns from a \code{.spz} file without decompressing
#' the entire matrix.
#'
#' @param path Path to a \code{.spz} file.
#' @param cols Integer vector of column indices (1-indexed).
#' @param threads Integer; number of threads (0 = all available). Default 0.
#' @return A \code{dgCMatrix} sparse matrix.
#'
#' @seealso \code{\link{st_slice_rows}}, \code{\link{st_slice}}
#' @examples
#' \dontrun{
#' mat <- st_slice_cols("data.spz", cols = 1:10)
#' dim(mat)
#' }
#'
#' @export
st_slice_cols <- function(path, cols, threads = 0L) {
  path <- normalizePath(path, mustWork = TRUE)
  cols <- as.integer(cols)
  A <- Rcpp_sp_read(path, reorder = TRUE)
  A[, cols, drop = FALSE]
}

#' Slice Rows from a StreamPress File
#'
#' Read a subset of rows from a \code{.spz} file using the pre-stored
#' transpose section. Requires that the file was written with
#' \code{include_transpose = TRUE}.
#'
#' @param path Path to a \code{.spz} file.
#' @param rows Integer vector of row indices (1-indexed).
#' @param threads Integer; number of threads (0 = all available). Default 0.
#' @return A \code{dgCMatrix} sparse matrix.
#'
#' @seealso \code{\link{st_slice_cols}}, \code{\link{st_slice}}
#' @examples
#' \dontrun{
#' mat <- st_slice_rows("data.spz", rows = 1:100)
#' dim(mat)
#' }
#'
#' @export
st_slice_rows <- function(path, rows, threads = 0L) {
  path <- normalizePath(path, mustWork = TRUE)
  rows <- as.integer(rows)
  # Read the transpose = CSC(A^T), columns of transpose = rows of A
  At <- Rcpp_sp_read_transpose(path)
  sub_t <- At[, rows, drop = FALSE]
  Matrix::t(sub_t)
}

#' Slice Rows and/or Columns from a StreamPress File
#'
#' Convenience function combining row and column slicing.
#'
#' @param path Path to a \code{.spz} file.
#' @param rows Optional integer vector of row indices (1-indexed).
#' @param cols Optional integer vector of column indices (1-indexed).
#' @param threads Integer; number of threads (0 = all available). Default 0.
#' @return A \code{dgCMatrix} sparse matrix.
#'
#' @seealso \code{\link{st_slice_cols}}, \code{\link{st_slice_rows}}
#' @examples
#' \dontrun{
#' mat <- st_slice("data.spz", rows = 1:100, cols = 1:10)
#' dim(mat)
#' }
#'
#' @export
st_slice <- function(path, rows = NULL, cols = NULL, threads = 0L) {
  path <- normalizePath(path, mustWork = TRUE)
  if (!is.null(rows) && !is.null(cols)) {
    A <- st_slice_cols(path, cols, threads = threads)
    return(A[rows, , drop = FALSE])
  } else if (!is.null(rows)) {
    return(st_slice_rows(path, rows, threads = threads))
  } else if (!is.null(cols)) {
    return(st_slice_cols(path, cols, threads = threads))
  } else {
    return(st_read(path, threads = threads))
  }
}

# =============================================================================
# Chunk Iteration
# =============================================================================

#' Get Column Ranges for Each Chunk in a StreamPress File
#'
#' Returns the column ranges (1-indexed, inclusive) for each chunk without
#' decompressing any data.
#'
#' @param path Path to a \code{.spz} file.
#' @return A \code{data.frame} with columns \code{start} and \code{end}.
#'
#' @seealso \code{\link{st_map_chunks}}
#' @examples
#' \dontrun{
#' ranges <- st_chunk_ranges("data.spz")
#' print(ranges)
#' }
#'
#' @export
st_chunk_ranges <- function(path) {
  info <- st_info(path)
  chunk_cols <- info$chunk_cols
  n <- info$cols
  starts <- seq(1L, n, by = chunk_cols)
  ends <- pmin(starts + chunk_cols - 1L, n)
  data.frame(start = starts, end = ends)
}

#' Apply a Function to Every Chunk in a StreamPress File
#'
#' Sequentially reads and decodes each chunk, applies \code{fn}, and
#' collects results.
#'
#' @param path Path to a \code{.spz} file.
#' @param fn Function taking \code{(chunk, col_start, col_end)} where
#'   \code{chunk} is a \code{dgCMatrix}.
#' @param transpose Logical; if \code{TRUE}, iterate over transpose chunks
#'   (row chunks of the original matrix). Default \code{FALSE}.
#' @param threads Integer; decode threads per chunk. Default 0 (all threads).
#' @return Invisible list of results from \code{fn}.
#'
#' @seealso \code{\link{st_chunk_ranges}}
#' @examples
#' \dontrun{
#' # Compute column sums per chunk
#' st_map_chunks("data.spz", function(chunk, s, e) Matrix::colSums(chunk))
#' }
#'
#' @export
st_map_chunks <- function(path, fn, transpose = FALSE, threads = 0L) {
  if (!transpose) {
    ranges <- st_chunk_ranges(path)
    results <- vector("list", nrow(ranges))
    for (i in seq_len(nrow(ranges))) {
      chunk <- st_slice_cols(path, ranges$start[i]:ranges$end[i], threads = threads)
      results[[i]] <- fn(chunk, ranges$start[i], ranges$end[i])
    }
  } else {
    info <- st_info(path)
    tc <- info$transp_chunk_cols
    if (is.null(tc) || tc == 0L) tc <- info$chunk_cols
    row_starts <- seq(1L, info$rows, by = tc)
    row_ends <- pmin(row_starts + tc - 1L, info$rows)
    results <- vector("list", length(row_starts))
    for (i in seq_along(row_starts)) {
      chunk <- st_slice_rows(path, row_starts[i]:row_ends[i], threads = threads)
      results[[i]] <- fn(chunk, row_starts[i], row_ends[i])
    }
  }
  invisible(results)
}

# =============================================================================
# Metadata-Based Filtering
# =============================================================================

#' Get Row Indices Matching Observation Metadata Filter
#'
#' Reads the obs table, applies a filter expression via \code{subset()},
#' and returns matching row indices.
#'
#' @param path Path to a \code{.spz} file.
#' @param ... Filter expressions passed to \code{subset()} on the obs table.
#' @return Integer vector of matching row indices (1-based).
#'
#' @seealso \code{\link{st_filter_rows}}, \code{\link{st_read_obs}}
#' @examples
#' \dontrun{
#' idx <- st_obs_indices("data.spz", cell_type == "B cell")
#' length(idx)
#' }
#'
#' @export
st_obs_indices <- function(path, ...) {
  obs <- st_read_obs(path)
  if (nrow(obs) == 0L) stop("File has no obs table")
  matched <- subset(obs, ...)
  which(seq_len(nrow(obs)) %in% as.integer(rownames(matched)))
}

#' Slice Rows Matching Observation Metadata Filter
#'
#' @param path Path to a \code{.spz} file.
#' @param ... Filter expression on obs columns (passed to \code{subset()}).
#' @param threads Integer decode threads. 0 = all.
#' @return A \code{dgCMatrix} sparse matrix.
#'
#' @seealso \code{\link{st_filter_cols}}, \code{\link{st_obs_indices}}
#' @examples
#' \dontrun{
#' mat <- st_filter_rows("data.spz", cell_type == "B cell")
#' dim(mat)
#' }
#'
#' @export
st_filter_rows <- function(path, ..., threads = 0L) {
  idx <- st_obs_indices(path, ...)
  if (length(idx) == 0L) stop("No rows match filter criteria")
  st_slice_rows(path, idx, threads = threads)
}

#' Slice Columns Matching Variable Metadata Filter
#'
#' @param path Path to a \code{.spz} file.
#' @param ... Filter expression on var columns (passed to \code{subset()}).
#' @param threads Integer decode threads. 0 = all.
#' @return A \code{dgCMatrix} sparse matrix.
#'
#' @seealso \code{\link{st_filter_rows}}, \code{\link{st_read_var}}
#' @examples
#' \dontrun{
#' mat <- st_filter_cols("data.spz", highly_variable == TRUE)
#' dim(mat)
#' }
#'
#' @export
st_filter_cols <- function(path, ..., threads = 0L) {
  var_df <- st_read_var(path)
  if (nrow(var_df) == 0L) stop("File has no var table")
  matched <- subset(var_df, ...)
  idx <- as.integer(rownames(matched))
  if (length(idx) == 0L) stop("No columns match filter criteria")
  st_slice_cols(path, idx, threads = threads)
}

# =============================================================================
# List Assembly (Streaming cbind Write)
# =============================================================================

#' Write a List of Matrices as a Single StreamPress File
#'
#' Column-concatenates a list of matrices and writes them as a single
#' \code{.spz} file. All matrices must have the same number of rows.
#'
#' @param x A list of \code{dgCMatrix} objects (or coercible).
#'   All must have identical \code{nrow}.
#' @param path Output \code{.spz} path.
#' @param obs Optional data.frame of cell metadata (\code{nrow} == total cols).
#' @param var Optional data.frame of gene metadata (\code{nrow} == nrow of mats).
#' @param chunk_bytes Target bytes per chunk. Default 64 MB.
#' @param chunk_cols Explicit column count per chunk. Overrides \code{chunk_bytes}.
#' @param include_transpose Logical. Default \code{TRUE}.
#' @param precision Value precision. Default \code{"auto"}.
#' @param threads Integer. 0 = all threads.
#' @param verbose Logical.
#' @return Invisibly, compression statistics.
#'
#' @seealso \code{\link{st_write}}
#' @examples
#' \dontrun{
#' mats <- list(mat1, mat2)
#' st_write_list(mats, "combined.spz")
#' }
#'
#' @export
st_write_list <- function(x, path, obs = NULL, var = NULL,
                          chunk_bytes = 64e6, chunk_cols = NULL,
                          include_transpose = TRUE, precision = "auto",
                          threads = 0L, verbose = FALSE) {
  stopifnot(is.list(x), length(x) >= 1L)
  mats <- lapply(x, function(m) {
    if (!inherits(m, "dgCMatrix")) m <- as(m, "dgCMatrix")
    m
  })
  nr <- nrow(mats[[1]])
  if (!all(vapply(mats, nrow, integer(1)) == nr))
    stop("All matrices must have the same nrow")
  combined <- do.call(cbind, mats)
  st_write(combined, path, obs = obs, var = var,
           chunk_bytes = chunk_bytes, chunk_cols = chunk_cols,
           include_transpose = include_transpose, precision = precision,
           threads = threads, verbose = verbose)
}

# Internal: auto-dispatch for StreamPress file input.
# Determines whether to use in-core or streaming mode based on available
# memory. Currently defaults to in-core CPU (streaming can be forced via
# streaming=TRUE in nmf()).
.st_dispatch <- function(path, k, resource = "auto") {
  list(mode = "IN_CORE_CPU", resource = "cpu", streaming = FALSE)
}

# Internal helper: format bytes for display
.format_bytes <- function(bytes) {
  if (bytes < 1024) return(sprintf("%d B", bytes))
  if (bytes < 1024^2) return(sprintf("%.1f KB", bytes / 1024))
  if (bytes < 1024^3) return(sprintf("%.1f MB", bytes / 1024^2))
  sprintf("%.1f GB", bytes / 1024^3)
}
