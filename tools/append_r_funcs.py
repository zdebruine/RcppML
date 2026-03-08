#!/usr/bin/env python3
"""Append new StreamPress R functions to R/streampress.R."""

filepath = "/mnt/home/debruinz/RcppML-2/R/streampress.R"

new_functions = r"""
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
#' @export
st_slice_cols <- function(path, cols, threads = 0L) {
  path <- normalizePath(path, mustWork = TRUE)
  Rcpp_sp_read(path, cols = as.integer(cols), reorder = TRUE)
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
#' @export
st_slice_rows <- function(path, rows, threads = 0L) {
  path <- normalizePath(path, mustWork = TRUE)
  rows <- as.integer(rows)
  # Read the transpose, then extract requested rows
  # Read transpose = CSC(A^T), columns of transpose = rows of A
  At <- Rcpp_sp_read_transpose(path)
  # At is CSC(A^T), so At[,rows] gives us the rows of A as columns
  # Then transpose back to get A[rows,]
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
  # TODO(python): For very large lists, implement streaming writer to avoid RAM peak
  combined <- do.call(cbind, mats)
  st_write(combined, path, obs = obs, var = var,
           chunk_bytes = chunk_bytes, chunk_cols = chunk_cols,
           include_transpose = include_transpose, precision = precision,
           threads = threads, verbose = verbose)
}
"""

with open(filepath, 'a') as f:
    f.write(new_functions)

print("Appended new R functions to streampress.R")
