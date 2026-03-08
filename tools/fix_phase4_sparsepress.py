#!/usr/bin/env python3
"""Phase 4: Convert R/sparsepress.R to thin deprecation wrappers."""
import os

SPARSEPRESS = "/mnt/home/debruinz/RcppML-2/R/sparsepress.R"

NEW_CONTENT = r'''#' @title SparsePress I/O (Deprecated)
#'
#' @description
#' These functions are deprecated. Use the \code{st_*} equivalents from
#' the \code{\link{streampress}} API instead.
#'
#' @name sparsepress-deprecated
#' @seealso \code{\link{streampress}}, \code{\link{st_write}}, \code{\link{st_read}},
#'   \code{\link{st_info}}, \code{\link{st_write_dense}}, \code{\link{st_read_dense}}
NULL

#' @rdname sparsepress-deprecated
#' @export
sp_write <- function(x, path, delta = TRUE, value_pred = FALSE, verbose = FALSE,
                     precision = "auto", row_sort = FALSE,
                     include_transpose = FALSE,
                     chunk_cols = 2048L) {
  .Deprecated("st_write", package = "RcppML",
              msg = "sp_write() is deprecated. Use st_write() instead.")
  st_write(x, path, delta = delta, value_pred = value_pred, verbose = verbose,
           precision = precision, row_sort = row_sort,
           include_transpose = include_transpose,
           chunk_cols = chunk_cols)
}

#' @rdname sparsepress-deprecated
#' @export
sp_read <- function(path, cols = NULL, reorder = TRUE) {
  .Deprecated("st_read", package = "RcppML",
              msg = "sp_read() is deprecated. Use st_read() instead.")
  st_read(path, cols = cols, reorder = reorder)
}

#' @rdname sparsepress-deprecated
#' @keywords internal
sp_read_transpose <- function(path) {
  .Deprecated("st_read_transpose", package = "RcppML")
  st_read_transpose(path)
}

#' @rdname sparsepress-deprecated
#' @export
sp_info <- function(path) {
  .Deprecated("st_info", package = "RcppML",
              msg = "sp_info() is deprecated. Use st_info() instead.")
  st_info(path)
}

#' @rdname sparsepress-deprecated
#' @keywords internal
sp_compress <- function(x, delta = TRUE, value_pred = FALSE) {
  .Deprecated("st_compress", package = "RcppML")
  if (!inherits(x, "dgCMatrix")) {
    x <- .to_dgCMatrix(x)
  }
  Rcpp_sp_compress(x, use_delta = delta, use_value_pred = value_pred)
}

#' @rdname sparsepress-deprecated
#' @keywords internal
sp_decompress <- function(x) {
  .Deprecated("st_decompress", package = "RcppML")
  if (!is.raw(x)) stop("'x' must be a raw vector")
  Rcpp_sp_decompress(x)
}

#' @rdname sparsepress-deprecated
#' @export
sp_write_dense <- function(x, path, include_transpose = FALSE,
                           chunk_cols = 2048L, codec = "raw",
                           delta = FALSE, verbose = FALSE) {
  .Deprecated("st_write_dense", package = "RcppML",
              msg = "sp_write_dense() is deprecated. Use st_write_dense() instead.")
  st_write_dense(x, path, include_transpose = include_transpose,
                 chunk_cols = chunk_cols, codec = codec,
                 delta = delta, verbose = verbose)
}

#' @rdname sparsepress-deprecated
#' @export
sp_read_dense <- function(path) {
  .Deprecated("st_read_dense", package = "RcppML",
              msg = "sp_read_dense() is deprecated. Use st_read_dense() instead.")
  st_read_dense(path)
}

#' @rdname sparsepress-deprecated
#' @keywords internal
sp_convert <- function(input, output, precision = "auto",
                       include_transpose = FALSE,
                       row_sort = TRUE,
                       verbose = 1L) {
  .Deprecated("st_convert", package = "RcppML")
  st_convert(input, output, precision = precision,
             include_transpose = include_transpose,
             row_sort = row_sort, verbose = verbose)
}


# ---- Internal Helpers (kept here for backwards compatibility) ----

# Internal helper: format bytes for display
.format_bytes <- function(bytes) {
  if (bytes < 1024) return(sprintf("%d B", bytes))
  if (bytes < 1024^2) return(sprintf("%.1f KB", bytes / 1024))
  if (bytes < 1024^3) return(sprintf("%.1f MB", bytes / 1024^2))
  sprintf("%.1f GB", bytes / 1024^3)
}


# ---- Sidecar Metadata Helpers ----

#' @keywords internal
.extract_h5ad_metadata <- function(path) {
  if (!requireNamespace("hdf5r", quietly = TRUE)) {
    stop("Package 'hdf5r' is required for h5ad metadata extraction.")
  }

  h5 <- hdf5r::H5File$new(path, mode = "r")
  on.exit(h5$close_all(), add = TRUE)

  meta <- list()

  if (h5$exists("obs")) {
    obs_group <- h5[["obs"]]
    obs_names <- obs_group$ls()$name
    meta$obs <- lapply(stats::setNames(obs_names, obs_names), function(nm) {
      tryCatch({
        dat <- obs_group[[nm]]
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
    meta$obs <- Filter(Negate(is.null), meta$obs)
  }

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

#' @keywords internal
.read_sidecar <- function(path) {
  if (!file.exists(path)) return(NULL)
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Package 'jsonlite' is required for sidecar metadata.")
  }
  jsonlite::fromJSON(path, simplifyVector = TRUE, simplifyDataFrame = FALSE)
}
'''

with open(SPARSEPRESS, 'w') as f:
    f.write(NEW_CONTENT)
print(f"[OK] Rewrote {SPARSEPRESS}")
