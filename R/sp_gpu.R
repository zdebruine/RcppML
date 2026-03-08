#' @title Read SparsePress File Directly to GPU Memory
#'
#' @description
#' Reads a \code{.spz} v2 file and decodes it directly on the GPU, returning
#' an opaque GPU-resident CSC matrix. This avoids the CPU-to-GPU transfer that
#' occurs when using \code{sp_read()} followed by \code{nmf(data, gpu = TRUE)}.
#'
#' The returned object is an external reference — the matrix data lives
#' entirely in GPU device memory. Pass it directly to \code{nmf()} for
#' zero-copy GPU NMF.
#'
#' @param path Path to a \code{.spz} file (v2 format required).
#' @param device Integer; CUDA device ID (default 0).
#'
#' @return An object of class \code{"gpu_sparse_matrix"} with fields:
#'   \describe{
#'     \item{m}{Number of rows}
#'     \item{n}{Number of columns}
#'     \item{nnz}{Number of non-zeros}
#'     \item{device}{CUDA device ID}
#'     \item{.col_ptr}{Opaque device pointer (numeric)}
#'     \item{.row_idx}{Opaque device pointer (numeric)}
#'     \item{.values}{Opaque device pointer (numeric)}
#'   }
#'
#' @details
#' Only \code{.spz} v2 format is supported for GPU decode. Use
#' \code{sp_convert()} to convert other formats to v2.
#'
#' The returned object has a finalizer that automatically frees GPU memory
#' when the R object is garbage-collected. You can also free it manually
#' with \code{sp_free_gpu()}.
#'
#' @examples
#' \dontrun{
#' # Read directly to GPU
#' gpu_data <- sp_read_gpu("data.spz")
#'
#' # Run NMF on GPU-resident data (zero-copy)
#' result <- nmf(gpu_data, k = 10)
#'
#' # Clean up (optional — GC will do this automatically)
#' sp_free_gpu(gpu_data)
#' }
#'
#' @seealso \code{\link{sp_read}}, \code{\link{sp_free_gpu}}, \code{\link{nmf}}
#' @export
sp_read_gpu <- function(path, device = 0L) {
  if (!gpu_available()) {
    stop("No GPU available. Use sp_read() for CPU decoding.", call. = FALSE)
  }

  path <- normalizePath(path, mustWork = TRUE)
  device <- as.integer(device)

  result <- .gpu_call("rcppml_sp_read_gpu",
    path_ptr = path,
    device_id = device,
    out_col_ptr_addr = double(1),
    out_row_idx_addr = double(1),
    out_values_addr  = double(1),
    out_m = integer(1),
    out_n = integer(1),
    out_nnz = double(1),
    out_status = integer(1))

  if (result$out_status != 0L) {
    stop(sprintf("GPU decode failed with status %d. Ensure file is .spz v2 format.",
                 result$out_status), call. = FALSE)
  }

  obj <- structure(
    list(
      m = result$out_m,
      n = result$out_n,
      nnz = result$out_nnz,
      device = device,
      .col_ptr = result$out_col_ptr_addr,
      .row_idx = result$out_row_idx_addr,
      .values  = result$out_values_addr
    ),
    class = "gpu_sparse_matrix"
  )

  # Register finalizer to free GPU memory on GC.
  # Use an environment to ensure the finalizer captures a strong reference
  # to the gpu_sparse_matrix object's pointer addresses.
  prevent_gc <- new.env(parent = emptyenv())
  prevent_gc$col_ptr <- obj$.col_ptr
  prevent_gc$row_idx <- obj$.row_idx
  prevent_gc$values  <- obj$.values
  reg.finalizer(prevent_gc, function(e) {
    if (!is.null(e$col_ptr) && e$col_ptr != 0) {
      tryCatch({
        dummy <- structure(
          list(.col_ptr = e$col_ptr, .row_idx = e$row_idx, .values = e$values),
          class = "gpu_sparse_matrix"
        )
        sp_free_gpu(dummy)
      }, error = function(err) NULL)
    }
  }, onexit = TRUE)
  # Keep the prevent_gc environment alive as long as obj lives
  obj$.prevent_gc <- prevent_gc

  obj
}

#' @title Free GPU-Resident Sparse Matrix
#'
#' @description
#' Explicitly frees CUDA device memory held by a \code{gpu_sparse_matrix} object.
#' This is optional — the memory will be freed automatically when the object
#' is garbage-collected.
#'
#' @param x A \code{gpu_sparse_matrix} object from \code{sp_read_gpu()}.
#' @return Invisibly returns \code{NULL}.
#'
#' @seealso \code{\link{sp_read_gpu}}
#' @export
#' @examples
#' \dontrun{
#' gpu_mat <- sp_read_gpu("matrix.spz")
#' sp_free_gpu(gpu_mat)
#' }
sp_free_gpu <- function(x) {
  if (!inherits(x, "gpu_sparse_matrix")) {
    stop("'x' must be a gpu_sparse_matrix object", call. = FALSE)
  }
  if (!gpu_available()) return(invisible(NULL))

  .gpu_call("rcppml_sp_free_gpu",
     col_ptr_addr = as.double(x$.col_ptr),
     row_idx_addr = as.double(x$.row_idx),
     values_addr  = as.double(x$.values),
     out_status = integer(1))

  # Modify the caller's variable in-place (R lists are copy-on-modify,

  # so we must write back to the caller's environment)
  var_name <- deparse(substitute(x))
  env <- parent.frame()
  if (exists(var_name, envir = env, inherits = FALSE)) {
    obj <- env[[var_name]]
    obj$.col_ptr <- 0
    obj$.row_idx <- 0
    obj$.values  <- 0
    env[[var_name]] <- obj
  }

  invisible(NULL)
}

#' Methods for gpu_sparse_matrix objects
#'
#' S3 methods for the \code{gpu_sparse_matrix} class returned by
#' \code{\link{sp_read_gpu}}.
#'
#' @param x a \code{gpu_sparse_matrix} object
#' @param ... additional arguments (unused)
#' @return
#' \describe{
#'   \item{\code{print}}{Invisibly returns \code{x}, prints summary to console.}
#'   \item{\code{dim}}{Integer vector of length 2: \code{c(nrow, ncol)}.}
#'   \item{\code{nrow}}{Number of rows (integer).}
#'   \item{\code{ncol}}{Number of columns (integer).}
#' }
#' @seealso \code{\link{sp_read_gpu}}, \code{\link{sp_free_gpu}}
#' @examples
#' \dontrun{
#' gpu_mat <- sp_read_gpu("data.spz")
#' print(gpu_mat)
#' dim(gpu_mat)
#' nrow(gpu_mat)
#' ncol(gpu_mat)
#' sp_free_gpu(gpu_mat)
#' }
#' @name gpu_sparse_matrix-methods
NULL

#' @rdname gpu_sparse_matrix-methods
#' @method print gpu_sparse_matrix
#' @export
print.gpu_sparse_matrix <- function(x, ...) {
  cat(sprintf("GPU Sparse Matrix (%d x %d, nnz = %.0f)\n",
              x$m, x$n, x$nnz))
  cat(sprintf("  Device: GPU %d\n", x$device))
  cat(sprintf("  Memory: ~%.1f MB on device\n",
              (x$nnz * 8 + (x$n + 1) * 4) / 1024^2))
  invisible(x)
}

#' @rdname gpu_sparse_matrix-methods
#' @method dim gpu_sparse_matrix
#' @export
dim.gpu_sparse_matrix <- function(x) {
  c(x$m, x$n)
}

#' @rdname gpu_sparse_matrix-methods
#' @method nrow gpu_sparse_matrix
#' @export
nrow.gpu_sparse_matrix <- function(x) x$m

#' @rdname gpu_sparse_matrix-methods
#' @method ncol gpu_sparse_matrix
#' @export
ncol.gpu_sparse_matrix <- function(x) x$n
