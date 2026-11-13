#' @title GPU NMF Backend
#'
#' @description Internal functions for GPU-accelerated NMF via CUDA.
#' Automatically detects GPU availability and dispatches to GPU when
#' beneficial, falling back to CPU otherwise.
#'
#' @name gpu-backend
#' @keywords internal
NULL

# Package-level GPU state (initialized once per session)
.gpu_env <- new.env(parent = emptyenv())
.gpu_env$available <- NA       # NA = not checked, TRUE/FALSE after check
.gpu_env$lib_loaded <- FALSE
.gpu_env$dll <- NULL
.gpu_env$lib_path <- NA_character_
.gpu_env$abi_found <- NA_integer_
.gpu_env$reason <- "not checked yet"
.gpu_env$num_gpus <- 0L
.gpu_env$gpu_names <- character(0)
.gpu_env$total_mem_mb <- numeric(0)
.gpu_env$free_mem_mb <- numeric(0)

# Interface version this R package expects from RcppML_gpu.so. It must equal
# RCPPML_GPU_ABI_VERSION in gpu_bridge_common.cuh; bump both together whenever
# the signature of any bridge entry point changes.
.GPU_ABI_VERSION <- 2L

# Bridge translation units compiled into RcppML_gpu.so.
.gpu_sources <- c("gpu_bridge_cluster.cu", "gpu_bridge_nmf.cu",
                  "gpu_bridge_svd.cu", "gpu_bridge_utils.cu",
                  "gpu_bridge_assess.cu")

# Helper: call GPU bridge .C() with dynamic symbol name.
# Uses do.call() to prevent Rcpp::compileAttributes() from scanning the
# symbol name and auto-registering it in the main package DLL.
# We resolve the symbol via getNativeSymbolInfo() to pin to the GPU DLL.
.gpu_call <- function(sym, ...) {
  sym_info <- getNativeSymbolInfo(paste0(sym), .gpu_env$dll)
  do.call(.C, c(list(sym_info), list(...)))
}

# Helper: extract CSC arrays from a dgCMatrix for .C() bridge.
# Returns a named list of (p, i, x, dim, nnz, lambda) suitable for
# splicing into .gpu_call() arguments. If mat is NULL, returns dummy
# arrays with nnz = 0 (sentinel) so the C code skips reconstruction.
.extract_sparse_csc <- function(mat, lambda = 0, symmetric = TRUE) {
  if (is.null(mat)) {
    list(p = as.integer(0L), i = as.integer(0L), x = as.double(0),
         dim = as.integer(0L), nnz = as.integer(0L), lambda = as.double(lambda))
  } else {
    mat <- as(mat, "dgCMatrix")
    list(p = as.integer(mat@p), i = as.integer(mat@i), x = as.double(mat@x),
         dim = as.integer(nrow(mat)), nnz = as.integer(length(mat@x)),
         lambda = as.double(lambda))
  }
}

# Like .extract_sparse_csc but for obs_mask where rows/cols may differ
.extract_sparse_csc_rect <- function(mat) {
  if (is.null(mat)) {
    list(p = as.integer(0L), i = as.integer(0L), x = as.double(0),
         rows = as.integer(0L), cols = as.integer(0L), nnz = as.integer(0L))
  } else {
    mat <- as(mat, "dgCMatrix")
    list(p = as.integer(mat@p), i = as.integer(mat@i), x = as.double(mat@x),
         rows = as.integer(nrow(mat)), cols = as.integer(ncol(mat)),
         nnz = as.integer(length(mat@x)))
  }
}

# File name of the GPU shared library on this platform.
.gpu_lib_name <- function() paste0("RcppML_gpu", .Platform$dynlib.ext)

# Per-user directory that install_gpu() writes the library to.
.gpu_user_dir <- function() tools::R_user_dir("RcppML", "cache")

# Places the GPU library may live, in search order. Only the installed
# package and the per-user cache are searched; the working directory never is.
.gpu_lib_candidates <- function() {
  lib_name <- .gpu_lib_name()
  pkg <- system.file("lib", lib_name, package = "RcppML")
  out <- c(if (nzchar(pkg)) pkg, file.path(.gpu_user_dir(), lib_name))
  unique(out)
}

#' Find GPU shared library
#' @return File path to the GPU shared library, or \code{NULL}.
#' @keywords internal
.find_gpu_lib <- function() {
  for (path in .gpu_lib_candidates()) {
    if (nzchar(path) && file.exists(path)) return(path)
  }
  NULL
}

# Read the interface version from a loaded GPU library (NA if the library
# predates the version symbol).
.gpu_lib_abi <- function(dll) {
  tryCatch({
    sym <- getNativeSymbolInfo("rcppml_gpu_abi_version", dll)
    as.integer(do.call(.C, list(sym, version = integer(1)))$version)
  }, error = function(e) NA_integer_)
}

# Load the first candidate library whose interface version matches.
# Sets .gpu_env$lib_loaded, $dll, $lib_path, $abi_found and $reason.
.gpu_load_lib <- function() {
  if (.gpu_env$lib_loaded) return(TRUE)
  cands <- .gpu_lib_candidates()
  cands <- cands[file.exists(cands)]
  if (length(cands) == 0L) {
    .gpu_env$reason <- paste0(
      "GPU library '", .gpu_lib_name(), "' not found; run RcppML::install_gpu()")
    return(FALSE)
  }
  for (path in cands) {
    # local = FALSE makes symbols globally visible so the C++ gateway can
    # find them via dlsym(RTLD_DEFAULT, ...)
    dll <- tryCatch(dyn.load(path, local = FALSE), error = function(e) e)
    if (inherits(dll, "error")) {
      .gpu_env$reason <- paste0("could not load '", path, "': ",
                                conditionMessage(dll))
      next
    }
    abi <- .gpu_lib_abi(dll)
    .gpu_env$abi_found <- abi
    if (is.na(abi) || abi != .GPU_ABI_VERSION) {
      try(dyn.unload(path), silent = TRUE)
      .gpu_env$reason <- paste0(
        "GPU library '", path, "' has interface version ",
        if (is.na(abi)) "<none>" else abi, " but this RcppML needs ",
        .GPU_ABI_VERSION, "; rebuild it with RcppML::install_gpu()")
      next
    }
    .gpu_env$dll <- dll
    .gpu_env$lib_path <- path
    .gpu_env$lib_loaded <- TRUE
    .gpu_env$reason <- "library loaded"
    return(TRUE)
  }
  FALSE
}

#' Check if GPU acceleration is available
#'
#' Detects CUDA GPUs at runtime. Result is cached for the session.
#' The GPU library is looked for in the installed package and in the per-user
#' directory written by \code{\link{install_gpu}}; when this returns
#' \code{FALSE}, \code{\link{gpu_status}} reports which step failed.
#'
#' @param force_recheck Re-probe GPUs even if already cached
#' @return logical TRUE if GPU is available
#' @seealso \code{\link{gpu_info}}, \code{\link{gpu_status}},
#'   \code{\link{install_gpu}}, \code{\link{nmf}}
#' @export
#' @examples
#' gpu_available()
gpu_available <- function(force_recheck = FALSE) {
  if (!is.na(.gpu_env$available) && !force_recheck) {
    return(.gpu_env$available)
  }

  .gpu_env$available <- FALSE
  .gpu_env$num_gpus <- 0L

  # If the library is already loaded, force_recheck only re-probes devices
  if (!.gpu_load_lib()) return(FALSE)

  # Probe GPUs via C bridge (all args are pointers per .C() convention)
  tryCatch({
    result <- .gpu_call("rcppml_gpu_detect",
                  num_gpus = integer(1),
                  total_mem_mb = double(8),
                  free_mem_mb = double(8),
                  max_gpus = as.integer(8L),
                  out_status = integer(1))

    if (result$out_status == 0L && result$num_gpus > 0L) {
      ng <- min(result$num_gpus, 8L)
      .gpu_env$available <- TRUE
      .gpu_env$num_gpus <- ng
      .gpu_env$total_mem_mb <- result$total_mem_mb[seq_len(ng)]
      .gpu_env$free_mem_mb <- result$free_mem_mb[seq_len(ng)]
      # GPU names not returned via .C() (char* is read-only in .C()),
      # so we just report generic names based on count
      .gpu_env$gpu_names <- paste0("GPU ", seq_len(ng) - 1L)
      .gpu_env$reason <- "ok"
    } else {
      .gpu_env$reason <- paste0(
        "library loaded, but no usable CUDA device was found ",
        "(needs a working driver and compute capability >= 7.0)")
    }
  }, error = function(e) {
    .gpu_env$available <- FALSE
    .gpu_env$reason <- paste0("device probe failed: ", conditionMessage(e))
  })

  return(.gpu_env$available)
}

#' Get GPU device information
#'
#' @return data.frame with GPU device details, or NULL if no GPU
#' @seealso \code{\link{gpu_available}}, \code{\link{nmf}}
#' @export
#' @examples
#' gpu_info()
gpu_info <- function() {
  if (!gpu_available()) return(NULL)

  data.frame(
    device = seq_len(.gpu_env$num_gpus) - 1L,
    name = .gpu_env$gpu_names,
    total_mb = .gpu_env$total_mem_mb,
    free_mb = .gpu_env$free_mem_mb,
    stringsAsFactors = FALSE
  )
}

# Locate nvcc: CUDA_HOME / CUDA_PATH first, then the PATH. "" if not found.
.find_nvcc <- function() {
  for (v in c("CUDA_HOME", "CUDA_PATH")) {
    home <- Sys.getenv(v, "")
    if (nzchar(home)) {
      cand <- file.path(home, "bin", "nvcc")
      if (file.exists(cand)) return(cand)
    }
  }
  unname(Sys.which("nvcc"))
}

# Release number from `nvcc --version`, or NA.
.nvcc_version <- function(nvcc) {
  out <- tryCatch(
    suppressWarnings(system2(nvcc, "--version", stdout = TRUE, stderr = TRUE)),
    error = function(e) character(0))
  hit <- grep("release", out, value = TRUE)
  if (length(hit) == 0L) return(NA_character_)
  sub(".*release ([0-9.]+).*", "\\1", hit[1])
}

# -gencode flags for the build. `arch` is a character vector such as
# c("sm_70", "sm_90"); NULL asks nvidia-smi for the cards in this machine and
# otherwise falls back to a list covering Volta through Hopper, filtered to
# what this nvcc can target.
.gpu_gencode_flags <- function(nvcc, arch = NULL) {
  if (is.null(arch)) {
    smi <- unname(Sys.which("nvidia-smi"))
    if (nzchar(smi)) {
      cc <- tryCatch(
        suppressWarnings(system2(smi,
          c("--query-gpu=compute_cap", "--format=csv,noheader"),
          stdout = TRUE, stderr = FALSE)),
        error = function(e) character(0))
      cc <- gsub("[^0-9]", "", cc)
      cc <- unique(cc[nzchar(cc)])
      if (length(cc) > 0L) arch <- paste0("sm_", cc)
    }
  }
  if (is.null(arch)) {
    arch <- c("sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90")
    supported <- tryCatch(
      suppressWarnings(system2(nvcc, "--list-gpu-arch",
                               stdout = TRUE, stderr = FALSE)),
      error = function(e) character(0))
    supported <- grep("^sm_[0-9]+$", trimws(supported), value = TRUE)
    if (length(supported) > 0L) {
      keep <- intersect(arch, supported)
      if (length(keep) > 0L) arch <- keep
    }
  }
  arch <- unique(sub("^compute_", "sm_", arch))
  if (!all(grepl("^sm_[0-9]+$", arch)))
    stop("'arch' must look like \"sm_80\"", call. = FALSE)
  num <- sub("^sm_", "", arch)
  num <- num[order(as.integer(num))]
  # Native code for each target, plus PTX for the oldest one so that a newer
  # card than the build machine's can still JIT-compile the kernels.
  c(as.vector(rbind("-gencode",
                    paste0("arch=compute_", num, ",code=sm_", num))),
    "-gencode", paste0("arch=compute_", num[1], ",code=compute_", num[1]))
}

#' Build and install the GPU library
#'
#' Compiles the CUDA bridge sources shipped with the package into
#' \code{RcppML_gpu.so} and writes it to the per-user directory
#' \code{tools::R_user_dir("RcppML", "cache")}, where
#' \code{\link{gpu_available}} looks for it. A normal package install does not
#' build this library, so run \code{install_gpu()} once after installing or
#' updating RcppML on a machine with the CUDA Toolkit. The build takes a few
#' minutes.
#'
#' @param nvcc Path to \code{nvcc}. Default: \code{$CUDA_HOME/bin/nvcc},
#'   \code{$CUDA_PATH/bin/nvcc}, or the first \code{nvcc} on the \code{PATH}.
#' @param arch Character vector of target architectures such as
#'   \code{"sm_70"} or \code{c("sm_80", "sm_90")}. Default: the compute
#'   capability of the installed cards as reported by \code{nvidia-smi}, or a
#'   multi-architecture build covering Volta through Hopper when
#'   \code{nvidia-smi} is unavailable (for example on a login node).
#' @param src_dir Directory holding the \code{gpu_bridge_*.cu} sources.
#'   Default: the \code{gpu} directory of the installed package.
#' @param include_dir Directory holding the \code{FactorNet} headers. Default:
#'   the \code{include} directory of the installed package.
#' @param flags Extra flags passed to \code{nvcc} when compiling.
#' @param quiet If \code{TRUE}, hide compiler output.
#' @return Invisibly, the path of the installed library.
#' @seealso \code{\link{gpu_status}}, \code{\link{gpu_available}}
#' @export
#' @examples
#' \dontrun{
#' install_gpu()
#' gpu_status()
#' }
install_gpu <- function(nvcc = NULL, arch = NULL, src_dir = NULL,
                        include_dir = NULL, flags = character(0),
                        quiet = FALSE) {
  if (.Platform$OS.type == "windows")
    stop("install_gpu() is supported on Linux only.", call. = FALSE)

  if (is.null(nvcc)) nvcc <- .find_nvcc()
  if (!nzchar(nvcc) || !file.exists(nvcc))
    stop("nvcc not found. Install the CUDA Toolkit and put nvcc on the PATH, ",
         "set CUDA_HOME, or pass nvcc = \"/path/to/nvcc\".", call. = FALSE)

  if (is.null(src_dir)) src_dir <- system.file("gpu", package = "RcppML")
  srcs <- file.path(src_dir, .gpu_sources)
  if (!nzchar(src_dir) || !all(file.exists(srcs)))
    stop("GPU bridge sources not found",
         if (nzchar(src_dir)) paste0(" in '", src_dir, "'"),
         ". Pass src_dir = the directory that holds gpu_bridge_nmf.cu ",
         "(for example the 'src' directory of an RcppML source checkout).",
         call. = FALSE)

  if (is.null(include_dir))
    include_dir <- system.file("include", package = "RcppML")
  if (!file.exists(file.path(include_dir, "FactorNet")))
    stop("FactorNet headers not found in '", include_dir, "'.", call. = FALSE)
  eigen_inc <- system.file("include", package = "RcppEigen")
  if (!nzchar(eigen_inc))
    stop("install_gpu() needs the RcppEigen package for its Eigen headers.",
         call. = FALSE)

  gencode <- .gpu_gencode_flags(nvcc, arch)

  build_dir <- tempfile("RcppML_gpu_build_")
  dir.create(build_dir)
  on.exit(unlink(build_dir, recursive = TRUE), add = TRUE)

  out_stream <- if (quiet) FALSE else ""
  compile_flags <- c(
    "-O3", gencode, "-std=c++17", "-Xcompiler", "-fPIC",
    "--expt-relaxed-constexpr", "-DNDEBUG",
    paste0("-I", shQuote(eigen_inc)),
    paste0("-I", shQuote(include_dir)),
    paste0("-I", shQuote(src_dir)),
    flags)

  objs <- character(0)
  for (src in srcs) {
    obj <- file.path(build_dir, sub("\\.cu$", ".o", basename(src)))
    if (!quiet) message("nvcc: compiling ", basename(src))
    status <- system2(nvcc,
      c(compile_flags, "-c", "-o", shQuote(obj), shQuote(src)),
      stdout = out_stream, stderr = out_stream)
    if (!identical(as.integer(status), 0L) || !file.exists(obj))
      stop("nvcc failed while compiling ", basename(src),
           " (exit status ", status, ").", call. = FALSE)
    objs <- c(objs, obj)
  }

  lib_tmp <- file.path(build_dir, .gpu_lib_name())
  if (!quiet) message("nvcc: linking ", .gpu_lib_name())
  status <- system2(nvcc,
    c("-shared", "-o", shQuote(lib_tmp), shQuote(objs),
      "-lcublas", "-lcusparse", "-lcusolver"),
    stdout = out_stream, stderr = out_stream)
  if (!identical(as.integer(status), 0L) || !file.exists(lib_tmp))
    stop("nvcc failed while linking (exit status ", status, ").",
         call. = FALSE)

  dest_dir <- .gpu_user_dir()
  dir.create(dest_dir, recursive = TRUE, showWarnings = FALSE)
  dest <- file.path(dest_dir, .gpu_lib_name())
  if (!file.copy(lib_tmp, dest, overwrite = TRUE))
    stop("Could not write '", dest, "'.", call. = FALSE)

  if (.gpu_env$lib_loaded) {
    message("Installed ", dest, ". A GPU library is already loaded in this ",
            "session; restart R to use the new build.")
  } else {
    .gpu_env$available <- NA
    ok <- gpu_available(force_recheck = TRUE)
    if (!quiet) message("Installed ", dest, ". gpu_available(): ", ok)
  }
  invisible(dest)
}

#' Report the state of GPU support
#'
#' Walks through each step \code{\link{gpu_available}} depends on and reports
#' where it stops: whether \code{nvcc} is present (needed only to build),
#' whether the GPU library was found and where, whether it loaded, whether its
#' interface version matches this package, and how many devices it sees.
#'
#' @param quiet If \code{TRUE}, return the result without printing it.
#' @return Invisibly, a list with elements \code{nvcc}, \code{nvcc_version},
#'   \code{searched}, \code{library}, \code{loaded}, \code{abi_expected},
#'   \code{abi_found}, \code{devices}, \code{available} and \code{reason}.
#' @seealso \code{\link{install_gpu}}, \code{\link{gpu_available}}
#' @export
#' @examples
#' gpu_status()
gpu_status <- function(quiet = FALSE) {
  available <- gpu_available()
  nvcc <- .find_nvcc()
  searched <- .gpu_lib_candidates()
  found <- .find_gpu_lib()
  st <- list(
    nvcc = if (nzchar(nvcc)) nvcc else NA_character_,
    nvcc_version = if (nzchar(nvcc)) .nvcc_version(nvcc) else NA_character_,
    searched = searched,
    library = if (.gpu_env$lib_loaded) .gpu_env$lib_path
              else if (!is.null(found)) found else NA_character_,
    loaded = .gpu_env$lib_loaded,
    abi_expected = .GPU_ABI_VERSION,
    abi_found = .gpu_env$abi_found,
    devices = .gpu_env$num_gpus,
    available = available,
    reason = .gpu_env$reason
  )
  if (!quiet) {
    yn <- function(x) if (isTRUE(x)) "yes" else "no"
    cat("RcppML GPU status\n")
    cat("  nvcc:            ",
        if (is.na(st$nvcc)) "not found (needed only for install_gpu())"
        else paste0(st$nvcc,
                    if (!is.na(st$nvcc_version))
                      paste0(" (CUDA ", st$nvcc_version, ")")),
        "\n", sep = "")
    cat("  library searched:\n")
    for (p in st$searched) cat("    ", p, "\n", sep = "")
    cat("  library found:   ",
        if (is.na(st$library)) "no" else st$library, "\n", sep = "")
    cat("  library loaded:  ", yn(st$loaded), "\n", sep = "")
    cat("  interface:       expected ", st$abi_expected, ", found ",
        if (is.na(st$abi_found)) "none" else st$abi_found, "\n", sep = "")
    cat("  devices:         ", st$devices, "\n", sep = "")
    cat("  available:       ", yn(st$available), "\n", sep = "")
    if (!isTRUE(st$available))
      cat("  reason:          ", st$reason, "\n", sep = "")
  }
  invisible(st)
}

# TRUE when a deflation SVD request carries element-wise constraints on more
# than one factor. The CPU path uses projection deflation for these; the GPU
# path still uses Hotelling deflation, which lets later factors repeat
# earlier ones.
.gpu_svd_constrained_deflation <- function(method, k, nonneg, L1, upper_bound) {
  identical(method, "deflation") && isTRUE(k > 1) &&
    (any(as.logical(nonneg)) || any(L1 > 0) || any(upper_bound > 0))
}

# Reason svd()/pca() cannot run a request on the GPU, or NULL if it can.
# svd() should call this before choosing the GPU and run on CPU (with a
# message, or a warning when resource = "gpu" was requested) when it returns
# a string.
#  - The GPU entry points take no `scale` argument, so a scaled (correlation)
#    PCA has to run on CPU.
#  - Constrained deflation with k > 1 needs projection deflation (CPU only).
#  - Cross-validated and masked SVD: the GPU kernels use an older holdout
#    rule and do not impute held-out entries, so they would not select the
#    same rank or hold out the same entries as CPU.
#  - IRLBA: the GPU code uses a different bidiagonal convention from CPU.
.gpu_svd_unsupported <- function(scale = FALSE, method = "deflation", k = 1L,
                                 nonneg = c(FALSE, FALSE), L1 = c(0, 0),
                                 upper_bound = c(0, 0),
                                 mask = NULL, test_fraction = 0,
                                 L2 = c(0, 0), L21 = c(0, 0),
                                 angular = c(0, 0), graph = FALSE,
                                 robust = FALSE) {
  if (isTRUE(scale)) return("scale = TRUE")
  # Only the unconstrained SVD has been verified on GPU in this release.
  if (any(as.logical(nonneg)) || any(L1 > 0) || any(upper_bound > 0) ||
      any(L2 > 0) || any(L21 > 0) || any(angular > 0) || isTRUE(graph) ||
      isTRUE(robust))
    return("constrained, penalized or robust SVD")
  if (identical(method, "irlba")) return("method = \"irlba\"")
  if (!is.null(mask)) return("a mask")
  if (isTRUE(test_fraction > 0))
    return("cross-validation (test_fraction > 0)")
  NULL
}

# Shared guard for the two GPU SVD entry points: these requests are answered
# differently on GPU than on CPU, so say so if one gets here (svd() is
# expected to have routed it to CPU through .gpu_svd_unsupported()).
.gpu_svd_warn_unsupported <- function(method, k, nonneg, L1, upper_bound,
                                      mask, test_fraction, ...) {
  why <- .gpu_svd_unsupported(FALSE, method, k, nonneg, L1, upper_bound,
                              mask = mask, test_fraction = test_fraction,
                              ...)
  if (!is.null(why))
    warning("The GPU SVD backend does not handle ", why, " the way the CPU ",
            "backend does; use resource = \"cpu\" for this fit.",
            call. = FALSE)
  invisible(NULL)
}

# Legacy .gpu_nmf_sparse() — REMOVED.
# .gpu_nmf_unified() and .gpu_nmf_cv_unified() — REMOVED (Phase 3).
# NMF GPU dispatch now goes through C++ gateway: Rcpp_nmf_full() →
# nmf/fit.hpp → gpu::bridge_nmf_sparse() → dlsym → gpu_bridge.cu

# -----------------------------------------------------------------------
# GPU NMF zero-copy dispatch (sp_read_gpu → NMF without re-upload)
# -----------------------------------------------------------------------

#' @keywords internal
.gpu_nmf_zerocopy <- function(gpu_mat, k, maxit, tol, seed,
                              L1 = c(0, 0), L2 = c(0, 0),
                              L21 = c(0, 0), ortho = c(0, 0),
                              upper_bound = c(0, 0),
                              cd_maxit = 10L, verbose = FALSE,
                              nonneg = c(TRUE, TRUE),
                              w_init = NULL,
                              loss_every = 1L, patience = 5L,
                              loss = "mse", huber_delta = 1.0,
                              irls_max_iter = 20L, irls_tol = 1e-4,
                              norm = "L1") {

  if (!gpu_available()) stop("GPU not available")
  if (!inherits(gpu_mat, "gpu_sparse_matrix"))
    stop("gpu_mat must be a gpu_sparse_matrix from sp_read_gpu()")

  m <- gpu_mat$m
  n <- gpu_mat$n
  nnz <- gpu_mat$nnz

  # Initialize W (k x m) and H (k x n)
  if (is.null(w_init)) {
    set.seed(seed)
    W <- matrix(runif(k * m), k, m)
  } else {
    W <- t(w_init)  # m x k → k x m
  }
  H <- matrix(runif(k * n), k, n)
  d <- rep(1.0, k)

  # Only Gaussian loss is supported here. Other losses used to be mapped to
  # MSE without notice; the sparse GPU IRLS path does not yet match CPU.
  if (!identical(tolower(loss), "mse"))
    stop("The zero-copy GPU path supports loss = \"mse\" only (got '", loss,
         "'). Read the file with st_read() and call nmf() on the dgCMatrix.",
         call. = FALSE)
  loss_type_int <- 0L

  if (!is.finite(nnz) || nnz > .Machine$integer.max)
    stop("The GPU backend supports at most 2^31 - 1 non-zeros.", call. = FALSE)

  # Map norm string to integer: L1=0, L2=1, none=2
  norm_type_int <- match(tolower(norm), c("l1", "l2", "none")) - 1L
  if (is.na(norm_type_int)) norm_type_int <- 0L

  result <- .gpu_call("rcppml_gpu_nmf_zerocopy_double",
    d_col_ptr_addr = as.double(gpu_mat$.col_ptr),
    d_row_idx_addr = as.double(gpu_mat$.row_idx),
    d_values_addr  = as.double(gpu_mat$.values),
    m = as.integer(m),
    n = as.integer(n),
    nnz_d = as.double(nnz),
    k = as.integer(k),
    W = as.double(W),
    H = as.double(H),
    d = as.double(d),
    max_iter = as.integer(maxit),
    tol = as.double(tol),
    # Penalty vectors follow the nmf() convention c(W, H)
    L1_H = as.double(L1[2]),
    L1_W = as.double(L1[1]),
    L2_H = as.double(L2[2]),
    L2_W = as.double(L2[1]),
    L21_H = as.double(L21[2]),
    L21_W = as.double(L21[1]),
    ortho_H = as.double(ortho[2]),
    ortho_W = as.double(ortho[1]),
    ub_H = as.double(upper_bound[2]),
    ub_W = as.double(upper_bound[1]),
    cd_maxit = as.integer(cd_maxit),
    verbose = as.integer(verbose),
    seed = as.integer(seed),
    loss_every = as.integer(loss_every),
    patience = as.integer(patience),
    nonneg_W = as.integer(nonneg[1]),
    nonneg_H = as.integer(nonneg[2]),
    loss_type = as.integer(loss_type_int),
    huber_delta = as.double(huber_delta),
    irls_max_iter = as.integer(irls_max_iter),
    irls_tol = as.double(irls_tol),
    norm_type = as.integer(norm_type_int),
    out_iter = as.integer(0L),
    out_converged = as.integer(0L),
    out_loss = as.double(0.0),
    out_status = as.integer(0L),
    out_tol = as.double(0.0)
  )

  if (result$out_status != 0L) {
    stop("GPU zero-copy NMF failed (status ", result$out_status, ")")
  }

  # Reshape W from flat k*m back to m x k
  W_out <- matrix(result$W, nrow = k, ncol = m)
  W_out <- t(W_out)

  # Reshape H from flat k*n back to k x n
  H_out <- matrix(result$H, nrow = k, ncol = n)

  list(
    w = W_out,
    d = result$d,
    h = H_out,
    iter = result$out_iter,
    converged = (result$out_converged != 0L),
    loss = result$out_loss,
    tol = result$out_tol,
    backend = "gpu"
  )
}

# -----------------------------------------------------------------------
# GPU SVD/PCA dispatch function
# -----------------------------------------------------------------------

#' @keywords internal
.gpu_svd_pca <- function(A, k_max, tol, max_iter, center, verbose, seed,
                         threads, L1_u, L1_v, L2_u, L2_v,
                         nonneg_u, nonneg_v, upper_bound_u, upper_bound_v,
                         L21_u = 0, L21_v = 0, angular_u = 0, angular_v = 0,
                         test_fraction, cv_seed, patience, mask_zeros,
                         method = "deflation", precision = "float",
                         graph_U = NULL, graph_V = NULL,
                         graph_lambda = c(0, 0),
                         obs_mask = NULL,
                         robust_delta = 0, irls_max_iter = 5L, irls_tol = 1e-4) {

  if (!gpu_available()) stop("GPU not available")
  .gpu_svd_warn_unsupported(method, k_max, c(nonneg_u, nonneg_v),
                            c(L1_u, L1_v), c(upper_bound_u, upper_bound_v),
                            obs_mask, test_fraction,
                            L2 = c(L2_u, L2_v), L21 = c(L21_u, L21_v),
                            angular = c(angular_u, angular_v),
                            graph = !is.null(graph_U) || !is.null(graph_V),
                            robust = isTRUE(robust_delta > 0))

  A <- .to_dgCMatrix(A)
  m <- nrow(A); n <- ncol(A); nnz <- length(A@x)

  # Map method name to algorithm integer:
  # 0 = deflation, 1 = irlba, 2 = lanczos, 3 = randomized, 4 = krylov
  algo_map <- c(deflation = 0L, irlba = 1L, lanczos = 2L, randomized = 3L, krylov = 4L)
  algo_int <- algo_map[method]
  if (is.na(algo_int)) {
    warning("GPU does not support method '", method, "', falling back to deflation")
    algo_int <- 0L
  }

  # Pre-allocate output arrays
  U_buf <- numeric(m * k_max)
  d_buf <- numeric(k_max)
  V_buf <- numeric(n * k_max)
  test_loss_buf <- numeric(k_max)
  iters_buf <- integer(k_max)
  row_means_buf <- numeric(m)

  # Choose precision entry point
  sym <- switch(precision,
    "float" = "rcppml_gpu_svd_pca_float",
    "rcppml_gpu_svd_pca_double"  # default
  )

  # Extract graph Laplacian and obs_mask CSC arrays
  gU <- .extract_sparse_csc(graph_U, lambda = graph_lambda[1])
  gV <- .extract_sparse_csc(graph_V, lambda = graph_lambda[2])
  om <- .extract_sparse_csc_rect(obs_mask)

  result <- .gpu_call(sym,
    col_ptr = as.integer(A@p),
    row_idx = as.integer(A@i),
    values  = as.double(A@x),
    m = as.integer(m),
    n = as.integer(n),
    nnz = as.integer(nnz),
    k_max = as.integer(k_max),
    U = as.double(U_buf),
    d = as.double(d_buf),
    V = as.double(V_buf),
    tol = as.double(tol),
    max_iter = as.integer(max_iter),
    center = as.integer(center),
    verbose = as.integer(verbose),
    seed = as.integer(seed),
    threads = as.integer(threads),
    L1_u = as.double(L1_u),
    L1_v = as.double(L1_v),
    L2_u = as.double(L2_u),
    L2_v = as.double(L2_v),
    nonneg_u = as.integer(nonneg_u),
    nonneg_v = as.integer(nonneg_v),
    ub_u = as.double(upper_bound_u),
    ub_v = as.double(upper_bound_v),
    L21_u = as.double(L21_u),
    L21_v = as.double(L21_v),
    angular_u = as.double(angular_u),
    angular_v = as.double(angular_v),
    test_fraction = as.double(test_fraction),
    cv_seed = as.integer(cv_seed),
    patience = as.integer(patience),
    mask_zeros = as.integer(mask_zeros),
    algorithm = as.integer(algo_int),
    graph_u_p = gU$p, graph_u_i = gU$i, graph_u_x = gU$x,
    graph_u_dim = gU$dim, graph_u_nnz = gU$nnz, graph_u_lambda = gU$lambda,
    graph_v_p = gV$p, graph_v_i = gV$i, graph_v_x = gV$x,
    graph_v_dim = gV$dim, graph_v_nnz = gV$nnz, graph_v_lambda = gV$lambda,
    obs_mask_p = om$p, obs_mask_i = om$i, obs_mask_x = om$x,
    obs_mask_rows = om$rows, obs_mask_cols = om$cols, obs_mask_nnz = om$nnz,
    out_k_selected = as.integer(0L),
    out_wall_time_ms = as.double(0),
    out_test_loss = as.double(test_loss_buf),
    out_iters_per_factor = as.integer(iters_buf),
    out_frobenius_norm_sq = as.double(0),
    out_row_means = as.double(row_means_buf),
    robust_delta = as.double(robust_delta),
    irls_max_iter = as.integer(irls_max_iter),
    irls_tol = as.double(irls_tol),
    out_status = as.integer(0L)
  )

  if (result$out_status != 0L) {
    stop("GPU SVD/PCA failed (status ", result$out_status, ")")
  }

  ksel <- result$out_k_selected

  # Reshape U from flat m*k_max -> m x ksel
  U_mat <- matrix(result$U, nrow = m, ncol = k_max)[, seq_len(ksel), drop = FALSE]
  V_mat <- matrix(result$V, nrow = n, ncol = k_max)[, seq_len(ksel), drop = FALSE]
  d_vec <- result$d[seq_len(ksel)]

  # k_computed: count non-zero iterations
  k_computed <- sum(result$out_iters_per_factor > 0)
  if (k_computed == 0) k_computed <- ksel

  iters_out <- result$out_iters_per_factor[seq_len(k_computed)]

  ret <- list(
    u = U_mat,
    d = d_vec,
    v = V_mat,
    centered = as.logical(result$center),
    wall_time_ms = result$out_wall_time_ms,
    iters_per_factor = iters_out,
    frobenius_norm_sq = result$out_frobenius_norm_sq
  )

  if (test_fraction > 0) {
    test_loss_out <- result$out_test_loss[seq_len(k_computed)]
    ret$test_loss <- test_loss_out
  }

  if (center) {
    ret$row_means <- result$out_row_means
  }

  ret
}

#' @title Dense-matrix GPU SVD/PCA
#' @description GPU SVD/PCA for a base R dense matrix (is.matrix(A) == TRUE).
#'   For deflation: uses cuBLAS GEMV instead of cuSPARSE SpMV (significantly
#'   faster for dense input). For other algorithms: converts to CSC internally.
#' @return A list with components \code{u}, \code{d}, and \code{v}.
#' @keywords internal
.gpu_svd_pca_dense <- function(A, k_max, tol, max_iter, center, verbose, seed,
                                threads, L1_u, L1_v, L2_u, L2_v,
                                nonneg_u, nonneg_v, upper_bound_u, upper_bound_v,
                                L21_u = 0, L21_v = 0, angular_u = 0, angular_v = 0,
                                test_fraction, cv_seed, patience, mask_zeros,
                                method = "deflation",
                                graph_U = NULL, graph_V = NULL,
                                graph_lambda = c(0, 0),
                                obs_mask = NULL,
                                robust_delta = 0, irls_max_iter = 5L, irls_tol = 1e-4) {

  if (!gpu_available()) stop("GPU not available")
  .gpu_svd_warn_unsupported(method, k_max, c(nonneg_u, nonneg_v),
                            c(L1_u, L1_v), c(upper_bound_u, upper_bound_v),
                            obs_mask, test_fraction,
                            L2 = c(L2_u, L2_v), L21 = c(L21_u, L21_v),
                            angular = c(angular_u, angular_v),
                            graph = !is.null(graph_U) || !is.null(graph_V),
                            robust = isTRUE(robust_delta > 0))

  if (!is.matrix(A)) A <- as.matrix(A)
  m <- nrow(A); n <- ncol(A)

  algo_map <- c(deflation = 0L, irlba = 1L, lanczos = 2L, randomized = 3L, krylov = 4L)
  algo_int <- algo_map[method]
  if (is.na(algo_int)) { warning("GPU does not support method '", method, "', using deflation"); algo_int <- 0L }

  U_buf   <- numeric(m * k_max)
  d_buf   <- numeric(k_max)
  V_buf   <- numeric(n * k_max)
  test_loss_buf <- numeric(k_max)
  iters_buf <- integer(k_max)
  row_means_buf <- numeric(m)

  # Extract graph Laplacian and obs_mask CSC arrays
  gU <- .extract_sparse_csc(graph_U, lambda = graph_lambda[1])
  gV <- .extract_sparse_csc(graph_V, lambda = graph_lambda[2])
  om <- .extract_sparse_csc_rect(obs_mask)

  result <- .gpu_call("rcppml_gpu_svd_pca_dense_float",
    A_data      = as.double(A),   # column-major flat vector
    m           = as.integer(m),
    n           = as.integer(n),
    k_max       = as.integer(k_max),
    U           = as.double(U_buf),
    d           = as.double(d_buf),
    V           = as.double(V_buf),
    tol         = as.double(tol),
    max_iter    = as.integer(max_iter),
    center      = as.integer(center),
    verbose     = as.integer(verbose),
    seed        = as.integer(seed),
    threads     = as.integer(threads),
    L1_u        = as.double(L1_u),
    L1_v        = as.double(L1_v),
    L2_u        = as.double(L2_u),
    L2_v        = as.double(L2_v),
    nonneg_u    = as.integer(nonneg_u),
    nonneg_v    = as.integer(nonneg_v),
    ub_u        = as.double(upper_bound_u),
    ub_v        = as.double(upper_bound_v),
    L21_u       = as.double(L21_u),
    L21_v       = as.double(L21_v),
    angular_u   = as.double(angular_u),
    angular_v   = as.double(angular_v),
    test_fraction = as.double(test_fraction),
    cv_seed     = as.integer(cv_seed),
    patience    = as.integer(patience),
    mask_zeros  = as.integer(mask_zeros),
    algorithm   = as.integer(algo_int),
    graph_u_p = gU$p, graph_u_i = gU$i, graph_u_x = gU$x,
    graph_u_dim = gU$dim, graph_u_nnz = gU$nnz, graph_u_lambda = gU$lambda,
    graph_v_p = gV$p, graph_v_i = gV$i, graph_v_x = gV$x,
    graph_v_dim = gV$dim, graph_v_nnz = gV$nnz, graph_v_lambda = gV$lambda,
    obs_mask_p = om$p, obs_mask_i = om$i, obs_mask_x = om$x,
    obs_mask_rows = om$rows, obs_mask_cols = om$cols, obs_mask_nnz = om$nnz,
    out_k_selected      = as.integer(0L),
    out_wall_time_ms    = as.double(0),
    out_test_loss       = as.double(test_loss_buf),
    out_iters_per_factor = as.integer(iters_buf),
    out_frobenius_norm_sq = as.double(0),
    out_row_means       = as.double(row_means_buf),
    robust_delta = as.double(robust_delta),
    irls_max_iter = as.integer(irls_max_iter),
    irls_tol = as.double(irls_tol),
    out_status          = as.integer(0L)
  )

  if (result$out_status != 0L) stop("GPU SVD/PCA (dense) failed (status ", result$out_status, ")")

  ksel    <- result$out_k_selected
  U_mat   <- matrix(result$U, nrow = m, ncol = k_max)[, seq_len(ksel), drop = FALSE]
  V_mat   <- matrix(result$V, nrow = n, ncol = k_max)[, seq_len(ksel), drop = FALSE]
  d_vec   <- result$d[seq_len(ksel)]
  k_computed <- sum(result$out_iters_per_factor > 0)
  if (k_computed == 0) k_computed <- ksel
  iters_out <- result$out_iters_per_factor[seq_len(k_computed)]

  ret <- list(
    u = U_mat, d = d_vec, v = V_mat,
    centered = as.logical(center),
    wall_time_ms = result$out_wall_time_ms,
    iters_per_factor = iters_out,
    frobenius_norm_sq = result$out_frobenius_norm_sq
  )
  if (test_fraction > 0) ret$test_loss <- result$out_test_loss[seq_len(k_computed)]
  if (center) ret$row_means <- result$out_row_means
  ret
}