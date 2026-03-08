#' Validation Helper Functions for NMF
#' 
#' Internal functions to consolidate validation logic and reduce code duplication.
#' @keywords internal
#' @name nmf_validation
NULL

# Convert any sparse matrix to dgCMatrix without triggering Matrix deprecation
# warnings (Matrix >= 1.5-0 deprecates direct as(<dgTMatrix>, "dgCMatrix")).
.to_dgCMatrix <- function(x) {
  if (inherits(x, "dgCMatrix")) return(x)
  as(as(x, "CsparseMatrix"), "generalMatrix")
}

#' Validate and prepare input data
#'
#' Accepts a matrix (dense or sparse), or a file path (character string).
#' File paths are auto-loaded based on extension:
#' \itemize{
#'   \item \code{.spz}: SparsePress compressed format
#'   \item \code{.rds}: R serialized object (must contain a matrix)
#'   \item \code{.mtx}, \code{.mtx.gz}: Matrix Market format (requires Matrix package)
#'   \item \code{.csv}, \code{.csv.gz}, \code{.tsv}, \code{.tsv.gz}: Delimited text
#'   \item \code{.h5}, \code{.hdf5}: HDF5 (requires hdf5r package)
#'   \item \code{.h5ad}: AnnData HDF5 (requires hdf5r package)
#' }
#' @param data A matrix, sparse matrix, or file path (character string).
#' @return A list with \code{data}, \code{is_sparse}, and \code{has_na}.
#' @keywords internal
validate_data <- function(data) {
  # --- File path input ---
  if (is.character(data) && length(data) == 1 && !inherits(data, "factor")) {
    data <- .load_sparse_file(data)
  }

  # Detect if sparse
  is_sparse <- inherits(data, "sparseMatrix")
  
  # Convert to appropriate format
  if (is_sparse) {
    if (class(data)[[1]] != "dgCMatrix") {
      data <- .to_dgCMatrix(data)
    }
    # Check for NA in sparse data
    if (any(is.na(data@x))) {
      na_count <- sum(is.na(data@x))
      na_frac <- na_count / length(data)
      warning(sprintf("Detected %d NA values (%.2f%% of data). Automatically creating mask for missing values.", 
                      na_count, na_frac * 100), call. = FALSE)
      return(list(data = data, is_sparse = TRUE, has_na = TRUE))
    }
  } else if (canCoerce(data, "matrix")) {
    tryCatch({
      data <- as.matrix(data)
    }, error = function(e) {
      stop("'data' was not coercible to a matrix")
    })
    if (!is.numeric(data)) {
      # Try numeric conversion; if it produces a non-matrix or non-numeric, error out
      tryCatch({
        data <- matrix(as.numeric(data), nrow = nrow(data), ncol = ncol(data))
      }, error = function(e) {
        stop("'data' was not coercible to a numeric matrix")
      })
      if (!is.numeric(data) || !is.matrix(data)) {
        stop("'data' was not coercible to a numeric matrix")
      }
    }
    # Check for NA in dense data
    if (any(is.na(data))) {
      na_count <- sum(is.na(data))
      na_frac <- na_count / length(data)
      warning(sprintf("Detected %d NA values (%.2f%% of data). Automatically creating mask for missing values.", 
                      na_count, na_frac * 100), call. = FALSE)
      return(list(data = data, is_sparse = FALSE, has_na = TRUE))
    }
  } else {
    stop("'data' was not coercible to a matrix")
  }
  
  list(data = data, is_sparse = is_sparse, has_na = FALSE)
}

#' Validate and expand penalty vectors
#' @param penalty Numeric scalar or length-2 vector
#' @param name Character name of penalty for error messages
#' @param allow_negative Logical; allow negative values (default FALSE)
#' @return Length-2 numeric vector \code{c(w, h)}
#' @keywords internal
validate_penalty <- function(penalty, name, allow_negative = FALSE) {
  if (length(penalty) == 1) {
    penalty <- rep(penalty, 2)
  } else if (length(penalty) != 2) {
    stop(sprintf("'%s' must be length 1 or 2 for c(w, h)", name))
  }
  
  if (!allow_negative && any(penalty < 0)) {
    stop(sprintf("'%s' values must be non-negative", name))
  }
  
  penalty
}

#' Validate all penalties at once
#' @param L1,L2,L21,angular,graph_lambda,upper_bound Penalty parameters to validate
#' @return Named list of validated length-2 penalty vectors
#' @keywords internal
validate_all_penalties <- function(L1, L2, L21, angular, graph_lambda, upper_bound) {
  L1 <- validate_penalty(L1, "L1")
  if (max(L1) >= 1 || min(L1) < 0) {
    stop("L1 penalties must be strictly in the range [0,1)")
  }
  
  L2 <- validate_penalty(L2, "L2")
  if (min(L2) < 0) {
    stop("L2 penalties must be strictly >= 0")
  }
  
  L21 <- validate_penalty(L21, "L21")
  if (min(L21) < 0) {
    stop("L21 penalties must be strictly >= 0")
  }
  
  angular <- validate_penalty(angular, "angular")
  if (min(angular) < 0) {
    stop("angular penalties must be strictly >= 0")
  }
  
  graph_lambda <- validate_penalty(graph_lambda, "graph_lambda")
  if (any(graph_lambda < 0)) {
    stop("'graph_lambda' values must be non-negative")
  }
  
  upper_bound <- validate_penalty(upper_bound, "upper_bound")
  if (any(upper_bound < 0)) {
    stop("'upper_bound' values must be non-negative")
  }
  
  list(L1 = L1, L2 = L2, L21 = L21, angular = angular, 
       graph_lambda = graph_lambda, upper_bound = upper_bound)
}

#' Validate cross-validation parameters
#' @param test_fraction Numeric in \code{[0, 1)} for CV test fraction
#' @param patience Numeric > 0 for early stopping patience
#' @return List with validated \code{test_fraction} and \code{patience}
#' @keywords internal
validate_cv_params <- function(test_fraction, patience) {
  # Validate test_fraction
  if (!is.numeric(test_fraction) || length(test_fraction) != 1) {
    stop("'test_fraction' must be a single numeric value")
  }
  if (test_fraction < 0 || test_fraction >= 1) {
    stop("'test_fraction' must be in the range [0, 1)")
  }
  
  # Validate patience
  if (!is.numeric(patience) || length(patience) != 1) {
    stop("'patience' must be a single numeric value")
  }
  
  list(test_fraction = test_fraction, patience = patience)
}

#' Validate graphs regularization matrices
#' @param graph_W,graph_H Optional sparse matrices for graph regularization
#' @param data_dims Integer vector \code{c(nrow, ncol)} of the input data
#' @param graph_lambda Length-2 numeric vector of graph lambda values
#' @return List with validated \code{graph_W} and \code{graph_H} (coerced to dgCMatrix)
#' @keywords internal
validate_graphs <- function(graph_W, graph_H, data_dims, graph_lambda) {
  n_features <- data_dims[1]
  n_samples <- data_dims[2]
  
  # Validate graph_W
  if (!is.null(graph_W)) {
    if (!is(graph_W, "dgCMatrix")) {
      if (canCoerce(graph_W, "dgCMatrix")) {
        graph_W <- .to_dgCMatrix(graph_W)
      } else {
        stop("'graph_W' must be coercible to dgCMatrix")
      }
    }
    if (nrow(graph_W) != n_features || ncol(graph_W) != n_features) {
      stop(sprintf("'graph_W' must be a %d x %d matrix (p x p where p is number of features)", 
                   n_features, n_features))
    }
    if (graph_lambda[1] <= 0) {
      warning("'graph_W' provided but 'graph_lambda[1]' <= 0; graph regularization will not be applied")
    }
  }
  
  # Validate graph_H
  if (!is.null(graph_H)) {
    if (!is(graph_H, "dgCMatrix")) {
      if (canCoerce(graph_H, "dgCMatrix")) {
        graph_H <- .to_dgCMatrix(graph_H)
      } else {
        stop("'graph_H' must be coercible to dgCMatrix")
      }
    }
    if (nrow(graph_H) != n_samples || ncol(graph_H) != n_samples) {
      stop(sprintf("'graph_H' must be a %d x %d matrix (n x n where n is number of samples)", 
                   n_samples, n_samples))
    }
    if (graph_lambda[2] <= 0) {
      warning("'graph_H' provided but 'graph_lambda[2]' <= 0; graph regularization will not be applied")
    }
  }
  
  list(graph_W = graph_W, graph_H = graph_H)
}

#' Validate mask parameter
#' @param mask NULL, character (\code{"zeros"} or \code{"NA"}), or matrix
#' @param sparse Logical; whether sparse mode is enabled
#' @param has_na Logical; whether data contains NA values
#' @return List with \code{mask_matrix} (dgCMatrix), \code{mask_zeros} (logical), and optionally \code{mask_na}
#' @keywords internal
validate_mask <- function(mask, sparse, has_na) {
  if (is.null(mask)) {
    # Handle NA in data
    if (has_na) {
      warning("NA values were detected in the data. Setting \"mask = 'NA'\"")
      mask <- "NA"
    } else {
      mask_matrix <- new("dgCMatrix")
      mask_zeros <- sparse
      return(list(mask_matrix = mask_matrix, mask_zeros = mask_zeros))
    }
  }
  
  if (class(mask)[[1]] == "character") {
    if (mask == "zeros") {
      mask_matrix <- new("dgCMatrix")
      mask_zeros <- TRUE
      if (sparse) {
        warning("Both 'mask = \"zeros\"' and 'sparse = TRUE' specified. Using mask = \"zeros\".")
      }
      return(list(mask_matrix = mask_matrix, mask_zeros = mask_zeros))
    } else if (mask == "NA") {
      # Will be handled by data preparation
      mask_matrix <- new("dgCMatrix")
      mask_zeros <- FALSE
      return(list(mask_matrix = mask_matrix, mask_zeros = mask_zeros, mask_na = TRUE))
    } else {
      stop("'mask' must be a matrix, 'zeros', 'NA', or NULL")
    }
  }
  
  # mask is a matrix
  mask_zeros <- FALSE
  if (sparse) {
    warning("'sparse = TRUE' is ignored when a mask matrix is provided")
  }
  
  if (!canCoerce(mask, "dgCMatrix")) {
    if (canCoerce(mask, "matrix")) {
      mask <- as.matrix(mask)
    } else {
      stop("could not coerce the value of 'mask' to a sparse pattern matrix (dgCMatrix)")
    }
  }
  
  # Use dMatrix intermediate to avoid ngCMatrix -> dgCMatrix deprecation warning
  mask_matrix <- .to_dgCMatrix(as(mask, "dMatrix"))
  
  list(mask_matrix = mask_matrix, mask_zeros = mask_zeros)
}

#' Validate simple logical/numeric parameters
#' @param sort_model Logical; whether to sort model factors
#' @param sparse Logical; whether to use sparse mode
#' @param nonneg Logical length 1 or 2; non-negativity constraints
#' @return List with validated \code{sort_model}, \code{sparse}, and \code{nonneg}
#' @keywords internal
validate_simple_params <- function(sort_model, sparse, nonneg) {
  # Validate sort_model
  if (!is.logical(sort_model) || length(sort_model) != 1) {
    stop("'sort_model' must be a single logical value")
  }
  
  # Validate sparse
  if (!is.logical(sparse) || length(sparse) != 1) {
    stop("'sparse' must be a single logical value")
  }
  
  # Validate nonneg
  if (!is.logical(nonneg)) {
    stop("'nonneg' must be logical")
  }
  if (length(nonneg) == 1) {
    nonneg <- rep(nonneg, 2)
  }
  if (length(nonneg) != 2 || any(is.na(nonneg))) {
    stop("'nonneg' must be length 1 or 2 with no NA values")
  }
  
  list(sort_model = sort_model, sparse = sparse, nonneg = nonneg)
}


# ---- File loader utilities -----------------------------------------------

#' Load a sparse matrix from a file path
#'
#' Auto-detects format from file extension and loads the matrix.
#' Supported formats: .spz, .rds, .mtx, .mtx.gz, .csv, .csv.gz,
#' .tsv, .tsv.gz, .h5, .hdf5, .h5ad
#'
#' @param path Character string giving path to the file.
#' @return A matrix (dense or dgCMatrix).
#' @keywords internal
.load_sparse_file <- function(path) {
  if (!file.exists(path)) {
    stop("File not found: ", path)
  }


  # Normalise extension (handle .gz double extensions)
  ext <- tolower(tools::file_ext(path))
  if (ext == "gz") {
    base_ext <- tolower(tools::file_ext(sub("\\.[^.]+$", "", path)))
    ext <- paste0(base_ext, ".gz")
  }

  data <- switch(ext,
    # --- SparsePress ---
    "spz" = {
      sp_read(path)
    },

    # --- R serialised ---
    "rds" = {
      obj <- readRDS(path)
      if (inherits(obj, "sparseMatrix") || is.matrix(obj)) {
        obj
      } else if (inherits(obj, "data.frame")) {
        as.matrix(obj)
      } else {
        stop("RDS file must contain a matrix, sparseMatrix, or data.frame")
      }
    },

    # --- Matrix Market ---
    "mtx" = , "mtx.gz" = {
      if (!requireNamespace("Matrix", quietly = TRUE)) {
        stop("Package 'Matrix' is required to read .mtx files")
      }
      .to_dgCMatrix(Matrix::readMM(path))
    },

    # --- Delimited text ---
    "csv" = , "csv.gz" = {
      message("Note: loading CSV as dense matrix; consider converting to .spz or .rds for efficiency")
      df <- utils::read.csv(path, header = TRUE, row.names = 1, check.names = FALSE)
      as.matrix(df)
    },
    "tsv" = , "tsv.gz" = {
      message("Note: loading TSV as dense matrix; consider converting to .spz or .rds for efficiency")
      df <- utils::read.delim(path, header = TRUE, row.names = 1, check.names = FALSE)
      as.matrix(df)
    },

    # --- HDF5 ---
    "h5" = , "hdf5" = {
      .load_h5(path)
    },
    "h5ad" = {
      .load_h5ad(path)
    },

    # --- Loom ---
    "loom" = {
      .load_loom(path)
    },

    # fallback
    stop("Unsupported file format '.", ext, "'. Supported: .spz, .rds, .mtx, .csv, .tsv, .h5, .hdf5, .h5ad, .loom")
  )

  data
}


#' Load matrix from generic HDF5 file
#'
#' Looks for datasets named "X", "matrix", or "data". Falls back to
#' the first 2-D dataset found.
#' @param path Path to .h5 file
#' @return A matrix or dgCMatrix
#' @keywords internal
.load_h5 <- function(path) {
  if (!requireNamespace("hdf5r", quietly = TRUE)) {
    stop("Package 'hdf5r' is required to read HDF5 files. Install with: install.packages('hdf5r')")
  }
  f <- hdf5r::H5File$new(path, mode = "r")
  on.exit(f$close_all(), add = TRUE)

  # Try common dataset names
  for (name in c("X", "matrix", "data", "counts")) {
    if (f$exists(name)) {
      ds <- f[[name]]
      if (inherits(ds, "H5Group")) {
        # Could be a sparse group (e.g. 10x format: data, indices, indptr, shape)
        return(.load_h5_sparse_group(ds))
      } else {
        return(ds$read())
      }
    }
  }

  # Fallback: find first 2-D dataset
  all_names <- f$ls()$name
  for (name in all_names) {
    ds <- f[[name]]
    if (inherits(ds, "H5D") && length(ds$dims) == 2) {
      return(ds$read())
    }
  }

  stop("No suitable matrix dataset found in HDF5 file. Expected 'X', 'matrix', 'data', or 'counts'")
}


#' Load sparse matrix from HDF5 group (10x-style CSC)
#' @param grp An hdf5r H5Group with data/indices/indptr/shape
#' @return A dgCMatrix
#' @keywords internal
.load_h5_sparse_group <- function(grp) {
  required <- c("data", "indices", "indptr")
  for (field in required) {
    if (!grp$exists(field)) {
      stop("HDF5 sparse group missing required dataset '", field, "'")
    }
  }
  x <- grp[["data"]]$read()
  i <- grp[["indices"]]$read()
  p <- grp[["indptr"]]$read()

  if (grp$exists("shape")) {
    dims <- grp[["shape"]]$read()
  } else {
    dims <- c(max(i) + 1L, length(p) - 1L)
  }

  methods::new("dgCMatrix",
    x = as.double(x),
    i = as.integer(i),
    p = as.integer(p),
    Dim = as.integer(dims)
  )
}


#' Load matrix from AnnData HDF5 (.h5ad)
#'
#' Reads the X matrix from an AnnData-format HDF5 file.
#' Supports both dense and sparse (CSC/CSR) X matrices.
#' @param path Path to .h5ad file
#' @return A matrix or dgCMatrix
#' @keywords internal
.load_h5ad <- function(path) {
  if (!requireNamespace("hdf5r", quietly = TRUE)) {
    stop("Package 'hdf5r' is required to read .h5ad files. Install with: install.packages('hdf5r')")
  }
  f <- hdf5r::H5File$new(path, mode = "r")
  on.exit(f$close_all(), add = TRUE)

  if (!f$exists("X")) {
    stop("AnnData file does not contain 'X' matrix")
  }

  X <- f[["X"]]
  if (inherits(X, "H5Group")) {
    # Sparse X — could be CSC or CSR
    mat <- .load_h5_sparse_group(X)
    # Check encoding_type attribute for CSR
    if (X$attr_exists("encoding-type")) {
      enc <- X$attr_open("encoding-type")$read()
      if (grepl("csr", enc, ignore.case = TRUE)) {
        # Transpose CSR to get CSC
        mat <- Matrix::t(mat)
      }
    }
    return(mat)
  } else {
    return(X$read())
  }
}


#' Load matrix from Loom file (.loom)
#'
#' Loom files store the main matrix in /matrix (genes x cells, row-major).
#' Gene names may be in /row_attrs/Gene, cell IDs in /col_attrs/CellID.
#'
#' @param path Path to .loom file
#' @return A dgCMatrix (transposed to features x samples)
#' @keywords internal
.load_loom <- function(path) {
  if (!requireNamespace("hdf5r", quietly = TRUE)) {
    stop("Package 'hdf5r' is required to read .loom files. ",
         "Install with: install.packages('hdf5r')")
  }

  f <- hdf5r::H5File$new(path, mode = "r")
  on.exit(f$close_all(), add = TRUE)

  if (!f$exists("matrix")) {
    stop("Loom file does not contain '/matrix' dataset")
  }

  # /matrix is stored as genes x cells (row-major)
  x <- f[["matrix"]]$read()

  # Transpose to features x samples (our convention)
  x <- t(x)

  # Try to get gene (row) names
  if (f$exists("row_attrs/Gene")) {
    rownames(x) <- f[["row_attrs/Gene"]]$read()
  } else if (f$exists("row_attrs/gene_names")) {
    rownames(x) <- f[["row_attrs/gene_names"]]$read()
  } else if (f$exists("row_attrs/gene")) {
    rownames(x) <- f[["row_attrs/gene"]]$read()
  }

  # Try to get cell (column) names
  if (f$exists("col_attrs/CellID")) {
    colnames(x) <- f[["col_attrs/CellID"]]$read()
  } else if (f$exists("col_attrs/cell_names")) {
    colnames(x) <- f[["col_attrs/cell_names"]]$read()
  } else if (f$exists("col_attrs/obs_names")) {
    colnames(x) <- f[["col_attrs/obs_names"]]$read()
  }

  .to_dgCMatrix(x)
}
