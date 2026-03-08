#!/usr/bin/env python3
"""Add st_add_transpose to R/streampress.R and update RcppExports."""

# 1. Append st_add_transpose to R/streampress.R
STREAMPRESS_R = "/mnt/home/debruinz/RcppML-2/R/streampress.R"

with open(STREAMPRESS_R) as f:
    content = f.read()

if 'st_add_transpose' not in content:
    content += r'''

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
#' \donttest{
#' library(Matrix)
#' A <- rsparsematrix(100, 50, 0.1)
#' f <- tempfile(fileext = ".spz")
#' st_write(A, f)  # no transpose
#' st_add_transpose(f)  # adds transpose section
#' unlink(f)
#' }
#'
#' @seealso \code{\link{st_write}}, \code{\link{st_read}}
#' @export
st_add_transpose <- function(path, verbose = TRUE) {
  path <- normalizePath(path, mustWork = TRUE)
  Rcpp_st_add_transpose(path, verbose = verbose)
  invisible(path)
}
'''
    with open(STREAMPRESS_R, 'w') as f:
        f.write(content)
    print("[1/3] Added st_add_transpose to R/streampress.R")
else:
    print("[1/3] st_add_transpose already in R/streampress.R")

# 2. Add to R/RcppExports.R
RCPP_EXPORTS_R = "/mnt/home/debruinz/RcppML-2/R/RcppExports.R"
with open(RCPP_EXPORTS_R) as f:
    content = f.read()

if 'Rcpp_st_add_transpose' not in content:
    content += r'''
Rcpp_st_add_transpose <- function(path, verbose = TRUE) {
    .Call('_RcppML_Rcpp_st_add_transpose', PACKAGE = 'RcppML', path, verbose)
}
'''
    with open(RCPP_EXPORTS_R, 'w') as f:
        f.write(content)
    print("[2/3] Added Rcpp_st_add_transpose to R/RcppExports.R")
else:
    print("[2/3] Rcpp_st_add_transpose already in R/RcppExports.R")

# 3. Add to src/RcppExports.cpp
RCPP_EXPORTS_CPP = "/mnt/home/debruinz/RcppML-2/src/RcppExports.cpp"
with open(RCPP_EXPORTS_CPP) as f:
    content = f.read()

if 'Rcpp_st_add_transpose' not in content:
    # Find the R_CallMethodDef table
    import re
    
    # Add the wrapper function before the CallEntries table
    call_entries_pos = content.find('static const R_CallMethodDef CallEntries[]')
    if call_entries_pos < 0:
        print("[3/3] ERROR: Cannot find CallEntries table")
    else:
        wrapper = r'''
// Rcpp_st_add_transpose
bool Rcpp_st_add_transpose(const std::string& path, bool verbose);
RcppExport SEXP _RcppML_Rcpp_st_add_transpose(SEXP pathSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::string& >::type path(pathSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(Rcpp_st_add_transpose(path, verbose));
    return rcpp_result_gen;
END_RCPP
}

'''
        content = content[:call_entries_pos] + wrapper + content[call_entries_pos:]
        
        # Add entry to CallEntries table
        # Find the NULL sentinel at end of table
        null_pos = content.find('    {NULL, NULL, 0}')
        if null_pos >= 0:
            entry = '    {"_RcppML_Rcpp_st_add_transpose", (DL_FUNC) &_RcppML_Rcpp_st_add_transpose, 2},\n'
            content = content[:null_pos] + entry + content[null_pos:]
        
        with open(RCPP_EXPORTS_CPP, 'w') as f:
            f.write(content)
        print("[3/3] Added Rcpp_st_add_transpose to src/RcppExports.cpp")
else:
    print("[3/3] Rcpp_st_add_transpose already in src/RcppExports.cpp")

print("[OK] All done")
