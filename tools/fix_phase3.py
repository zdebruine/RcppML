#!/usr/bin/env python3
"""Phase 3: Change DEFAULT_CHUNK_COLS from 256 to 2048 and add chunk_cols param to sp_write."""
import sys

# 1. Update header_v2.hpp
path = "inst/include/sparsepress/format/header_v2.hpp"
with open(path, "r") as f:
    content = f.read()
content = content.replace(
    "static constexpr uint32_t DEFAULT_CHUNK_COLS = 256;",
    "static constexpr uint32_t DEFAULT_CHUNK_COLS = 2048;")
with open(path, "w") as f:
    f.write(content)
print(f"[1/5] header_v2.hpp: DEFAULT_CHUNK_COLS -> 2048")

# 2. Update header_v3.hpp
path = "inst/include/sparsepress/format/header_v3.hpp"
with open(path, "r") as f:
    content = f.read()
content = content.replace(
    "static constexpr uint32_t DEFAULT_CHUNK_COLS = 256;",
    "static constexpr uint32_t DEFAULT_CHUNK_COLS = 2048;")
with open(path, "w") as f:
    f.write(content)
print(f"[2/5] header_v3.hpp: DEFAULT_CHUNK_COLS -> 2048")

# 3. Add chunk_cols param to Rcpp_sp_write in sparsepress_bridge.cpp
path = "src/sparsepress_bridge.cpp"
with open(path, "r") as f:
    content = f.read()

# Add chunk_cols parameter to Rcpp_sp_write function signature
content = content.replace(
    """List Rcpp_sp_write(const S4& A, const std::string& path,
                   bool use_delta = true,
                   bool use_value_pred = false,
                   bool verbose = false,
                   const std::string& precision = "auto",
                   bool row_sort = false,
                   bool include_transpose = false) {""",
    """List Rcpp_sp_write(const S4& A, const std::string& path,
                   bool use_delta = true,
                   bool use_value_pred = false,
                   bool verbose = false,
                   const std::string& precision = "auto",
                   bool row_sort = false,
                   bool include_transpose = false,
                   int chunk_cols = 2048) {""")

# Find where write_v2 is called and add chunk_cols
# The CompressConfig_v2 already has chunk_cols field, just need to set it
content = content.replace(
    """        cfg2.verbose = verbose ? 2 : 0;""",
    """        cfg2.verbose = verbose ? 2 : 0;
        cfg2.chunk_cols = static_cast<uint32_t>(chunk_cols);""")

with open(path, "w") as f:
    f.write(content)

print(f"[3/5] sparsepress_bridge.cpp: Added chunk_cols param and wired to config")

# 4. Add chunk_cols to sp_write in R/sparsepress.R
path = "R/sparsepress.R"
with open(path, "r") as f:
    content = f.read()

content = content.replace(
    """sp_write <- function(x, path, delta = TRUE, value_pred = FALSE, verbose = FALSE,
                     precision = "auto", row_sort = FALSE,
                     include_transpose = FALSE) {""",
    """sp_write <- function(x, path, delta = TRUE, value_pred = FALSE, verbose = FALSE,
                     precision = "auto", row_sort = FALSE,
                     include_transpose = FALSE,
                     chunk_cols = 2048L) {""")

# Also update the Rcpp_sp_write call to pass chunk_cols
content = content.replace(
    """  stats <- Rcpp_sp_write(x, path,
                         use_delta = delta,""",
    """  stats <- Rcpp_sp_write(x, path,
                         use_delta = delta,""")

# Need to find the full call and add chunk_cols at the end
# Let me find the pattern
import re
# Find the Rcpp_sp_write call and add chunk_cols before the closing paren
old_call = re.search(
    r'stats <- Rcpp_sp_write\(x, path,\s*\n\s*use_delta = delta,\s*\n\s*use_value_pred = value_pred,\s*\n\s*verbose = verbose,\s*\n\s*precision = precision,\s*\n\s*row_sort = row_sort,\s*\n\s*include_transpose = include_transpose\)',
    content)
if old_call:
    old_text = old_call.group(0)
    new_text = old_text.replace(
        "include_transpose = include_transpose)",
        "include_transpose = include_transpose,\n                         chunk_cols = as.integer(chunk_cols))")
    content = content.replace(old_text, new_text)
    print("[4/5] R/sparsepress.R: Added chunk_cols param to sp_write()")
else:
    print("[4/5] WARNING: Could not find Rcpp_sp_write call pattern in sp_write")

# Update @param docs
content = content.replace(
    "#' @param include_transpose Logical; also store transposed chunks.",
    "#' @param include_transpose Logical; also store transposed chunks.\n#' @param chunk_cols Integer; columns per chunk. Default 2048.")

# Also update sp_write_dense default from 256 to 2048
content = content.replace(
    """sp_write_dense <- function(x, path, include_transpose = FALSE,
                           chunk_cols = 256L, codec = "raw",""",
    """sp_write_dense <- function(x, path, include_transpose = FALSE,
                           chunk_cols = 2048L, codec = "raw",""")

with open(path, "w") as f:
    f.write(content)

# 5. Update RcppExports for Rcpp_sp_write
path = "R/RcppExports.R"
with open(path, "r") as f:
    content = f.read()

# Add chunk_cols param to Rcpp_sp_write wrapper
old_wrapper = re.search(
    r'Rcpp_sp_write <- function\(A, path, use_delta = TRUE, use_value_pred = FALSE, verbose = FALSE, precision = "auto", row_sort = FALSE, include_transpose = FALSE\)',
    content)
if old_wrapper:
    content = content.replace(
        old_wrapper.group(0),
        'Rcpp_sp_write <- function(A, path, use_delta = TRUE, use_value_pred = FALSE, verbose = FALSE, precision = "auto", row_sort = FALSE, include_transpose = FALSE, chunk_cols = 2048L)')
    # Update the .Call too
    content = content.replace(
        ".Call(`_RcppML_Rcpp_sp_write`, A, path, use_delta, use_value_pred, verbose, precision, row_sort, include_transpose)",
        ".Call(`_RcppML_Rcpp_sp_write`, A, path, use_delta, use_value_pred, verbose, precision, row_sort, include_transpose, chunk_cols)")
    print("[5/5] R/RcppExports.R: Updated Rcpp_sp_write wrapper")
else:
    print("[5/5] WARNING: Could not find Rcpp_sp_write in RcppExports.R")

with open(path, "w") as f:
    f.write(content)

# 6. Update src/RcppExports.cpp
path = "src/RcppExports.cpp"
with open(path, "r") as f:
    content = f.read()

# Update forward declaration
content = content.replace(
    'List Rcpp_sp_write(const S4& A, const std::string& path, bool use_delta, bool use_value_pred, bool verbose, const std::string& precision, bool row_sort, bool include_transpose);',
    'List Rcpp_sp_write(const S4& A, const std::string& path, bool use_delta, bool use_value_pred, bool verbose, const std::string& precision, bool row_sort, bool include_transpose, int chunk_cols);')

# Update SEXP wrapper signature
content = content.replace(
    'RcppExport SEXP _RcppML_Rcpp_sp_write(SEXP ASEXP, SEXP pathSEXP, SEXP use_deltaSEXP, SEXP use_value_predSEXP, SEXP verboseSEXP, SEXP precisionSEXP, SEXP row_sortSEXP, SEXP include_transposeSEXP) {',
    'RcppExport SEXP _RcppML_Rcpp_sp_write(SEXP ASEXP, SEXP pathSEXP, SEXP use_deltaSEXP, SEXP use_value_predSEXP, SEXP verboseSEXP, SEXP precisionSEXP, SEXP row_sortSEXP, SEXP include_transposeSEXP, SEXP chunk_colsSEXP) {')

# Add parameter extraction
content = content.replace(
    '    Rcpp::traits::input_parameter< bool >::type include_transpose(include_transposeSEXP);\n    rcpp_result_gen = Rcpp::wrap(Rcpp_sp_write(A, path, use_delta, use_value_pred, verbose, precision, row_sort, include_transpose));',
    '    Rcpp::traits::input_parameter< bool >::type include_transpose(include_transposeSEXP);\n    Rcpp::traits::input_parameter< int >::type chunk_cols(chunk_colsSEXP);\n    rcpp_result_gen = Rcpp::wrap(Rcpp_sp_write(A, path, use_delta, use_value_pred, verbose, precision, row_sort, include_transpose, chunk_cols));')

# Update registration table
content = content.replace(
    '{"_RcppML_Rcpp_sp_write", (DL_FUNC) &_RcppML_Rcpp_sp_write, 8},',
    '{"_RcppML_Rcpp_sp_write", (DL_FUNC) &_RcppML_Rcpp_sp_write, 9},')

with open(path, "w") as f:
    f.write(content)
print("[6/6] src/RcppExports.cpp: Updated Rcpp_sp_write bindings")

print("\nPhase 3 complete!")
