#!/usr/bin/env python3
"""Phase 6: Add Rcpp_get_available_ram_mb to RcppExports.R and src/RcppExports.cpp"""

# --- R side ---
r_path = "R/RcppExports.R"
with open(r_path, "r") as f:
    r_content = f.read()

r_binding = """
Rcpp_get_available_ram_mb <- function() {
    .Call('_RcppML_Rcpp_get_available_ram_mb', PACKAGE = 'RcppML')
}
"""

# Add before the Rcpp_st_add_transpose entry
marker_r = "\nRcpp_st_add_transpose"
r_content = r_content.replace(marker_r, r_binding + "\nRcpp_st_add_transpose", 1)

with open(r_path, "w") as f:
    f.write(r_content)
print("OK: Added R binding for Rcpp_get_available_ram_mb")

# --- C++ side ---
cpp_path = "src/RcppExports.cpp"
with open(cpp_path, "r") as f:
    cpp_content = f.read()

cpp_wrapper = """
// Rcpp_get_available_ram_mb
RcppExport SEXP _RcppML_Rcpp_get_available_ram_mb() {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    rcpp_result_gen = Rcpp::wrap(Rcpp_get_available_ram_mb());
    return rcpp_result_gen;
END_RCPP
}

"""

# Find the Rcpp_st_add_transpose wrapper and insert before it
marker_cpp = "// Rcpp_st_add_transpose"
if marker_cpp not in cpp_content:
    # Try alternate marker
    marker_cpp = "RcppExport SEXP _RcppML_Rcpp_st_add_transpose"
cpp_content = cpp_content.replace(marker_cpp, cpp_wrapper + marker_cpp, 1)

# Also add to the registration table
# Find the R_CallMethodDef entries and add our new entry
reg_marker = '    {"_RcppML_Rcpp_st_add_transpose"'
reg_new = '    {"_RcppML_Rcpp_get_available_ram_mb", (DL_FUNC) &_RcppML_Rcpp_get_available_ram_mb, 0},\n' + reg_marker
cpp_content = cpp_content.replace(reg_marker, reg_new, 1)

with open(cpp_path, "w") as f:
    f.write(cpp_content)
print("OK: Added C++ wrapper and registration for Rcpp_get_available_ram_mb")
