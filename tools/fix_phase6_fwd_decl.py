#!/usr/bin/env python3
"""Phase 6: Fix RcppExports.cpp - add forward declaration for Rcpp_get_available_ram_mb"""

path = "src/RcppExports.cpp"
with open(path, "r") as f:
    content = f.read()

# Find the wrapper we inserted and add forward declaration before it
old = """// Rcpp_get_available_ram_mb
RcppExport SEXP _RcppML_Rcpp_get_available_ram_mb() {"""

new = """// Rcpp_get_available_ram_mb
double Rcpp_get_available_ram_mb();
RcppExport SEXP _RcppML_Rcpp_get_available_ram_mb() {"""

content = content.replace(old, new, 1)

with open(path, "w") as f:
    f.write(content)

print("OK: Added forward declaration for Rcpp_get_available_ram_mb")
