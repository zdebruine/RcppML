#!/usr/bin/env python3
"""Phase 6: Add Rcpp_get_available_ram_mb() to sparsepress_bridge.cpp"""
import re

path = "src/sparsepress_bridge.cpp"
with open(path, "r") as f:
    content = f.read()

# Add #include <FactorNet/core/platform.hpp> after the existing sparsepress includes
old_include = '#include <sparsepress/format/header_v3.hpp>'
new_include = old_include + '\n#include <FactorNet/core/platform.hpp>  // get_available_ram_bytes()'

content = content.replace(old_include, new_include, 1)

# Add Rcpp_get_available_ram_mb() before Rcpp_st_add_transpose
ram_func = """
// =============================================================================
// RAM detection: expose platform-conditional RAM query to R
// =============================================================================

//' @title Get available system RAM in megabytes
//' @return Double: available RAM in MB
//' @keywords internal
// [[Rcpp::export]]
double Rcpp_get_available_ram_mb() {
    return static_cast<double>(FactorNet::get_available_ram_bytes()) / (1024.0 * 1024.0);
}

"""

# Insert before the st_add_transpose section
marker = "// =============================================================================\n// st_add_transpose:"
content = content.replace(marker, ram_func + marker, 1)

with open(path, "w") as f:
    f.write(content)

print("OK: Added Rcpp_get_available_ram_mb() to sparsepress_bridge.cpp")
