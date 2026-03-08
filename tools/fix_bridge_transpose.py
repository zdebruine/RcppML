#!/usr/bin/env python3
"""Add st_add_transpose Rcpp bridge function."""

BRIDGE = "/mnt/home/debruinz/RcppML-2/src/sparsepress_bridge.cpp"

# Read current content
with open(BRIDGE) as f:
    content = f.read()

# Add include for transpose.hpp after the existing includes
old_include = '#include <sparsepress/format/header_v3.hpp>'
new_include = old_include + '\n#include <streampress/transpose.hpp>'
if '#include <streampress/transpose.hpp>' not in content:
    content = content.replace(old_include, new_include)
    print("[1/2] Added #include <streampress/transpose.hpp>")
else:
    print("[1/2] Include already present")

# Append Rcpp function at end
FUNC = r'''

// =============================================================================
// st_add_transpose: add transpose section to existing v2 .spz file
// =============================================================================

//' @title Add transpose section to an existing v2 .spz file
//' @param path Path to the .spz v2 file
//' @param verbose Logical; print progress
//' @return Logical TRUE on success
//' @keywords internal
// [[Rcpp::export]]
bool Rcpp_st_add_transpose(const std::string& path, bool verbose = true) {
    return streampress::add_transpose(path, verbose);
}
'''

if 'Rcpp_st_add_transpose' not in content:
    content += FUNC
    print("[2/2] Added Rcpp_st_add_transpose function")
else:
    print("[2/2] Function already present")

with open(BRIDGE, 'w') as f:
    f.write(content)
print("[OK] Bridge updated")
