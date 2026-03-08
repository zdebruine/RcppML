#!/usr/bin/env python3
"""Fix sparsepress_bridge.cpp to add has_obs/has_var/transp_chunk_cols to Rcpp_sp_metadata v2 return."""

path = "/mnt/home/debruinz/RcppML-2/src/sparsepress_bridge.cpp"
with open(path, "r") as f:
    content = f.read()

old = '''            Named("row_sorted") = static_cast<bool>(hdr.row_sorted),
            Named("has_transpose") = (hdr.transpose_offset != 0),
            Named("has_metadata") = (hdr.metadata_offset != 0)
        );'''

new = '''            Named("row_sorted") = static_cast<bool>(hdr.row_sorted),
            Named("has_transpose") = (hdr.transpose_offset != 0),
            Named("has_metadata") = (hdr.metadata_offset != 0),
            Named("has_obs") = (hdr.obs_table_offset() != 0),
            Named("has_var") = (hdr.var_table_offset() != 0),
            Named("transp_chunk_cols") = static_cast<int>(
                hdr.transp_chunk_cols() > 0 ? hdr.transp_chunk_cols() : hdr.chunk_cols),
            Named("transpose_offset") = static_cast<double>(hdr.transpose_offset)
        );'''

if old in content:
    content = content.replace(old, new)
    print("Applied: Added has_obs/has_var/transp_chunk_cols to Rcpp_sp_metadata v2 return")
else:
    print("SKIP: Already applied or pattern not found")

with open(path, "w") as f:
    f.write(content)
print("Done")
