#!/usr/bin/env python3
"""Fix sparsepress_bridge.cpp with all needed changes:
1. Add obs_raw/var_raw params to Rcpp_sp_write
2. Add obs/var buffer handling to v2 write path
3. Add has_obs/has_var/transp_chunk_cols to Rcpp_sp_metadata
Run on compute node to bypass NFS caching."""

path = "/mnt/home/debruinz/RcppML-2/src/sparsepress_bridge.cpp"
with open(path, "r") as f:
    content = f.read()

changes = 0

# 1. Add obs_raw/var_raw params to Rcpp_sp_write signature
old_sig = """List Rcpp_sp_write(const S4& A, const std::string& path,
                   bool use_delta = true,
                   bool use_value_pred = false,
                   bool verbose = false,
                   const std::string& precision = "auto",
                   bool row_sort = false,
                   bool include_transpose = false,
                   int chunk_cols = 2048) {"""

new_sig = """List Rcpp_sp_write(const S4& A, const std::string& path,
                   bool use_delta = true,
                   bool use_value_pred = false,
                   bool verbose = false,
                   const std::string& precision = "auto",
                   bool row_sort = false,
                   bool include_transpose = false,
                   int chunk_cols = 2048,
                   Nullable<RawVector> obs_raw = R_NilValue,
                   Nullable<RawVector> var_raw = R_NilValue) {"""

if old_sig in content:
    content = content.replace(old_sig, new_sig)
    changes += 1
    print("1. Added obs_raw/var_raw to Rcpp_sp_write signature")
elif "obs_raw = R_NilValue" in content:
    print("1. SKIP: obs_raw already in signature")
else:
    print("1. WARNING: Could not find Rcpp_sp_write signature")

# 2. Add use_v2 condition for obs_raw/var_raw
old_use_v2 = """    bool use_v2 = (precision != "auto" && precision != "fp64") ||
                  row_sort || include_transpose ||
                  !rownames.empty() || !colnames.empty();"""

new_use_v2 = """    bool use_v2 = (precision != "auto" && precision != "fp64") ||
                  row_sort || include_transpose ||
                  !rownames.empty() || !colnames.empty() ||
                  obs_raw.isNotNull() || var_raw.isNotNull();"""

if old_use_v2 in content:
    content = content.replace(old_use_v2, new_use_v2)
    changes += 1
    print("2. Updated use_v2 condition to include obs_raw/var_raw")
elif "obs_raw.isNotNull()" in content:
    print("2. SKIP: obs_raw already in use_v2 condition")
else:
    print("2. WARNING: Could not find use_v2 condition")

# 3. Add obs/var buffer handling after cfg2 setup but before compress_v2
old_cfg = """        sparsepress::v2::CompressConfig_v2 cfg2;
        cfg2.precision = precision;
        cfg2.row_sort = row_sort;
        cfg2.include_transpose = include_transpose;
        cfg2.verbose = verbose ? 2 : 0;

        sparsepress::v2::CompressStats_v2 stats2;"""

new_cfg = """        sparsepress::v2::CompressConfig_v2 cfg2;
        cfg2.precision = precision;
        cfg2.row_sort = row_sort;
        cfg2.include_transpose = include_transpose;
        cfg2.verbose = verbose ? 2 : 0;

        // Pass pre-serialized obs/var table buffers
        if (obs_raw.isNotNull()) {
            RawVector obs_rv(obs_raw);
            cfg2.obs_buf.assign(obs_rv.begin(), obs_rv.end());
        }
        if (var_raw.isNotNull()) {
            RawVector var_rv(var_raw);
            cfg2.var_buf.assign(var_rv.begin(), var_rv.end());
        }

        sparsepress::v2::CompressStats_v2 stats2;"""

if old_cfg in content:
    content = content.replace(old_cfg, new_cfg)
    changes += 1
    print("3. Added obs/var buffer handling to v2 write path")
elif "cfg2.obs_buf.assign" in content:
    print("3. SKIP: obs_buf handling already present")
else:
    print("3. WARNING: Could not find cfg2 setup block")

# 4. Add has_obs/has_var to v2 write return value
old_v2_ret = """        return List::create(
            Named("raw_bytes") = static_cast<double>(stats2.raw_size),
            Named("compressed_bytes") = static_cast<double>(stats2.compressed_size),
            Named("ratio") = stats2.ratio(),
            Named("compress_ms") = stats2.compress_time_ms,
            Named("num_chunks") = static_cast<int>(stats2.num_chunks),
            Named("version") = 2,
            Named("has_rownames") = !rownames.empty(),
            Named("has_colnames") = !colnames.empty()
        );"""

new_v2_ret = """        return List::create(
            Named("raw_bytes") = static_cast<double>(stats2.raw_size),
            Named("compressed_bytes") = static_cast<double>(stats2.compressed_size),
            Named("ratio") = stats2.ratio(),
            Named("compress_ms") = stats2.compress_time_ms,
            Named("num_chunks") = static_cast<int>(stats2.num_chunks),
            Named("version") = 2,
            Named("has_rownames") = !rownames.empty(),
            Named("has_colnames") = !colnames.empty(),
            Named("has_transpose") = include_transpose,
            Named("has_obs") = obs_raw.isNotNull(),
            Named("has_var") = var_raw.isNotNull()
        );"""

if old_v2_ret in content:
    content = content.replace(old_v2_ret, new_v2_ret)
    changes += 1
    print("4. Added has_transpose/has_obs/has_var to v2 write return")
elif '"has_obs") = obs_raw.isNotNull()' in content:
    print("4. SKIP: has_obs already in v2 write return")
else:
    print("4. WARNING: Could not find v2 write return block")

# 5. Add has_obs/has_var/transp_chunk_cols to Rcpp_sp_metadata v2 return
old_meta = """            Named("row_sorted") = static_cast<bool>(hdr.row_sorted),
            Named("has_transpose") = (hdr.transpose_offset != 0),
            Named("has_metadata") = (hdr.metadata_offset != 0)
        );"""

new_meta = """            Named("row_sorted") = static_cast<bool>(hdr.row_sorted),
            Named("has_transpose") = (hdr.transpose_offset != 0),
            Named("has_metadata") = (hdr.metadata_offset != 0),
            Named("has_obs") = (hdr.obs_table_offset() != 0),
            Named("has_var") = (hdr.var_table_offset() != 0),
            Named("transp_chunk_cols") = static_cast<int>(
                hdr.transp_chunk_cols() > 0 ? hdr.transp_chunk_cols() : hdr.chunk_cols),
            Named("transpose_offset") = static_cast<double>(hdr.transpose_offset)
        );"""

if old_meta in content:
    content = content.replace(old_meta, new_meta)
    changes += 1
    print("5. Added has_obs/has_var/transp_chunk_cols to Rcpp_sp_metadata")
elif 'hdr.obs_table_offset()' in content:
    print("5. SKIP: already has obs_table_offset in metadata return")
else:
    print("5. WARNING: Could not find metadata v2 return block")

# 6. Add roxygen for obs_raw/var_raw before Rcpp_sp_write
old_roxygen = """//' @param chunk_cols Columns per chunk
//' @return A list with compression statistics"""

new_roxygen = """//' @param chunk_cols Columns per chunk
//' @param obs_raw Pre-serialized obs table (raw vector) or NULL
//' @param var_raw Pre-serialized var table (raw vector) or NULL
//' @return A list with compression statistics"""

if old_roxygen in content:
    content = content.replace(old_roxygen, new_roxygen)
    changes += 1
    print("6. Added obs_raw/var_raw roxygen docs")
elif "obs_raw Pre-serialized" in content:
    print("6. SKIP: roxygen already has obs_raw docs")
else:
    print("6. WARNING: Could not find roxygen block")

with open(path, "w") as f:
    f.write(content)

print(f"\nDone. Applied {changes} changes to {path}")
