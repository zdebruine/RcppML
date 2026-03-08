#!/usr/bin/env python3
"""Fix sparsepress_bridge.cpp to add codec/delta params to Rcpp_sp_write_dense."""
import sys

path = "src/sparsepress_bridge.cpp"
with open(path, "r") as f:
    content = f.read()

# 1. Replace function signature
content = content.replace(
    """List Rcpp_sp_write_dense(const NumericMatrix& A, const std::string& path,
                         bool include_transpose = false,
                         int chunk_cols = 256) {""",
    """List Rcpp_sp_write_dense(const NumericMatrix& A, const std::string& path,
                         bool include_transpose = false,
                         int chunk_cols = 256,
                         int codec = 0,
                         bool delta = false) {""")

# 2. Add codec validation
content = content.replace(
    """    uint32_t m = static_cast<uint32_t>(A.nrow());
    uint32_t n = static_cast<uint32_t>(A.ncol());

    // Convert R's column-major double matrix to float for storage
    // (R uses double internally, SPZ v3 default is float32)
    std::vector<float> fdata(static_cast<size_t>(m) * n);""",
    """    uint32_t m = static_cast<uint32_t>(A.nrow());
    uint32_t n = static_cast<uint32_t>(A.ncol());

    // Validate codec
    if (codec < 0 || codec > 4)
        stop("Invalid codec value. Must be 0-4.");

    sparsepress::v3::DenseCodec dc =
        static_cast<sparsepress::v3::DenseCodec>(codec);

    // Convert R's column-major double matrix to float for storage
    std::vector<float> fdata(static_cast<size_t>(m) * n);""")

# 3. Update write_v3 call
content = content.replace(
    """    sparsepress::v3::write_v3<float>(
        path, fdata.data(), m, n,
        static_cast<uint32_t>(chunk_cols), include_transpose);""",
    """    sparsepress::v3::write_v3<float>(
        path, fdata.data(), m, n,
        static_cast<uint32_t>(chunk_cols), include_transpose, dc, delta);""")

# 4. Update return list
content = content.replace(
    """    return List::create(
        Named("raw_bytes") = raw_bytes,
        Named("file_bytes") = file_bytes,
        Named("version") = 3,
        Named("rows") = static_cast<int>(m),
        Named("cols") = static_cast<int>(n),
        Named("num_chunks") = static_cast<int>(num_chunks),
        Named("has_transpose") = include_transpose
    );""",
    """    return List::create(
        Named("raw_bytes") = raw_bytes,
        Named("file_bytes") = file_bytes,
        Named("version") = 3,
        Named("rows") = static_cast<int>(m),
        Named("cols") = static_cast<int>(n),
        Named("num_chunks") = static_cast<int>(num_chunks),
        Named("has_transpose") = include_transpose,
        Named("codec") = sparsepress::v3::dense_codec_name(dc),
        Named("delta") = delta
    );""")

# 5. Update roxygen docs
content = content.replace(
    """//' @param chunk_cols Columns per chunk (default 256)
//' @return A list with write statistics""",
    """//' @param chunk_cols Columns per chunk (default 256)
//' @param codec Compression codec: 0=RAW_FP32, 1=FP16, 2=QUANT8, 3=FP16_RANS, 4=FP32_RANS
//' @param delta Apply XOR-delta encoding before entropy coding
//' @return A list with write statistics""")

with open(path, "w") as f:
    f.write(content)

count = content.count("codec")
print(f"sparsepress_bridge.cpp updated. 'codec' appears {count} times.")
if count == 0:
    print("WARNING: no replacements made!", file=sys.stderr)
    sys.exit(1)
