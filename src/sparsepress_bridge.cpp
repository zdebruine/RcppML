/**
 * @file sparsepress_bridge.cpp
 * @brief Rcpp bridge for SparsePress compression/decompression.
 *
 * Converts between R's dgCMatrix and sparsepress::CSCMatrix,
 * then delegates to the header-only sparsepress library.
 * Supports both v1 (monolithic) and v2 (chunked) formats.
 */

#include <Rcpp.h>

// Redirect sparsepress verbose logging to R's message stream.
// This avoids the CRAN NOTE about compiled code writing to stderr.
#define SPARSEPRESS_LOG(...) REprintf(__VA_ARGS__)

#include <sparsepress/sparsepress.hpp>
#include <sparsepress/sparsepress_v2.hpp>
#include <sparsepress/sparsepress_v3.hpp>
#include <sparsepress/format/header_v2.hpp>
#include <sparsepress/format/header_v3.hpp>

using namespace Rcpp;

// =============================================================================
// Helper: R dgCMatrix → sparsepress CSCMatrix
// =============================================================================
static sparsepress::CSCMatrix dgc_to_csc(const S4& dgc) {
    IntegerVector dims = dgc.slot("Dim");
    IntegerVector p = dgc.slot("p");
    IntegerVector i = dgc.slot("i");
    NumericVector x = dgc.slot("x");

    uint32_t m = static_cast<uint32_t>(dims[0]);
    uint32_t n = static_cast<uint32_t>(dims[1]);
    uint64_t nnz = static_cast<uint64_t>(x.size());

    sparsepress::CSCMatrix mat(m, n, nnz);

    for (uint32_t j = 0; j <= n; ++j)
        mat.p[j] = static_cast<uint32_t>(p[j]);
    for (uint64_t k = 0; k < nnz; ++k) {
        mat.i[k] = static_cast<uint32_t>(i[k]);
        mat.x[k] = x[k];
    }

    return mat;
}

// =============================================================================
// Helper: Extract dimnames from dgCMatrix as string vectors
// =============================================================================
static void extract_dimnames(const S4& dgc,
                             std::vector<std::string>& rownames,
                             std::vector<std::string>& colnames) {
    rownames.clear();
    colnames.clear();

    List dimnames;
    try {
        dimnames = dgc.slot("Dimnames");
    } catch (...) {
        return;
    }
    if (dimnames.size() < 2) return;

    if (!Rf_isNull(dimnames[0])) {
        CharacterVector rn = dimnames[0];
        rownames.reserve(rn.size());
        for (int i = 0; i < rn.size(); ++i)
            rownames.push_back(std::string(rn[i]));
    }
    if (!Rf_isNull(dimnames[1])) {
        CharacterVector cn = dimnames[1];
        colnames.reserve(cn.size());
        for (int i = 0; i < cn.size(); ++i)
            colnames.push_back(std::string(cn[i]));
    }
}

// =============================================================================
// Helper: sparsepress CSCMatrix → R dgCMatrix (with optional dimnames)
// =============================================================================
static S4 csc_to_dgc(const sparsepress::CSCMatrix& mat,
                      const std::vector<std::string>& rownames = {},
                      const std::vector<std::string>& colnames = {}) {
    IntegerVector dims = IntegerVector::create(
        static_cast<int>(mat.m), static_cast<int>(mat.n));

    IntegerVector p(mat.n + 1);
    for (uint32_t j = 0; j <= mat.n; ++j)
        p[j] = static_cast<int>(mat.p[j]);

    IntegerVector i(mat.nnz);
    NumericVector x(mat.nnz);
    for (uint64_t k = 0; k < mat.nnz; ++k) {
        i[k] = static_cast<int>(mat.i[k]);
        x[k] = mat.x[k];
    }

    S4 dgc("dgCMatrix");
    dgc.slot("Dim") = dims;
    dgc.slot("p") = p;
    dgc.slot("i") = i;
    dgc.slot("x") = x;

    // Restore dimnames if available
    SEXP rn = R_NilValue;
    SEXP cn = R_NilValue;
    if (!rownames.empty()) {
        CharacterVector rnames(rownames.size());
        for (size_t idx = 0; idx < rownames.size(); ++idx)
            rnames[idx] = rownames[idx];
        rn = rnames;
    }
    if (!colnames.empty()) {
        CharacterVector cnames(colnames.size());
        for (size_t idx = 0; idx < colnames.size(); ++idx)
            cnames[idx] = colnames[idx];
        cn = cnames;
    }
    dgc.slot("Dimnames") = List::create(rn, cn);
    dgc.slot("factors") = List::create();

    return dgc;
}

// =============================================================================
// Rcpp exports
// =============================================================================

//' @title Write a dgCMatrix to a SparsePress (.spz) file
//' @param A A sparse matrix (dgCMatrix)
//' @param path Output file path (.spz)
//' @param use_delta Use delta prediction (better for structured data)
//' @param use_value_pred Use value prediction (for integer data)
//' @param verbose Print compression stats
//' @param precision Value precision: "auto", "fp32", "fp16", "quant8", "fp64"
//' @param row_sort Sort rows by nnz for better compression
//' @param include_transpose Also store CSC(A^T) in file
//' @return A list with compression statistics
//' @keywords internal
// [[Rcpp::export]]
List Rcpp_sp_write(const S4& A, const std::string& path,
                   bool use_delta = true,
                   bool use_value_pred = false,
                   bool verbose = false,
                   const std::string& precision = "auto",
                   bool row_sort = false,
                   bool include_transpose = false) {
    sparsepress::CSCMatrix mat = dgc_to_csc(A);

    // Extract dimnames for metadata
    std::vector<std::string> rownames, colnames;
    extract_dimnames(A, rownames, colnames);

    // Use v2 if any v2-specific features are requested
    bool use_v2 = (precision != "auto" && precision != "fp64") ||
                  row_sort || include_transpose ||
                  !rownames.empty() || !colnames.empty();

    if (use_v2) {
        sparsepress::v2::CompressConfig_v2 cfg2;
        cfg2.precision = precision;
        cfg2.row_sort = row_sort;
        cfg2.include_transpose = include_transpose;
        cfg2.verbose = verbose ? 2 : 0;

        sparsepress::v2::CompressStats_v2 stats2;
        std::vector<uint8_t> compressed = sparsepress::v2::compress_v2(mat, cfg2, &stats2);
        sparsepress::v2::write_v2(path, compressed);

        return List::create(
            Named("raw_bytes") = static_cast<double>(stats2.raw_size),
            Named("compressed_bytes") = static_cast<double>(stats2.compressed_size),
            Named("ratio") = stats2.ratio(),
            Named("compress_ms") = stats2.compress_time_ms,
            Named("num_chunks") = static_cast<int>(stats2.num_chunks),
            Named("version") = 2,
            Named("has_rownames") = !rownames.empty(),
            Named("has_colnames") = !colnames.empty(),
            Named("has_transpose") = include_transpose
        );
    }

    // v1 path (legacy)
    sparsepress::CompressConfig cfg;
    cfg.use_delta_prediction = use_delta;
    cfg.use_value_prediction = use_value_pred;
    cfg.verbose = verbose;

    sparsepress::CompressStats stats;
    std::vector<uint8_t> compressed = sparsepress::compress(mat, cfg, &stats);
    sparsepress::write_compressed(path, compressed);

    return List::create(
        Named("raw_bytes") = static_cast<double>(stats.raw_size),
        Named("compressed_bytes") = static_cast<double>(stats.compressed_size),
        Named("ratio") = stats.ratio(),
        Named("compress_ms") = stats.compress_time_ms,
        Named("version") = 1,
        Named("has_rownames") = !rownames.empty(),
        Named("has_colnames") = !colnames.empty()
    );
}

//' @title Read a SparsePress (.spz) file into a dgCMatrix
//' @param path Input file path (.spz)
//' @param cols Optional integer vector of column indices to read (1-indexed)
//' @param reorder Whether to undo row permutation (default TRUE)
//' @return A sparse matrix (dgCMatrix)
//' @keywords internal
// [[Rcpp::export]]
S4 Rcpp_sp_read(const std::string& path,
                Nullable<IntegerVector> cols = R_NilValue,
                bool reorder = true) {
    std::vector<uint8_t> data = sparsepress::read_compressed(path);

    // Check version to decide decode path
    if (data.size() >= 6) {
        uint16_t version;
        std::memcpy(&version, data.data() + 4, 2);
        if (version == 2) {
            // v2 path: chunked decode with metadata support
            sparsepress::v2::DecompressConfig_v2 dcfg;
            dcfg.reorder = reorder;

            if (cols.isNotNull()) {
                IntegerVector col_vec(cols);
                // Convert from R 1-indexed to C++ 0-indexed
                dcfg.col_start = col_vec[0] - 1;
                dcfg.col_end = col_vec[col_vec.size() - 1];  // exclusive
            }

            sparsepress::v2::Metadata meta;
            sparsepress::CSCMatrix mat = sparsepress::v2::decompress_v2(
                data.data(), data.size(), dcfg, &meta);

            // Extract dimnames from metadata
            auto rownames = meta.get_rownames();
            auto colnames = meta.get_colnames();

            return csc_to_dgc(mat, rownames, colnames);
        }
    }

    sparsepress::CompressStats stats;
    sparsepress::CSCMatrix mat = sparsepress::decompress(data, &stats);

    // v1 files don't store dimnames, return with NULL dimnames
    return csc_to_dgc(mat);
}

//' @title Read the pre-stored transpose from a v2 SparsePress (.spz) file
//' @param path Input file path (.spz)
//' @return A sparse matrix (dgCMatrix) containing CSC(A^T)
//' @keywords internal
// [[Rcpp::export]]
S4 Rcpp_sp_read_transpose(const std::string& path) {
    std::vector<uint8_t> data = sparsepress::v2::read_v2(path);

    if (!sparsepress::v2::has_transpose(data.data(), data.size())) {
        stop("File does not contain a pre-stored transpose. "
             "Use sp_write(..., include_transpose = TRUE) to create one.");
    }

    sparsepress::v2::DecompressConfig_v2 dcfg;
    sparsepress::CSCMatrix mat = sparsepress::v2::decompress_v2_transpose(
        data.data(), data.size(), dcfg);

    return csc_to_dgc(mat);
}

//' @title Get metadata from a SparsePress (.spz) file without decompressing
//' @param path Input file path (.spz)
//' @return A list with matrix metadata
//' @keywords internal
// [[Rcpp::export]]
List Rcpp_sp_metadata(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) stop("Cannot open file: " + path);

    // Get file size
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    // Read enough bytes for either v1 (72) or v2 (128) header
    const size_t max_header = 128;
    if (file_size < 6) {
        fclose(f);
        stop("File too small to be a valid .spz file");
    }

    uint8_t header_bytes[128];
    size_t read_bytes = std::min(static_cast<size_t>(file_size), max_header);
    if (fread(header_bytes, 1, read_bytes, f) != read_bytes) {
        fclose(f);
        stop("Failed to read header");
    }
    fclose(f);

    // Verify magic
    if (header_bytes[0] != 'S' || header_bytes[1] != 'P' ||
        header_bytes[2] != 'R' || header_bytes[3] != 'Z') {
        stop("Not a valid SparsePress file (bad magic bytes)");
    }

    // Check version
    uint16_t version;
    std::memcpy(&version, header_bytes + 4, 2);

    if (version == 3 && read_bytes >= sparsepress::v3::HEADER_SIZE_V3) {
        // v3 header (dense format)
        sparsepress::v3::FileHeader_v3 hdr =
            sparsepress::v3::FileHeader_v3::deserialize(header_bytes);

        double raw_bytes = static_cast<double>(hdr.m) * hdr.n *
            sparsepress::v3::dense_value_bytes(
                static_cast<sparsepress::v3::DenseValueType>(hdr.value_type));
        double ratio = (file_size > 0) ?
            raw_bytes / file_size : 0.0;

        return List::create(
            Named("rows") = static_cast<int>(hdr.m),
            Named("cols") = static_cast<int>(hdr.n),
            Named("nnz") = static_cast<double>(hdr.nnz),
            Named("density_pct") = 100.0,
            Named("file_bytes") = static_cast<double>(file_size),
            Named("raw_bytes") = raw_bytes,
            Named("ratio") = ratio,
            Named("version") = 3,
            Named("value_type") = std::string(
                sparsepress::v3::dense_value_type_name(
                    static_cast<sparsepress::v3::DenseValueType>(hdr.value_type))),
            Named("chunk_cols") = static_cast<int>(hdr.chunk_cols),
            Named("num_chunks") = static_cast<int>(hdr.num_chunks),
            Named("has_transpose") = static_cast<bool>(hdr.has_transpose),
            Named("has_metadata") = (hdr.metadata_offset != 0)
        );
    }

    if (version == 2 && read_bytes >= sparsepress::v2::HEADER_SIZE_V2) {
        // v2 header
        sparsepress::v2::FileHeader_v2 hdr =
            sparsepress::v2::FileHeader_v2::deserialize(header_bytes);

        size_t raw_size = 12 +
            (static_cast<size_t>(hdr.n) + 1) * 4 +
            static_cast<size_t>(hdr.nnz) * 4 +
            static_cast<size_t>(hdr.nnz) *
                sparsepress::v2::value_type_bytes(
                    static_cast<sparsepress::v2::ValueType_v2>(hdr.value_type));
        double ratio = (file_size > 0) ?
            static_cast<double>(raw_size) / file_size : 0.0;

        return List::create(
            Named("rows") = static_cast<int>(hdr.m),
            Named("cols") = static_cast<int>(hdr.n),
            Named("nnz") = static_cast<double>(hdr.nnz),
            Named("density_pct") = static_cast<double>(hdr.density) * 100.0,
            Named("file_bytes") = static_cast<double>(file_size),
            Named("raw_bytes") = static_cast<double>(raw_size),
            Named("ratio") = ratio,
            Named("version") = static_cast<int>(hdr.version),
            Named("value_type") = std::string(sparsepress::v2::value_type_name(
                static_cast<sparsepress::v2::ValueType_v2>(hdr.value_type))),
            Named("chunk_cols") = static_cast<int>(hdr.chunk_cols),
            Named("num_chunks") = static_cast<int>(hdr.num_chunks),
            Named("row_sorted") = static_cast<bool>(hdr.row_sorted),
            Named("has_transpose") = (hdr.transpose_offset != 0),
            Named("has_metadata") = (hdr.metadata_offset != 0)
        );
    }

    // v1 header
    if (file_size < static_cast<long>(sparsepress::HEADER_SIZE)) {
        stop("File too small to be a valid v1 .spz file");
    }

    sparsepress::FileHeader header = sparsepress::FileHeader::deserialize(header_bytes);

    double density = (header.m > 0 && header.n > 0)
        ? static_cast<double>(header.nnz) / (static_cast<double>(header.m) * header.n) * 100.0
        : 0.0;

    bool has_delta = (header.flags & sparsepress::FLAG_DELTA_PREDICTION) != 0;
    bool has_value = (header.flags & sparsepress::FLAG_VALUE_PREDICTION) != 0;
    bool is_integer = (header.flags & sparsepress::FLAG_INTEGER_VALUES) != 0;

    size_t raw_size = 12 +
        (static_cast<size_t>(header.n) + 1) * 4 +
        static_cast<size_t>(header.nnz) * 4 +
        static_cast<size_t>(header.nnz) * 8;
    double ratio = (file_size > 0)
        ? static_cast<double>(raw_size) / file_size
        : 0.0;

    return List::create(
        Named("rows") = static_cast<int>(header.m),
        Named("cols") = static_cast<int>(header.n),
        Named("nnz") = static_cast<double>(header.nnz),
        Named("density_pct") = density,
        Named("file_bytes") = static_cast<double>(file_size),
        Named("raw_bytes") = static_cast<double>(raw_size),
        Named("ratio") = ratio,
        Named("version") = static_cast<int>(header.version),
        Named("integer_values") = is_integer,
        Named("delta_prediction") = has_delta,
        Named("value_prediction") = has_value
    );
}

//' @title Convert a dgCMatrix to SparsePress format (in-memory)
//' @param A A sparse matrix (dgCMatrix)
//' @param use_delta Use delta prediction
//' @param use_value_pred Use value prediction
//' @return A raw vector containing the compressed data
//' @keywords internal
// [[Rcpp::export]]
RawVector Rcpp_sp_compress(const S4& A,
                           bool use_delta = true,
                           bool use_value_pred = false) {
    sparsepress::CSCMatrix mat = dgc_to_csc(A);

    sparsepress::CompressConfig cfg;
    cfg.use_delta_prediction = use_delta;
    cfg.use_value_prediction = use_value_pred;

    std::vector<uint8_t> compressed = sparsepress::compress(mat, cfg);

    RawVector out(compressed.size());
    std::memcpy(out.begin(), compressed.data(), compressed.size());
    return out;
}

//' @title Decompress a SparsePress raw vector to a dgCMatrix
//' @param data A raw vector containing compressed SparsePress data
//' @return A sparse matrix (dgCMatrix)
//' @keywords internal
// [[Rcpp::export]]
S4 Rcpp_sp_decompress(const RawVector& data) {
    std::vector<uint8_t> compressed(data.begin(), data.end());
    sparsepress::CSCMatrix mat = sparsepress::decompress(compressed);
    return csc_to_dgc(mat);
}

// =============================================================================
// SPZ v3 Dense Format
// =============================================================================

//' @title Write a dense matrix to an SPZ v3 file
//' @param A A numeric matrix (dense)
//' @param path Output file path (.spz)
//' @param include_transpose Also store transposed panels for streaming NMF
//' @param chunk_cols Columns per chunk (default 256)
//' @return A list with write statistics
//' @keywords internal
// [[Rcpp::export]]
List Rcpp_sp_write_dense(const NumericMatrix& A, const std::string& path,
                         bool include_transpose = false,
                         int chunk_cols = 256) {
    uint32_t m = static_cast<uint32_t>(A.nrow());
    uint32_t n = static_cast<uint32_t>(A.ncol());

    // Convert R's column-major double matrix to float for storage
    // (R uses double internally, SPZ v3 default is float32)
    std::vector<float> fdata(static_cast<size_t>(m) * n);
    const double* src = A.begin();
    for (size_t k = 0; k < fdata.size(); ++k) {
        fdata[k] = static_cast<float>(src[k]);
    }

    sparsepress::v3::write_v3<float>(
        path, fdata.data(), m, n,
        static_cast<uint32_t>(chunk_cols), include_transpose);

    uint32_t num_chunks = (n + chunk_cols - 1) / chunk_cols;
    double raw_bytes = static_cast<double>(m) * n * sizeof(float);

    // Get actual file size
    FILE* f = fopen(path.c_str(), "rb");
    double file_bytes = 0;
    if (f) {
        fseek(f, 0, SEEK_END);
        file_bytes = static_cast<double>(ftell(f));
        fclose(f);
    }

    return List::create(
        Named("raw_bytes") = raw_bytes,
        Named("file_bytes") = file_bytes,
        Named("version") = 3,
        Named("rows") = static_cast<int>(m),
        Named("cols") = static_cast<int>(n),
        Named("num_chunks") = static_cast<int>(num_chunks),
        Named("has_transpose") = include_transpose
    );
}

//' @title Read an SPZ v3 dense file into a numeric matrix
//' @param path Input file path (.spz v3)
//' @return A numeric matrix
//' @keywords internal
// [[Rcpp::export]]
NumericMatrix Rcpp_sp_read_dense(const std::string& path) {
    // Read file into memory
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) stop("Cannot open file: " + path);
    fseek(f, 0, SEEK_END);
    size_t file_size = static_cast<size_t>(ftell(f));
    fseek(f, 0, SEEK_SET);

    std::vector<uint8_t> data(file_size);
    if (fread(data.data(), 1, file_size, f) != file_size) {
        fclose(f);
        stop("Failed to read file");
    }
    fclose(f);

    // Verify version
    uint16_t ver = sparsepress::v3::detect_version(data.data(), file_size);
    if (ver != 3) {
        stop("Not an SPZ v3 file (version=" + std::to_string(ver) +
             "). Use sp_read() for v1/v2 sparse files.");
    }

    std::vector<float> fdata;
    uint32_t m, n;
    sparsepress::v3::read_full_matrix<float>(data.data(), file_size, fdata, m, n);

    NumericMatrix result(m, n);
    double* dst = result.begin();
    for (size_t k = 0; k < fdata.size(); ++k) {
        dst[k] = static_cast<double>(fdata[k]);
    }

    return result;
}

//' @title Get metadata from an SPZ v3 dense file
//' @param path Input file path (.spz v3)
//' @return A list with file metadata
//' @keywords internal
// [[Rcpp::export]]
List Rcpp_sp_metadata_v3(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) stop("Cannot open file: " + path);

    uint8_t buf[128];
    size_t nread = fread(buf, 1, 128, f);

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fclose(f);

    if (nread < 8) stop("File too small");

    uint16_t ver = sparsepress::v3::detect_version(buf, nread);
    if (ver != 3) {
        stop("Not an SPZ v3 file (version=" + std::to_string(ver) + ")");
    }

    sparsepress::v3::FileHeader_v3 hdr =
        sparsepress::v3::FileHeader_v3::deserialize(buf);

    double raw_bytes = static_cast<double>(hdr.m) * hdr.n *
        sparsepress::v3::dense_value_bytes(
            static_cast<sparsepress::v3::DenseValueType>(hdr.value_type));

    return List::create(
        Named("rows") = static_cast<int>(hdr.m),
        Named("cols") = static_cast<int>(hdr.n),
        Named("version") = 3,
        Named("value_type") = std::string(
            sparsepress::v3::dense_value_type_name(
                static_cast<sparsepress::v3::DenseValueType>(hdr.value_type))),
        Named("chunk_cols") = static_cast<int>(hdr.chunk_cols),
        Named("num_chunks") = static_cast<int>(hdr.num_chunks),
        Named("has_transpose") = static_cast<bool>(hdr.has_transpose),
        Named("file_bytes") = static_cast<double>(file_size),
        Named("raw_bytes") = raw_bytes
    );
}
