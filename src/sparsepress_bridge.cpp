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
#include <FactorNet/core/platform.hpp>  // get_available_ram_bytes()
#include <streampress/transpose.hpp>

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
                   bool include_transpose = false,
                   int chunk_cols = 2048,
                   Nullable<RawVector> obs_raw = R_NilValue,
                   Nullable<RawVector> var_raw = R_NilValue) {
    sparsepress::CSCMatrix mat = dgc_to_csc(A);

    // Extract dimnames for metadata
    std::vector<std::string> rownames, colnames;
    extract_dimnames(A, rownames, colnames);

    // Always use v2 format — v1 is legacy
    bool use_v2 = true;

    if (use_v2) {
        sparsepress::v2::CompressConfig_v2 cfg2;
        cfg2.precision = precision;
        cfg2.row_sort = row_sort;
        cfg2.include_transpose = include_transpose;
        cfg2.verbose = verbose ? 2 : 0;
        cfg2.chunk_cols = static_cast<uint32_t>(chunk_cols);

        // Populate obs/var buffers from R raw vectors
        if (obs_raw.isNotNull()) {
            RawVector obs_rv(obs_raw.get());
            cfg2.obs_buf.assign(obs_rv.begin(), obs_rv.end());
        }
        if (var_raw.isNotNull()) {
            RawVector var_rv(var_raw.get());
            cfg2.var_buf.assign(var_rv.begin(), var_rv.end());
        }

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
            Named("has_transpose") = include_transpose,
            Named("has_obs") = obs_raw.isNotNull(),
            Named("has_var") = var_raw.isNotNull()
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
            Named("has_metadata") = (hdr.metadata_offset != 0),
            Named("has_obs") = (hdr.obs_table_offset() != 0),
            Named("has_var") = (hdr.var_table_offset() != 0),
            Named("transp_chunk_cols") = static_cast<int>(
                hdr.transp_chunk_cols() > 0 ? hdr.transp_chunk_cols() : hdr.chunk_cols),
            Named("transpose_offset") = static_cast<double>(hdr.transpose_offset)
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
//' @param codec Compression codec: 0=RAW_FP32, 1=FP16, 2=QUANT8, 3=FP16_RANS, 4=FP32_RANS
//' @param delta Apply XOR-delta encoding before entropy coding
//' @return A list with write statistics
//' @keywords internal
// [[Rcpp::export]]
List Rcpp_sp_write_dense(const NumericMatrix& A, const std::string& path,
                         bool include_transpose = false,
                         int chunk_cols = 256,
                         int codec = 0,
                         bool delta = false) {
    uint32_t m = static_cast<uint32_t>(A.nrow());
    uint32_t n = static_cast<uint32_t>(A.ncol());

    // Validate codec
    if (codec < 0 || codec > 4)
        stop("Invalid codec value. Must be 0-4.");

    sparsepress::v3::DenseCodec dc =
        static_cast<sparsepress::v3::DenseCodec>(codec);

    // Convert R's column-major double matrix to float for storage
    std::vector<float> fdata(static_cast<size_t>(m) * n);
    const double* src = A.begin();
    for (size_t k = 0; k < fdata.size(); ++k) {
        fdata[k] = static_cast<float>(src[k]);
    }

    sparsepress::v3::write_v3<float>(
        path, fdata.data(), m, n,
        static_cast<uint32_t>(chunk_cols), include_transpose, dc, delta);

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
        Named("has_transpose") = include_transpose,
        Named("codec") = sparsepress::v3::dense_codec_name(dc),
        Named("delta") = delta
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

// =============================================================================
// Obs/var table serialization and reading
// =============================================================================

// Helper: serialize an R data.frame to obs_var_table binary format
static std::vector<uint8_t> serialize_dataframe(const DataFrame& df) {
    using namespace sparsepress::v2;

    int ncols = df.size();
    int nrows = df.nrows();
    CharacterVector colnames = df.names();

    std::vector<ColumnData> columns(ncols);

    for (int j = 0; j < ncols; ++j) {
        ColumnData& col = columns[j];
        col.name = as<std::string>(colnames[j]);
        SEXP s = df[j];

        if (Rf_isLogical(s)) {
            col.type = ColType::BOOL;
            LogicalVector lv(s);
            col.bool_data.resize(nrows);
            for (int i = 0; i < nrows; ++i)
                col.bool_data[i] = LogicalVector::is_na(lv[i]) ? NA_BOOL : static_cast<uint8_t>(lv[i]);
        } else if (Rf_isFactor(s)) {
            col.type = ColType::STRING_DICT;
            IntegerVector factor_codes(s);
            CharacterVector levels = factor_codes.attr("levels");
            col.dict.resize(levels.size());
            for (int k = 0; k < levels.size(); ++k)
                col.dict[k] = as<std::string>(levels[k]);
            col.codes.resize(nrows);
            for (int i = 0; i < nrows; ++i) {
                if (IntegerVector::is_na(factor_codes[i]))
                    col.codes[i] = NA_UINT32;
                else
                    col.codes[i] = static_cast<uint32_t>(factor_codes[i] - 1);
            }
        } else if (Rf_isInteger(s)) {
            col.type = ColType::INT32;
            IntegerVector iv(s);
            col.int_data.resize(nrows);
            for (int i = 0; i < nrows; ++i)
                col.int_data[i] = IntegerVector::is_na(iv[i]) ? NA_INT32 : iv[i];
        } else if (Rf_isReal(s)) {
            col.type = ColType::FLOAT64;
            NumericVector nv(s);
            col.dbl_data.resize(nrows);
            for (int i = 0; i < nrows; ++i)
                col.dbl_data[i] = NumericVector::is_na(nv[i]) ? na_float64() : nv[i];
        } else if (Rf_isString(s)) {
            col.type = ColType::STRING_DICT;
            CharacterVector sv(s);
            std::map<std::string, uint32_t> str_to_idx;
            col.codes.resize(nrows);
            for (int i = 0; i < nrows; ++i) {
                if (CharacterVector::is_na(sv[i])) {
                    col.codes[i] = NA_UINT32;
                } else {
                    std::string str_val = as<std::string>(sv[i]);
                    auto it = str_to_idx.find(str_val);
                    if (it == str_to_idx.end()) {
                        uint32_t idx = static_cast<uint32_t>(col.dict.size());
                        str_to_idx[str_val] = idx;
                        col.dict.push_back(str_val);
                        col.codes[i] = idx;
                    } else {
                        col.codes[i] = it->second;
                    }
                }
            }
        } else {
            stop("Unsupported column type for obs/var table serialization");
        }
    }

    return obs_var_table_serialize(static_cast<uint32_t>(nrows), columns);
}

// Helper: deserialize obs_var_table buffer to R DataFrame
static DataFrame deserialize_to_dataframe(const uint8_t* buf, size_t buf_bytes) {
    using namespace sparsepress::v2;

    std::vector<ColumnData> columns = obs_var_table_deserialize(buf, buf_bytes);

    if (columns.empty()) return DataFrame::create();

    uint32_t n_rows_out = 0;
    const auto& c0 = columns[0];
    switch (c0.type) {
        case ColType::INT32:      n_rows_out = c0.int_data.size(); break;
        case ColType::FLOAT32:    n_rows_out = c0.flt_data.size(); break;
        case ColType::FLOAT64:    n_rows_out = c0.dbl_data.size(); break;
        case ColType::BOOL:       n_rows_out = c0.bool_data.size(); break;
        case ColType::UINT32:     n_rows_out = c0.uint_data.size(); break;
        case ColType::STRING_DICT: n_rows_out = c0.codes.size(); break;
    }

    List result(columns.size());
    CharacterVector names(columns.size());

    for (size_t j = 0; j < columns.size(); ++j) {
        const auto& col = columns[j];
        names[j] = col.name;
        switch (col.type) {
            case ColType::INT32: {
                IntegerVector iv(col.int_data.size());
                for (size_t i = 0; i < col.int_data.size(); ++i)
                    iv[i] = (col.int_data[i] == NA_INT32) ? NA_INTEGER : col.int_data[i];
                result[j] = iv;
                break;
            }
            case ColType::FLOAT32: {
                NumericVector nv(col.flt_data.size());
                for (size_t i = 0; i < col.flt_data.size(); ++i)
                    nv[i] = std::isnan(col.flt_data[i]) ? NA_REAL : static_cast<double>(col.flt_data[i]);
                result[j] = nv;
                break;
            }
            case ColType::FLOAT64: {
                NumericVector nv(col.dbl_data.size());
                for (size_t i = 0; i < col.dbl_data.size(); ++i)
                    nv[i] = std::isnan(col.dbl_data[i]) ? NA_REAL : col.dbl_data[i];
                result[j] = nv;
                break;
            }
            case ColType::BOOL: {
                LogicalVector lv(col.bool_data.size());
                for (size_t i = 0; i < col.bool_data.size(); ++i)
                    lv[i] = (col.bool_data[i] == NA_BOOL) ? NA_LOGICAL : static_cast<int>(col.bool_data[i]);
                result[j] = lv;
                break;
            }
            case ColType::UINT32: {
                IntegerVector iv(col.uint_data.size());
                for (size_t i = 0; i < col.uint_data.size(); ++i)
                    iv[i] = (col.uint_data[i] == NA_UINT32) ? NA_INTEGER : static_cast<int>(col.uint_data[i]);
                result[j] = iv;
                break;
            }
            case ColType::STRING_DICT: {
                CharacterVector r_levels(col.dict.size());
                for (size_t k = 0; k < col.dict.size(); ++k)
                    r_levels[k] = col.dict[k];
                IntegerVector factor_codes(col.codes.size());
                for (size_t i = 0; i < col.codes.size(); ++i)
                    factor_codes[i] = (col.codes[i] == NA_UINT32) ? NA_INTEGER : static_cast<int>(col.codes[i] + 1);
                factor_codes.attr("levels") = r_levels;
                factor_codes.attr("class") = "factor";
                result[j] = factor_codes;
                break;
            }
        }
    }

    result.attr("names") = names;
    result.attr("class") = "data.frame";
    result.attr("row.names") = IntegerVector::create(NA_INTEGER, -static_cast<int>(n_rows_out));
    return as<DataFrame>(result);
}

// Helper: read a table blob from file at given offset
static DataFrame read_table_at_offset(const std::string& path, uint64_t table_off) {
    using namespace sparsepress::v2;

    if (table_off == 0) return DataFrame::create();

    FILE* f = fopen(path.c_str(), "rb");
    if (!f) stop("Cannot open file: " + path);

    fseek(f, static_cast<long>(table_off), SEEK_SET);
    uint8_t tbl_hdr_buf[16];
    if (fread(tbl_hdr_buf, 1, 16, f) != 16) {
        fclose(f);
        stop("Failed to read table header");
    }

    ObsVarTableHeader tbl_hdr;
    std::memcpy(tbl_hdr.magic, tbl_hdr_buf, 4);
    std::memcpy(&tbl_hdr.n_rows, tbl_hdr_buf + 4, 4);
    std::memcpy(&tbl_hdr.n_cols, tbl_hdr_buf + 8, 4);
    std::memcpy(&tbl_hdr.header_bytes, tbl_hdr_buf + 12, 4);

    if (tbl_hdr.magic[0] != 'O' || tbl_hdr.magic[1] != 'V' ||
        tbl_hdr.magic[2] != 'T' || tbl_hdr.magic[3] != 'B') {
        fclose(f);
        stop("Invalid obs/var table magic bytes");
    }

    size_t desc_bytes = static_cast<size_t>(tbl_hdr.n_cols) * sizeof(ColDescriptor);
    std::vector<uint8_t> desc_buf(desc_bytes);
    if (fread(desc_buf.data(), 1, desc_bytes, f) != desc_bytes) {
        fclose(f);
        stop("Failed to read table descriptors");
    }

    uint64_t max_end = 0;
    for (uint32_t i = 0; i < tbl_hdr.n_cols; ++i) {
        ColDescriptor desc;
        std::memcpy(&desc, desc_buf.data() + i * sizeof(ColDescriptor), sizeof(ColDescriptor));
        uint64_t col_end = desc.data_offset;
        ColType ct = static_cast<ColType>(desc.col_type);
        switch (ct) {
            case ColType::INT32:   col_end += static_cast<uint64_t>(tbl_hdr.n_rows) * 4; break;
            case ColType::FLOAT32: col_end += static_cast<uint64_t>(tbl_hdr.n_rows) * 4; break;
            case ColType::FLOAT64: col_end += static_cast<uint64_t>(tbl_hdr.n_rows) * 8; break;
            case ColType::BOOL:    col_end += tbl_hdr.n_rows; break;
            case ColType::UINT32:  col_end += static_cast<uint64_t>(tbl_hdr.n_rows) * 4; break;
            case ColType::STRING_DICT:
                col_end += static_cast<uint64_t>(tbl_hdr.n_rows) * 4;
                if (desc.dict_bytes > 0) {
                    uint64_t dict_end = desc.dict_offset + desc.dict_bytes;
                    if (dict_end > max_end) max_end = dict_end;
                }
                break;
        }
        if (col_end > max_end) max_end = col_end;
    }

    size_t hdr_plus_desc = 16 + desc_bytes;
    size_t total_bytes = std::max(hdr_plus_desc, static_cast<size_t>(max_end));
    std::vector<uint8_t> full_buf(total_bytes);
    fseek(f, static_cast<long>(table_off), SEEK_SET);
    if (fread(full_buf.data(), 1, total_bytes, f) != total_bytes) {
        fclose(f);
        stop("Failed to read table data");
    }
    fclose(f);

    return deserialize_to_dataframe(full_buf.data(), total_bytes);
}


//' @title Serialize a data.frame to obs/var table binary format
//' @param df A data.frame
//' @return A raw vector containing the serialized table
//' @keywords internal
// [[Rcpp::export]]
RawVector Rcpp_st_serialize_table(const DataFrame& df) {
    std::vector<uint8_t> buf = serialize_dataframe(df);
    RawVector out(buf.size());
    if (!buf.empty())
        std::memcpy(out.begin(), buf.data(), buf.size());
    return out;
}

//' @title Read obs table from a v2 .spz file
//' @param path Path to .spz file
//' @return A data.frame, or empty data.frame if no obs table
//' @keywords internal
// [[Rcpp::export]]
DataFrame Rcpp_st_read_obs(const std::string& path) {
    using namespace sparsepress::v2;

    FILE* f = fopen(path.c_str(), "rb");
    if (!f) stop("Cannot open file: " + path);

    uint8_t hdr_buf[HEADER_SIZE_V2];
    if (fread(hdr_buf, 1, HEADER_SIZE_V2, f) != HEADER_SIZE_V2) {
        fclose(f);
        stop("Failed to read header");
    }
    fclose(f);

    FileHeader_v2 hdr = FileHeader_v2::deserialize(hdr_buf);
    if (!hdr.valid() || hdr.version != 2)
        stop("Not a valid v2 .spz file");

    return read_table_at_offset(path, hdr.obs_table_offset());
}

//' @title Read var table from a v2 .spz file
//' @param path Path to .spz file
//' @return A data.frame, or empty data.frame if no var table
//' @keywords internal
// [[Rcpp::export]]
DataFrame Rcpp_st_read_var(const std::string& path) {
    using namespace sparsepress::v2;

    FILE* f = fopen(path.c_str(), "rb");
    if (!f) stop("Cannot open file: " + path);

    uint8_t hdr_buf[HEADER_SIZE_V2];
    if (fread(hdr_buf, 1, HEADER_SIZE_V2, f) != HEADER_SIZE_V2) {
        fclose(f);
        stop("Failed to read header");
    }
    fclose(f);

    FileHeader_v2 hdr = FileHeader_v2::deserialize(hdr_buf);
    if (!hdr.valid() || hdr.version != 2)
        stop("Not a valid v2 .spz file");

    return read_table_at_offset(path, hdr.var_table_offset());
}
