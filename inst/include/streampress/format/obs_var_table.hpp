/**
 * @file obs_var_table.hpp
 * @brief Binary column-store format for obs/var metadata tables in .spz files.
 *
 * Layout in file:
 *   [ObsVarTableHeader: 16 bytes]
 *   [ColDescriptor × n_cols: 112 bytes each]
 *   [raw data blobs for each column]
 *
 * Supported column types: INT32, FLOAT32, FLOAT64, BOOL, UINT32, STRING_DICT.
 * See ColType enum for sentinel NA values.
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace streampress {
namespace v2 {

// ============================================================================
// Sentinel NA values
// ============================================================================

static constexpr int32_t  NA_INT32   = INT32_MIN;
static constexpr uint32_t NA_UINT32  = UINT32_MAX;
// NA_FLOAT32 / NA_FLOAT64 use quiet_NaN; defined as functions to avoid constexpr issues
inline float  na_float32() { return std::numeric_limits<float>::quiet_NaN(); }
inline double na_float64() { return std::numeric_limits<double>::quiet_NaN(); }
static constexpr uint8_t  NA_BOOL    = 255;

// ============================================================================
// Column type enum
// ============================================================================

enum class ColType : uint8_t {
    INT32       = 0,   // raw int32_t array; NA = INT32_MIN
    FLOAT32     = 1,   // raw float array;   NA = NaN
    FLOAT64     = 2,   // raw double array;  NA = NaN
    BOOL        = 3,   // raw uint8_t array (0/1); NA = 255
    UINT32      = 4,   // raw uint32_t array; NA = UINT32_MAX
    STRING_DICT = 5,   // uint32_t code array (NA = UINT32_MAX) + null-delimited dictionary
};

// ============================================================================
// Fixed-size column descriptor: 112 bytes
// ============================================================================

struct ColDescriptor {
    char     name[64];        // null-terminated column name; 63 chars max
    uint8_t  col_type;        // ColType enum
    uint8_t  nullable;        // 1 if column contains NA sentinels
    uint8_t  _pad[2];         // alignment
    uint32_t dict_bytes;      // byte count of null-delimited dict (STRING_DICT only; 0 otherwise)
    uint64_t data_offset;     // byte offset from start of table blob to raw data array
    uint64_t dict_offset;     // byte offset from start of table blob to dictionary blob
    uint8_t  _reserved[20];   // zero, future use
    uint8_t  _align[4];       // pad to 112

    ColDescriptor() { std::memset(this, 0, sizeof(ColDescriptor)); }
};

static_assert(sizeof(ColDescriptor) == 112, "ColDescriptor must be 112 bytes");

// ============================================================================
// Table header: 16 bytes
// ============================================================================

static constexpr uint8_t OBS_VAR_MAGIC[4] = {'O', 'V', 'T', 'B'};

struct ObsVarTableHeader {
    uint8_t  magic[4];        // "OVTB"
    uint32_t n_rows;          // number of rows (cells or genes)
    uint32_t n_cols;          // number of data columns
    uint32_t header_bytes;    // byte size of this header + all descriptors

    ObsVarTableHeader() {
        std::memset(this, 0, sizeof(ObsVarTableHeader));
        std::memcpy(magic, OBS_VAR_MAGIC, 4);
    }

    bool valid() const {
        return std::memcmp(magic, OBS_VAR_MAGIC, 4) == 0;
    }
};

static_assert(sizeof(ObsVarTableHeader) == 16, "ObsVarTableHeader must be 16 bytes");

// ============================================================================
// Column data holder for serialization
// ============================================================================

struct ColumnData {
    std::string name;
    ColType type;
    bool nullable = false;

    // Only one of these is populated, depending on type:
    std::vector<int32_t>  int_data;     // INT32
    std::vector<float>    flt_data;     // FLOAT32
    std::vector<double>   dbl_data;     // FLOAT64
    std::vector<uint8_t>  bool_data;    // BOOL
    std::vector<uint32_t> uint_data;    // UINT32
    // STRING_DICT:
    std::vector<uint32_t> codes;        // code indices
    std::vector<std::string> dict;      // unique strings (levels)
};

// ============================================================================
// Serialize an obs/var table to a byte buffer
// ============================================================================

inline std::vector<uint8_t> obs_var_table_serialize(
    uint32_t n_rows,
    const std::vector<ColumnData>& columns)
{
    uint32_t n_cols = static_cast<uint32_t>(columns.size());
    uint32_t hdr_bytes = sizeof(ObsVarTableHeader) + n_cols * sizeof(ColDescriptor);

    // First pass: compute data sizes and offsets
    std::vector<ColDescriptor> descs(n_cols);
    std::vector<std::vector<uint8_t>> blobs(n_cols);
    std::vector<std::vector<uint8_t>> dicts(n_cols);

    for (uint32_t i = 0; i < n_cols; ++i) {
        const auto& col = columns[i];
        ColDescriptor& d = descs[i];

        // Copy name (truncate to 63 chars)
        std::strncpy(d.name, col.name.c_str(), 63);
        d.name[63] = '\0';
        d.col_type = static_cast<uint8_t>(col.type);
        d.nullable = col.nullable ? 1 : 0;

        switch (col.type) {
            case ColType::INT32: {
                blobs[i].resize(n_rows * sizeof(int32_t));
                std::memcpy(blobs[i].data(), col.int_data.data(), blobs[i].size());
                break;
            }
            case ColType::FLOAT32: {
                blobs[i].resize(n_rows * sizeof(float));
                std::memcpy(blobs[i].data(), col.flt_data.data(), blobs[i].size());
                break;
            }
            case ColType::FLOAT64: {
                blobs[i].resize(n_rows * sizeof(double));
                std::memcpy(blobs[i].data(), col.dbl_data.data(), blobs[i].size());
                break;
            }
            case ColType::BOOL: {
                blobs[i].resize(n_rows * sizeof(uint8_t));
                std::memcpy(blobs[i].data(), col.bool_data.data(), blobs[i].size());
                break;
            }
            case ColType::UINT32: {
                blobs[i].resize(n_rows * sizeof(uint32_t));
                std::memcpy(blobs[i].data(), col.uint_data.data(), blobs[i].size());
                break;
            }
            case ColType::STRING_DICT: {
                // Codes array
                blobs[i].resize(n_rows * sizeof(uint32_t));
                std::memcpy(blobs[i].data(), col.codes.data(), blobs[i].size());
                // Dictionary: null-delimited strings
                for (const auto& s : col.dict) {
                    dicts[i].insert(dicts[i].end(), s.begin(), s.end());
                    dicts[i].push_back('\0');
                }
                d.dict_bytes = static_cast<uint32_t>(dicts[i].size());
                break;
            }
        }
    }

    // Compute offsets (relative to start of table blob = after header)
    uint64_t cursor = hdr_bytes;
    for (uint32_t i = 0; i < n_cols; ++i) {
        descs[i].data_offset = cursor;
        cursor += blobs[i].size();
        if (!dicts[i].empty()) {
            descs[i].dict_offset = cursor;
            cursor += dicts[i].size();
        }
    }

    // Build the buffer
    uint64_t total_bytes = cursor;
    std::vector<uint8_t> buf(total_bytes, 0);

    // Write header
    ObsVarTableHeader th;
    th.n_rows = n_rows;
    th.n_cols = n_cols;
    th.header_bytes = hdr_bytes;
    std::memcpy(buf.data(), &th, sizeof(ObsVarTableHeader));

    // Write descriptors
    std::memcpy(buf.data() + sizeof(ObsVarTableHeader),
                descs.data(), n_cols * sizeof(ColDescriptor));

    // Write data blobs and dicts
    for (uint32_t i = 0; i < n_cols; ++i) {
        std::memcpy(buf.data() + descs[i].data_offset,
                    blobs[i].data(), blobs[i].size());
        if (!dicts[i].empty()) {
            std::memcpy(buf.data() + descs[i].dict_offset,
                        dicts[i].data(), dicts[i].size());
        }
    }

    return buf;
}

// ============================================================================
// Deserialize an obs/var table from a byte buffer
// ============================================================================

inline std::vector<ColumnData> obs_var_table_deserialize(
    const uint8_t* buf,
    size_t buf_bytes)
{
    if (buf_bytes < sizeof(ObsVarTableHeader)) {
        throw std::runtime_error("obs_var_table: buffer too small for header");
    }

    ObsVarTableHeader th;
    std::memcpy(&th, buf, sizeof(ObsVarTableHeader));
    if (!th.valid()) {
        throw std::runtime_error("obs_var_table: invalid magic (expected OVTB)");
    }

    uint32_t n_rows = th.n_rows;
    uint32_t n_cols = th.n_cols;

    if (buf_bytes < th.header_bytes) {
        throw std::runtime_error("obs_var_table: buffer too small for descriptors");
    }

    // Read descriptors
    const ColDescriptor* descs = reinterpret_cast<const ColDescriptor*>(
        buf + sizeof(ObsVarTableHeader));

    std::vector<ColumnData> columns(n_cols);

    for (uint32_t i = 0; i < n_cols; ++i) {
        const ColDescriptor& d = descs[i];
        ColumnData& col = columns[i];

        col.name = std::string(d.name);
        col.type = static_cast<ColType>(d.col_type);
        col.nullable = (d.nullable != 0);

        const uint8_t* data_ptr = buf + d.data_offset;

        switch (col.type) {
            case ColType::INT32: {
                col.int_data.resize(n_rows);
                std::memcpy(col.int_data.data(), data_ptr, n_rows * sizeof(int32_t));
                break;
            }
            case ColType::FLOAT32: {
                col.flt_data.resize(n_rows);
                std::memcpy(col.flt_data.data(), data_ptr, n_rows * sizeof(float));
                break;
            }
            case ColType::FLOAT64: {
                col.dbl_data.resize(n_rows);
                std::memcpy(col.dbl_data.data(), data_ptr, n_rows * sizeof(double));
                break;
            }
            case ColType::BOOL: {
                col.bool_data.resize(n_rows);
                std::memcpy(col.bool_data.data(), data_ptr, n_rows * sizeof(uint8_t));
                break;
            }
            case ColType::UINT32: {
                col.uint_data.resize(n_rows);
                std::memcpy(col.uint_data.data(), data_ptr, n_rows * sizeof(uint32_t));
                break;
            }
            case ColType::STRING_DICT: {
                col.codes.resize(n_rows);
                std::memcpy(col.codes.data(), data_ptr, n_rows * sizeof(uint32_t));
                // Parse null-delimited dictionary
                if (d.dict_bytes > 0) {
                    const char* dp = reinterpret_cast<const char*>(buf + d.dict_offset);
                    const char* dend = dp + d.dict_bytes;
                    while (dp < dend) {
                        std::string s(dp);
                        col.dict.push_back(s);
                        dp += s.size() + 1;
                    }
                }
                break;
            }
        }
    }

    return columns;
}

}  // namespace v2
}  // namespace streampress
