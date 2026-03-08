#!/usr/bin/env python3
"""Append the remaining bridge functions to sparsepress_bridge.cpp."""
import sys

# Find where we need to cut and append
filepath = "/mnt/home/debruinz/RcppML-2/src/sparsepress_bridge.cpp"

with open(filepath, 'r') as f:
    lines = f.readlines()

# Find the truncated line "// @title Serialize..."
cut_idx = None
for i in range(len(lines)-1, -1, -1):
    if '// @title Serialize a data.frame' in lines[i]:
        cut_idx = i
        break

if cut_idx is None:
    print("Could not find truncation point")
    sys.exit(1)

# Remove from cut point onward
lines = lines[:cut_idx]

# Append the remaining code
remaining = r"""
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
"""

with open(filepath, 'w') as f:
    f.writelines(lines)
    f.write(remaining)

print(f"Done. Wrote {len(lines)} original lines + remaining functions")
