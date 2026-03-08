#!/usr/bin/env python3
"""Fix bridge functions using line-based approach for robustness."""

filepath = "/mnt/home/debruinz/RcppML-2/src/sparsepress_bridge.cpp"

with open(filepath, 'r') as f:
    lines = f.readlines()

# Find line numbers of key markers
serialize_start = None
deserialize_start = None
read_table_start = None

for i, line in enumerate(lines):
    stripped = line.strip()
    if 'static std::vector<uint8_t> serialize_dataframe' in stripped:
        serialize_start = i
    if 'static DataFrame deserialize_to_dataframe' in stripped:
        deserialize_start = i
    if 'static DataFrame read_table_at_offset' in stripped:
        read_table_start = i

print(f"serialize_dataframe starts at line {serialize_start}")
print(f"deserialize_to_dataframe starts at line {deserialize_start}")
print(f"read_table_at_offset starts at line {read_table_start}")

# Find the comment line before read_table_at_offset
comment_start = read_table_start
for i in range(read_table_start - 1, deserialize_start, -1):
    if lines[i].strip().startswith('//'):
        comment_start = i
    elif lines[i].strip() == '':
        continue
    else:
        break

new_serialize = r'''static std::vector<uint8_t> serialize_dataframe(const DataFrame& df) {
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
'''

# Build new file
new_lines = lines[:serialize_start]
for l in new_serialize.split('\n'):
    new_lines.append(l + '\n')
new_lines.extend(lines[comment_start:])

# Fix magic char literals
import re
content = ''.join(new_lines)
content = re.sub(r"tbl_hdr\.magic\[0\] != O ", "tbl_hdr.magic[0] != 'O' ", content)
content = re.sub(r"tbl_hdr\.magic\[1\] != V ", "tbl_hdr.magic[1] != 'V' ", content)
content = re.sub(r"tbl_hdr\.magic\[2\] != T ", "tbl_hdr.magic[2] != 'T' ", content)
content = re.sub(r"tbl_hdr\.magic\[3\] != B\)", "tbl_hdr.magic[3] != 'B')", content)

with open(filepath, 'w') as f:
    f.write(content)

total = content.count('\n')
print(f"Done. File now has {total} lines.")
