# StreamPress (.spz) API Reference

## Format Overview

StreamPress (`.spz`) is a compressed binary format for sparse CSC matrices,
designed for genomics and single-cell data. It stores column-chunked,
rANS-encoded sparse matrices with optional metadata, dimnames, obs/var tables,
and pre-stored transpose sections.

- **Magic bytes**: `SPRZ`
- **Format version**: 2 (never changes for backwards compatibility)
- **Default chunk size**: 2048 columns or ~64 MB (whichever is specified)

## Backwards Compatibility

The `FileHeader_v2` contains 32 reserved bytes (offsets 96–127), which are
zero-filled on old files. New features use these bytes via typed accessor methods:

| Bytes | Field | Zero meaning |
|-------|-------|--------------|
| 0–7 | `obs_table_offset` (uint64) | No obs table |
| 8–15 | `var_table_offset` (uint64) | No var table |
| 16–19 | `transp_chunk_cols` (uint32) | Same as `chunk_cols` |
| 20 | `flags` (uint8, bit 0 = has_dimnames) | No dimnames flag |
| 21–31 | Reserved (zero) | Future use |

Old GEO corpus files (written with `include_transpose=FALSE`, no obs/var) remain
fully readable. `st_read()` and `st_slice_cols()` work on all v2 files.

---

## R API Reference

### Write Functions

#### `st_write(x, path, ...)`

Write a sparse matrix to `.spz` format.

```r
st_write(x, path,
         obs = NULL,               # data.frame cell metadata (nrow == nrow(x))
         var = NULL,               # data.frame gene metadata (nrow == ncol(x))
         delta = TRUE,             # delta prediction for gaps
         value_pred = FALSE,       # value prediction (integer data)
         verbose = FALSE,
         precision = "auto",       # "auto", "fp32", "fp16", "quant8", "fp64"
         row_sort = FALSE,         # sort rows by nnz
         include_transpose = TRUE, # store CSC(A^T) for row slicing
         chunk_cols = NULL,        # explicit column count per chunk (NULL = use chunk_bytes)
         chunk_bytes = 64e6,       # target bytes per chunk (64 MB default)
         transp_chunk_cols = NULL,  # transpose chunk sizing
         transp_chunk_bytes = NULL, # transpose chunk bytes
         threads = 0L)             # 0 = all threads
```

**Returns**: List with `raw_bytes`, `compressed_bytes`, `ratio`, `compress_ms`,
`num_chunks`, `version`, `has_transpose`, `has_obs`, `has_var`.

#### `st_write_list(x, path, ...)`

Write a list of matrices as a single column-concatenated `.spz` file.
All matrices must have the same `nrow`. Currently uses `do.call(cbind, x)`.

```r
st_write_list(x, path, obs = NULL, var = NULL,
              chunk_bytes = 64e6, chunk_cols = NULL,
              include_transpose = TRUE, precision = "auto",
              threads = 0L, verbose = FALSE)
```

### Read Functions

#### `st_read(path, cols, reorder, threads)`

Read full or partial matrix from `.spz` file.

```r
st_read(path, cols = NULL, reorder = TRUE, threads = 0L)
```

**Returns**: `dgCMatrix` with dimnames wired from file metadata.

#### `st_read_obs(path)` / `st_read_var(path)`

Read embedded observation (cell) or variable (gene) metadata tables.

```r
obs <- st_read_obs("file.spz")  # data.frame or empty data.frame
var <- st_read_var("file.spz")
```

### Slice Functions

#### `st_slice_cols(path, cols, threads)`

Read a subset of columns.

```r
sub <- st_slice_cols("file.spz", cols = 1:100, threads = 4L)
```

#### `st_slice_rows(path, rows, threads)`

Read a subset of rows using the pre-stored transpose section.
Requires `include_transpose = TRUE` at write time.

```r
sub <- st_slice_rows("file.spz", rows = 1:500, threads = 4L)
```

#### `st_slice(path, rows, cols, threads)`

Convenience function combining row and column slicing.

```r
sub <- st_slice("file.spz", rows = 1:100, cols = 1:50)
```

### Filter Functions

#### `st_filter_rows(path, ..., threads)`

Slice rows matching obs filter criteria.

```r
sub <- st_filter_rows("file.spz", cell_type == "T cell")
```

#### `st_filter_cols(path, ..., threads)`

Slice columns matching var filter criteria.

```r
sub <- st_filter_cols("file.spz", chromosome == 1)
```

#### `st_obs_indices(path, ...)`

Return integer vector of row indices matching obs filter criteria.

```r
idx <- st_obs_indices("file.spz", score > 0.5)
```

### Chunk Iteration

#### `st_chunk_ranges(path)`

Return a data.frame with `start` and `end` columns (1-indexed, inclusive),
one row per chunk.

```r
cr <- st_chunk_ranges("file.spz")
# Returns: data.frame(start = c(1, 2049, ...), end = c(2048, 4096, ...))
```

#### `st_map_chunks(path, fn, transpose, threads)`

Apply a function to every chunk sequentially.

```r
results <- st_map_chunks("file.spz", function(chunk, col_start, col_end) {
  colSums(chunk)
}, threads = 4L)
```

### Metadata

#### `st_info(path)`

Return file metadata without decompression.

```r
info <- st_info("file.spz")
# Returns list: rows, cols, nnz, density, chunk_cols, version,
#   has_transpose, has_obs, has_var, transp_chunk_cols
```

---

## C++ API Reference

Include `<streampress/streampress_api.hpp>` for the public C++ API.

### Namespaces

All public API is in `streampress::api`.

### Option Structs

```cpp
struct WriteOptions {
    bool     include_transpose = true;
    bool     use_delta         = true;
    bool     value_pred        = false;
    bool     row_sort          = false;
    uint32_t chunk_cols        = 0;       // 0 = compute from chunk_bytes
    uint64_t chunk_bytes       = 64ULL * 1024 * 1024;  // 64 MB
    uint32_t transp_chunk_cols = 0;
    int      threads           = 0;       // 0 = all
    std::string precision      = "auto";
};

struct ReadOptions {
    bool reorder = true;
    int  threads = 0;
};
```

### Result Structs

```cpp
struct WriteStats {
    size_t   raw_size, compressed_size;
    double   compress_time_ms;
    uint32_t num_chunks;
    bool     has_transpose, has_obs, has_var;
    double   ratio() const;
};

struct ReadResult {
    std::vector<uint32_t> col_ptr;
    std::vector<uint32_t> row_ind;
    std::vector<float>    values;
    uint32_t m, n;
    uint64_t nnz;
    std::vector<std::string> rownames, colnames;
};

struct FileInfo {
    uint32_t m, n;
    uint64_t nnz;
    uint32_t chunk_cols, transp_chunk_cols;
    bool     has_transpose, has_obs_table, has_var_table, has_dimnames;
    float    density;
    uint64_t file_bytes;
    uint16_t version;
};
```

### Functions

```cpp
WriteStats write_sparse(const std::string& path, const CSCMatrix& mat,
                         const std::vector<uint8_t>& obs_buf = {},
                         const std::vector<uint8_t>& var_buf = {},
                         const WriteOptions& opts = {});

ReadResult read_sparse(const std::string& path, const ReadOptions& opts = {});

ReadResult slice_cols(const std::string& path,
                       uint32_t col_start, uint32_t col_end,
                       const ReadOptions& opts = {});

FileInfo info(const std::string& path);
```

### ChunkIterator

```cpp
#include <streampress/chunk_iterator.hpp>

streampress::ChunkIterator it("file.spz", /*threads=*/4);
while (it.has_next()) {
    streampress::ChunkRef chunk = it.next();
    // chunk.col_ptr, chunk.row_ind, chunk.values
    // chunk.col_start, chunk.col_end (0-based)
    // chunk.nrows
}
it.reset();  // rewind
```

---

## Obs/Var Table Format

### ColType Enum

| Value | Type | NA Sentinel |
|-------|------|-------------|
| 0 | INT32 | `INT32_MIN` |
| 1 | FLOAT32 | `NaN` |
| 2 | FLOAT64 | `NaN` |
| 3 | BOOL | `255` |
| 4 | UINT32 | `UINT32_MAX` |
| 5 | STRING_DICT | codes `UINT32_MAX` = NA |

### ColDescriptor (112 bytes)

```
[0:63]   name[64]      — null-terminated column name
[64]     col_type      — ColType enum
[65]     nullable      — 1 if column may contain NAs
[66:67]  _pad[2]
[68:71]  dict_bytes    — STRING_DICT dictionary size
[72:79]  data_offset   — offset from table start to raw data
[80:87]  dict_offset   — offset from table start to dictionary
[88:107] _reserved[20] — zero
[108:111] _align[4]    — padding
```

### File Layout

```
[ObsVarTableHeader (16 bytes)]
  magic[4] = "OVTB"
  n_rows (uint32)
  n_cols (uint32)
  header_bytes (uint32)
[ColDescriptor × n_cols]
[Raw data blobs...]
[Dictionary blobs (STRING_DICT only)...]
```

---

## Performance Notes

- **Threading**: All I/O functions accept `threads` parameter (0 = all available)
- **Chunk sizing**: Use `chunk_bytes = 64e6` for ~64 MB chunks; the library
  computes `chunk_cols` from matrix dimensions
- **Memory budget**: For large files, use `st_map_chunks()` to process one chunk
  at a time instead of loading the full matrix
- **pread safety**: All parallel reads use `safe_pread()` which wraps both POSIX
  `pread` (Linux/macOS) and Windows `ReadFile` with `OVERLAPPED`

## Platform Support

- **Linux**: Full support, uses `pread` for thread-safe reads
- **macOS**: Full support via `pread`
- **Windows**: Supported via `ReadFile` + `OVERLAPPED` wrapper in `platform_io.hpp`
