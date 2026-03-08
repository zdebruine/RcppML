# WORKQUEUE_3 — StreamPress Architecture Revision

**Agent**: Claude Opus 4.6 — autonomous implementation mode  
**Parallel agents**: Two other agents (WORKQUEUE_1, WORKQUEUE_2) are running simultaneously.
Your scope is the **StreamPress format library**: the `.spz` I/O infrastructure,
format headers, compression, chunk sizing, the SparsePress→StreamPress rename,
streaming transpose, and memory-aware auto-dispatch in `nmf_thin.R`.

---

## Project Context

**RcppML** is an R package (`/mnt/home/debruinz/RcppML-2`) for Non-negative Matrix
Factorization (NMF). The `.spz` file format (currently called "SparsePress") is
the primary storage format for large single-cell genomics matrices that don't fit
in RAM. The C++ library is under `inst/include/sparsepress/` and the IoLoader
infrastructure is under `inst/include/FactorNet/io/`.

This workqueue implements the **StreamPress Architecture Revision** documented in
`PRODUCTION_AUDIT.md §6`. Read that section in full before starting — it contains
the complete design rationale, all phase descriptions, and the compatibility
guarantees.

Key constraints:
- **v2 sparse format (`SPRZ` magic, version=2)**: ~1 TB of GEO reprocessed
  single-cell data is stored in this format. **NEVER modify the v2 format spec**.
  All existing v2 files must remain fully readable after every change you make.
- **v3 dense format (version=3)**: No production files exist yet. Can be extended
  freely using the 48 reserved bytes in `FileHeader_v3`.
- **WORKQUEUE_2 owns `fit_chunked.hpp`**: Do not modify that file. If you need to
  update `#include` paths in `fit_chunked.hpp` due to the rename, do it by creating
  shim headers that forward to the new location — not by editing `fit_chunked.hpp`.
- **No external dependencies**: All compression code reuses existing
  `sparsepress/codec/rans.hpp` and `sparsepress/transform/value_map.hpp`.

---

## Compute Environment (CRITICAL — READ CAREFULLY)

Login node = `port` (hostname `port.clipper.gvsu.edu`). **NEVER run compute on `port`**.

Available compute nodes (SSH directly):
- **c001, c004** — CPU
- **b004** — bigmem (large RAM, useful for testing large file operations)
- **g051** — GPU (if needed)

```bash
# All compute through SSH:
ssh c001 'cd /mnt/home/debruinz/RcppML-2 && module load r/4.5.2 && R --no-save -e "..."'
```

Pre-approved: `ssh`, `squeue`, `R`, `Rscript`, `hostname`, `cat`, `head`, `tail`,
`ls`, `git`, `module`. Use VS Code tools for read-only operations.

### Build Workflow
```bash
ssh c001 'cd /mnt/home/debruinz/RcppML-2 && module load r/4.5.2 && R --no-save -e "devtools::document()"'
ssh c001 'cd /mnt/home/debruinz/RcppML-2 && bash tools/fix_rcpp_info_bug.sh'
ssh c001 'cd /mnt/home/debruinz/RcppML-2 && module load r/4.5.2 && R --no-save -e "devtools::install(quick=TRUE)"'
```

**CRITICAL**: Always run `bash tools/fix_rcpp_info_bug.sh` after `devtools::document()`.
Rcpp 1.1.0 inserts an undefined `info` symbol into `src/RcppExports.cpp`. The script
patches it. Skipping this causes package load failure.

### Running Tests
```bash
ssh c001 'cd /mnt/home/debruinz/RcppML-2 && module load r/4.5.2 && OMP_NUM_THREADS=4 R --no-save -e "devtools::test()"'
```
Baseline: `[ FAIL 0 | WARN 15 | SKIP 148 | PASS 1987 ]`. Do not regress.

---

## Phase 0 — Locate the GEO Reprocessed Corpus

**Goal**: Identify the exact paths of the GEO reprocessed `.spz` files before
implementing anything that touches them.

The old `cellcensus_500k.spz` and `cellcensus_900k.spz` benchmarking files have
been deleted. The new corpus comes from GEO reprocessed single-cell downloads.

```bash
# Find active jobs that may be writing spz output:
squeue -u debruinz -t R -o "%j %Z %N %o" | head -30

# Search for .spz files under the project directory:
ssh c001 'find /mnt/projects/debruinz_project/ -name "*.spz" 2>/dev/null | head -30'
ssh c001 'find /mnt/projects/debruinz_project/ -name "*.spz" 2>/dev/null | wc -l'
ssh c001 'du -ch $(find /mnt/projects/debruinz_project/ -name "*.spz" 2>/dev/null | head -20) 2>/dev/null | tail -3'
```

Once found:
1. Record the corpus directory path in `PRODUCTION_AUDIT.md §6.1` (edit the
   "TBD: confirm from squeue" placeholder).
2. Validate a few sample files:
   ```bash
   ssh c001 'cd /mnt/home/debruinz/RcppML-2 && module load r/4.5.2 && R --no-save -e "
     library(RcppML)
     files <- list.files(\"/mnt/projects/debruinz_project/PATH\", \"*.spz\", full.names=TRUE)
     sample_files <- sample(files, min(10, length(files)))
     for (f in sample_files) cat(f, sp_info(f)\$version, sp_info(f)\$m, sp_info(f)\$n, \"\n\")
   "'
   ```
3. Confirm: all files are version=2 (sparse). Record count and total size.

---

## Phase 1 — Fix SpzLoader: Seek-Based Streaming [CRITICAL BUG FIX]

**Files**:
- `inst/include/FactorNet/io/spz_loader.hpp`
- `inst/include/FactorNet/io/dense_spz_loader.hpp`
- New: `inst/include/FactorNet/io/file_reader.hpp`

**The bug**: Both loaders read the ENTIRE file into RAM in the constructor:
```cpp
file_data_.resize(file_size_);
fread(file_data_.data(), 1, file_size_, f);
```
A 50 GB `.spz` file requires 50 GB RAM just to open it. This is a correctness bug.

### Step 1a: Create `file_reader.hpp`

Create `inst/include/FactorNet/io/file_reader.hpp` — a platform-conditional
random-access file reader. This utility is also needed by WORKQUEUE_2's
`get_available_ram_bytes()` (which may use a separately named header). The name
`file_reader.hpp` is specific to file I/O; keep it focused:

```cpp
#pragma once
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <unistd.h>
#  include <sys/stat.h>
#endif

namespace streampress {

/// Platform-agnostic random-access file reader.
/// Supports positional reads without moving any file cursor.
/// Thread-safe for concurrent pread() calls on POSIX platforms.
class FileReader {
public:
    explicit FileReader(const std::string& path);
    ~FileReader();

    /// Read `size` bytes from `offset` into `buf`.
    /// Returns number of bytes actually read.
    size_t pread(uint64_t offset, void* buf, size_t size) const;

    uint64_t file_size() const { return file_size_; }
    bool is_open() const;
    void close();

private:
    uint64_t file_size_ = 0;
#ifdef _WIN32
    HANDLE hFile_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
#endif
};

} // namespace streampress
```

Implement it in the same header (inline), handling three platforms:
- **Linux/macOS/NFS**: `open(path, O_RDONLY)` → `::pread(fd_, buf, size, offset)`
- **Windows**: `CreateFileW(...)` → `ReadFile()` with `OVERLAPPED` and a
  `LARGE_INTEGER` offset. Use `MultiByteToWideChar` to convert the path.

Test that it compiles on the current Linux node. Windows compilation is verified
via CI later.

### Step 1b: Fix `SpzLoader<Scalar>`

In `spz_loader.hpp`, replace the `file_data_` bulk-read approach:

**Remove**:
- `std::vector<uint8_t> file_data_` member
- Any `file_data_.resize(file_size_); fread(file_data_.data(), ...)` call in the constructor

**Add**:
- `streampress::FileReader reader_` member (include the new header)
- Keep in RAM: `FileHeader_v2 header_` and `std::vector<ChunkDescriptor_v2> fwd_descs_`
  and (if transpose exists) `std::vector<ChunkDescriptor_v2> trans_descs_`
- These are tiny: `sizeof(FileHeader_v2)=128` + `num_chunks * 24 bytes`
  (for a 3M-cell matrix with chunk_cols=256, that's 11,719 chunks × 24 bytes ≈ 270 KB)

**In-core threshold** (optional optimization for small files):
```cpp
static constexpr uint64_t IN_CORE_THRESHOLD = 2ULL * 1024 * 1024 * 1024; // 2 GB
// If file_size_ < IN_CORE_THRESHOLD, fall back to legacy bulk-read for speed.
```

The `next_forward()` and `next_transpose()` methods should now:
1. Look up the chunk offset from `fwd_descs_[chunk_idx_]`
2. Call `reader_.pread(chunk.byte_offset, buf.data(), chunk.byte_size)`
3. Decompress the chunk bytes (same logic as before, now operating on a local buffer)

### Step 1c: Fix `DenseSpzLoader<Scalar>` identically

Same approach: replace `file_data_` bulk-read with `FileReader` + positional reads,
keeping only `FileHeader_v3` and `std::vector<DenseChunkDescriptor>` in RAM.

### Step 1d: Test

```bash
ssh c001 'cd /mnt/home/debruinz/RcppML-2 && module load r/4.5.2 && OMP_NUM_THREADS=4 R --no-save -e "
  devtools::install(quick=TRUE)
  library(RcppML)
  data(monocle3_subset)
  f <- tempfile(fileext=\".spz\")
  sp_write(monocle3_subset, f)
  A <- sp_read(f)
  stopifnot(all.equal(as.matrix(monocle3_subset), as.matrix(A)))
  res <- nmf(f, k=4, maxit=3)
  cat(\"SpzLoader fix OK\n\")
  unlink(f)
"'
```

After fixing, verify with the GEO corpus (from Phase 0) that `sp_info()` works
on existing v2 files.

---

## Phase 2 — Dense v3 Compression

**Files**:
- `inst/include/sparsepress/format/header_v3.hpp`
- `inst/include/sparsepress/sparsepress_v3.hpp`

**Goal**: Add real compression to the dense v3 format, reusing the existing
`rans.hpp` and `value_map.hpp` codecs. Backwards compatible: existing v3 files
with `reserved[0]=0` continue to read as `RAW_FP32` (no change).

### Step 2a: Extend `FileHeader_v3` codec field

In `header_v3.hpp`, assign the first two reserved bytes:

```cpp
// In FileHeader_v3, change:
//   uint8_t reserved[48];
// to (do NOT reduce the struct size — keep it 128 bytes total):

// reserved[0]: dense codec
enum class DenseCodec : uint8_t {
    RAW_FP32  = 0,  // original format, no change
    FP16      = 1,  // 50% size, lossless fp16 truncation
    QUANT8    = 2,  // 75% size, 8-bit range quantization
    FP16_RANS = 3,  // fp16 + XOR-column-delta + rANS entropy coding
    FP32_RANS = 4,  // fp32 + XOR-column-delta + rANS entropy coding
};
// reserved[1]: delta_encode flag: 0=none, 1=XOR-delta between adjacent column bytes

// The struct MUST remain exactly 128 bytes. If you add named fields to the
// union/anonymous struct, verify sizeof(FileHeader_v3) == 128 with a static_assert.
static_assert(sizeof(FileHeader_v3) == 128, "FileHeader_v3 must be 128 bytes");
```

### Step 2b: Update `DenseChunkDescriptor`

In `header_v3.hpp`, extend `DenseChunkDescriptor` to carry both compressed and
uncompressed sizes:

```cpp
struct DenseChunkDescriptor {
    uint64_t col_start;           // first column index in this chunk
    uint32_t num_cols;            // number of columns
    uint32_t _pad;                // alignment padding (was implicit)
    uint64_t byte_offset;         // file offset of compressed bytes
    uint64_t byte_size;           // compressed byte size (= uncompressed if codec=0)
    uint64_t uncompressed_size;   // uncompressed byte size (m * num_cols * val_bytes)
};
static_assert(sizeof(DenseChunkDescriptor) == 40, "DenseChunkDescriptor size check");
```

**Backwards compat note**: Old v3 files have the original 24-byte
`DenseChunkDescriptor`. Handle this in the reader by checking if
`header.reserved[0] == 0` (RAW_FP32) — if so, `uncompressed_size = byte_size`
(since they were equal for uncompressed data). This avoids needing a new version
number.

**IMPORTANT**: Update `sizeof(DenseChunkDescriptor)` usage throughout
`sparsepress_v3.hpp` to use the new 40-byte size. Old files must still read
correctly. Check the existing code path that parses chunk descriptors and ensure
it handles both sizes (or rely on codec=0 implying old format).

### Step 2c: Compression pipeline in `sparsepress_v3.hpp`

Add the write path (codec selectable, default = RAW_FP32 for now to preserve
existing behavior):

```cpp
// Write path: fp32 panel → encode → write compressed bytes
std::vector<uint8_t> encode_dense_chunk(
    const Scalar* col_data,   // m * num_cols float values, column-major
    uint32_t m, uint32_t num_cols,
    DenseCodec codec, bool delta_encode);
```

Pipeline for `FP16_RANS`:
1. `fp32_to_fp16`: reuse `sparsepress::transform::value_map::to_fp16()` (col-major)
2. `xor_delta`: for each column, XOR adjacent fp16 bit patterns (reduces residual
   entropy for smooth data like normalized scRNA-seq)
3. `rans_encode`: reuse `sparsepress::codec::rans_encode()` on the fp16 byte stream
4. Write compressed bytes; store `uncompressed_size` and compressed `byte_size`

Read path is the reverse:
```cpp
void decode_dense_chunk(
    const uint8_t* compressed, uint64_t compressed_size,
    Scalar* out, uint32_t m, uint32_t num_cols,
    DenseCodec codec, bool delta_encode,
    uint64_t uncompressed_size);
```

**Default codec**: Keep the default as `RAW_FP32` initially. Users opt in to
compression via `sp_write_dense(x, path, codec="fp16_rans")`. This ensures no
regression on existing v3 round-trip tests.

---

## Phase 3 — User-Configurable Chunk Size (Default 2048)

**Files**:
- `inst/include/sparsepress/sparsepress_v2.hpp` (or wherever `DEFAULT_CHUNK_COLS` is defined; search for it)
- `R/sparsepress.R`

**Key change**: `DEFAULT_CHUNK_COLS` from **256 → 2048**.

```bash
# Find where DEFAULT_CHUNK_COLS is defined:
```
Use `grep_search` tool to find it.

### Step 3a: Change the constant

After finding the definition location, change from:
```cpp
static constexpr uint32_t DEFAULT_CHUNK_COLS = 256;
```
to:
```cpp
static constexpr uint32_t DEFAULT_CHUNK_COLS = 2048;
```

### Step 3b: Expose `chunk_cols` in R write API

In `R/sparsepress.R`, find `sp_write()`. Add a `chunk_cols` parameter:
```r
sp_write <- function(x, path, delta = TRUE, value_pred = FALSE, verbose = FALSE,
                     include_transpose = FALSE,
                     chunk_cols = 2048L,   # NEW — was hardcoded
                     ...) {
```

Pass `chunk_cols` through to the underlying `Rcpp_sp_write()` call. If `Rcpp_sp_write`
doesn't accept a `chunk_cols` argument yet, you need to add it to the C++ wrapper
in `src/RcppFunctions.cpp` (or wherever `Rcpp_sp_write` is defined) and to the
sparsepress write function signature. This requires a `devtools::document()` rebuild.

### Step 3c: Add `choose_chunk_cols()` helper (optional, used by auto-dispatch)

Create `inst/include/FactorNet/io/chunk_size.hpp`:
```cpp
#pragma once
#include <cstdint>
#include <algorithm>

namespace streampress {

/// Heuristic chunk column count for streaming NMF.
/// Targets ~1% of available RAM per chunk.
inline uint32_t choose_chunk_cols(uint64_t m, uint64_t ram_avail_bytes) {
    if (ram_avail_bytes == 0) return 2048;
    uint64_t bytes_per_col = m * sizeof(float);
    uint64_t target_bytes = ram_avail_bytes / 100;
    uint64_t auto_cols = target_bytes / bytes_per_col;
    return (uint32_t)std::clamp(auto_cols, (uint64_t)256, (uint64_t)32768);
}

} // namespace streampress
```

---

## Phase 4 — Rename SparsePress → StreamPress (Shim-Based)

**Strategy**: Use shim headers so that ALL existing `#include <sparsepress/...>`
paths continue to compile without modification. This is critical for parallel
execution — WORKQUEUE_2 is editing `fit_chunked.hpp` which has `#include
<sparsepress/...>` includes; you must NOT break those while they are being edited.

### Step 4a: Create `inst/include/streampress/` directory

Copy/move all content from `inst/include/sparsepress/` to `inst/include/streampress/`.
Do a global find-and-replace of `namespace sparsepress` → `namespace streampress`
and `sparsepress::` → `streampress::` within the new streampress/ files ONLY.

```bash
ssh c001 'cp -r /mnt/home/debruinz/RcppML-2/inst/include/sparsepress /mnt/home/debruinz/RcppML-2/inst/include/streampress'
ssh c001 'cd /mnt/home/debruinz/RcppML-2 && find inst/include/streampress -name "*.hpp" -exec sed -i "s/namespace sparsepress/namespace streampress/g; s/sparsepress::/streampress::/g" {} \;'
```

Also update internal `#include <sparsepress/...>` within the new `streampress/`
files to `#include <streampress/...>`:
```bash
ssh c001 'cd /mnt/home/debruinz/RcppML-2 && find inst/include/streampress -name "*.hpp" -exec sed -i "s|#include <sparsepress/|#include <streampress/|g" {} \;'
```

### Step 4b: Convert `inst/include/sparsepress/` to shim headers

Replace each `sparsepress/*.hpp` file with a one-line shim that includes the
new streampress equivalent AND imports the namespace:

```bash
# For each .hpp in inst/include/sparsepress/:
for FILE in .../sparsepress/*.hpp; do
    BASENAME=$(basename $FILE)
    echo "#pragma once
// Compatibility shim — SparsePress is now StreamPress.
// This header will be removed in a future version.
#include <streampress/${BASENAME}>
namespace sparsepress = streampress;" > $FILE
done
```

Do the same for subdirectories: `codec/`, `format/`, `model/`, `transform/`.
Use a recursive find command.

**IMPORTANT**: The v2/v3 on-disk magic bytes (`SPRZ`, `SPEN`) in format headers
are NOT renamed — these are file-format constants, not API names.

### Step 4c: New R file `R/streampress.R`

Create `R/streampress.R` with the new `st_*` API. These are NOT mere wrappers
but the primary functions going forward. Move the documentation and implementation
from `R/sparsepress.R` to `R/streampress.R`, renamed:

| Old | New |
|-----|-----|
| `sp_write()` | `st_write()` |
| `sp_read()` | `st_read()` |
| `sp_info()` | `st_info()` |
| `sp_write_dense()` | `st_write_dense()` |
| `sp_read_dense()` | `st_read_dense()` |

Then in `R/sparsepress.R`, replace all function bodies with deprecation wrappers:
```r
#' @rdname streampress-deprecated
#' @export
sp_write <- function(...) {
  .Deprecated("st_write", package = "RcppML",
              msg = "sp_write() is deprecated. Use st_write() instead.")
  st_write(...)
}
```

Keep `R/sparsepress.R` but make it thin. Export both `sp_*` and `st_*`.

### Step 4d: Update `DESCRIPTION`

Add `streampress` to any `LinkingTo` or package name references if applicable.
The package is still called "RcppML" — this is an internal rename of a subsystem.

### Step 4e: Verify no regressions

```bash
ssh c001 'cd /mnt/home/debruinz/RcppML-2 && module load r/4.5.2 && OMP_NUM_THREADS=4 R --no-save -e "
  devtools::install(quick=TRUE)
  library(RcppML)
  # Old API still works:
  data(monocle3_subset)
  f <- tempfile(fileext=\".spz\")
  sp_write(monocle3_subset, f)      # should warn about deprecation
  A <- sp_read(f)
  stopifnot(all.equal(as.matrix(monocle3_subset), as.matrix(A)))
  # New API works:
  st_write(monocle3_subset, f)
  B <- st_read(f)
  stopifnot(all.equal(as.matrix(monocle3_subset), as.matrix(B)))
  cat(\"Rename OK\n\")
  unlink(f)
"'
```

---

## Phase 5 — Streaming Distributed Transpose

**New file**: `inst/include/streampress/transpose.hpp`

The transpose section needed for W-update in NMF ALS is written to the same `.spz`
file as the forward section. This phase adds two capabilities:

### Mode A: During-write transpose (`include_transpose=TRUE`)

Modify `st_write()` so that when `include_transpose=TRUE`, it accumulates a
sorted-row bucketing structure as it streams forward chunks and writes the
transpose section when the file is closed.

Algorithm (memory-efficient bucket sort):
```
For each forward chunk [j0, j1):
  For each nonzero (i, j, v) in the chunk:
    append (j, v) to bucket[i]  // bucket indexed by row
After all chunks:
  For each row i:
    write sorted bucket[i] as a "transpose chunk" (one chunk per panel of rows)
  Update FileHeader with transpose_offset
```

Working memory: O(nnz_per_chunk + bucket list) ≈ O(2 × chunk_nnz). For very
large matrices (>100M nonzeros), the buckets may exceed RAM — handle this by
flushing to a temp file and merging (external sort). Use `tmpdir` parameter for
this.

### Mode B: Post-hoc transpose (`st_add_transpose()`)

New R function:
```r
#' @export
st_add_transpose <- function(path, tmpdir = tempdir(), verbose = TRUE) {
  ...
}
```

Backed by a C++ function `Rcpp_st_add_transpose(path, tmpdir, verbose)` that:
1. Opens the `.spz` file, reads forward chunks
2. Uses the same bucket-sort algorithm as Mode A
3. Appends the transpose section, patches FileHeader

Export via `// [[Rcpp::export]]`.

---

## Phase 6 — Memory-Aware Auto-Dispatch in `nmf_thin.R`

**File**: `R/nmf_thin.R`  
**Goal**: When the user passes a path to `nmf()` or `cv_nmf()`, automatically
determine the optimal execution mode without user configuration.

### Step 6a: R-level helper function

Add to `R/streampress.R` (or new `R/dispatch.R`):

```r
.st_dispatch <- function(path, k, resource = "auto") {
  info <- st_info(path)
  file_size <- file.info(path)$size
  
  # Estimate decompressed size: depends on codec in file header
  # For v2 sparse: compressed_file_size * ~8 (typical 8x compression on scRNA)
  # For v3 dense: check codec field; raw = 1x, fp16 = ~2x
  decomp_factor <- if (info$version == 2L) 8.0 else 1.0
  est_decomp_bytes <- file_size * decomp_factor
  
  # Available RAM
  avail_ram <- .get_available_ram_bytes()  # calls C++ get_available_ram_bytes()
  
  # Available VRAM (0 if no GPU or GPU not requested)
  avail_vram <- if (resource %in% c("gpu", "auto") && isTRUE(.gpu_available())) {
    .get_available_vram_bytes()
  } else 0L
  
  safety <- 0.70
  
  if (avail_vram > 0 && est_decomp_bytes < avail_vram * safety) {
    list(mode = "IN_CORE_GPU", resource = "gpu", streaming = FALSE)
  } else if (avail_vram > 0 && est_decomp_bytes < avail_ram * safety) {
    list(mode = "CPU_TO_GPU", resource = "gpu", streaming = FALSE)  
  } else if (avail_vram > 0) {
    list(mode = "STREAMING_GPU", resource = "gpu", streaming = TRUE)
  } else if (est_decomp_bytes < avail_ram * safety) {
    list(mode = "IN_CORE_CPU", resource = "cpu", streaming = FALSE)
  } else {
    list(mode = "STREAMING_CPU", resource = "cpu", streaming = TRUE)
  }
}
```

### Step 6b: Wire into `nmf_thin.R`

Find the section in `nmf_thin.R` where `.spz` file paths are handled (search for
`grepl(".spz"` or the `is.character(data)` branch). Before the C++ dispatch call,
add:

```r
if (is.character(data) && grepl("\\.spz$", data, ignore.case = TRUE)) {
  if (is.null(dispatch) || dispatch == "auto") {
    mode_info <- .st_dispatch(data, k, resource = resource)
    if (verbose) message("StreamPress auto-dispatch: ", mode_info$mode)
    resource <- mode_info$resource
    streaming <- mode_info$streaming
  } else {
    # Manual override — fire a non-suppressable warning
    message(
      "WARNING: `dispatch` is set manually to '", dispatch, "'.\n",
      "  Auto-dispatch ensures sufficient RAM is available before loading.\n",
      "  Manual dispatch may cause out-of-memory errors or crashes.\n",
      "  Remove `dispatch=` to restore safe automatic mode."
    )
  }
}
```

Add `dispatch = NULL` as a parameter to `nmf()` in `R/nmf_thin.R`.

### Step 6c: Platform-conditional RAM/VRAM detection

The `get_available_ram_bytes()` C++ function from WORKQUEUE_2 (Task 3, placed in
`inst/include/FactorNet/core/platform.hpp`) is needed here. If WORKQUEUE_2 has
already created it, include and reuse it:

```cpp
#include <FactorNet/core/platform.hpp>  // get_available_ram_bytes()
```

If WORKQUEUE_2 has not yet created it, create it yourself in the same location.
This function must be Windows/macOS/Linux conditional (see WORKQUEUE_2 Task 3
for the full implementation with all three platform branches).

Add a VRAM query function alongside it:
```cpp
inline uint64_t get_available_vram_bytes() {
#ifdef RCPPML_CUDA_ENABLED
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        return (uint64_t)free_bytes;
    }
#endif
    return 0ULL;
}
```

Expose both at R level via:
```cpp
// [[Rcpp::export]]
double Rcpp_get_available_ram_mb() {
    return (double)get_available_ram_bytes() / (1024.0 * 1024.0);
}
```

---

## Phase 7 — GEO Corpus Compatibility + k=64 Benchmark Setup

### Step 7a: Corpus compatibility test script

Create `tools/verify_spz_corpus.R`:

```r
#!/usr/bin/env Rscript
# Verifies that all .spz files in the GEO reprocessed corpus are readable
# after the StreamPress rename.
#
# Usage: Rscript tools/verify_spz_corpus.R [corpus_dir] [n_sample]

library(RcppML)

args <- commandArgs(trailingOnly = TRUE)
corpus_dir <- if (length(args) >= 1) args[1] else {
  stop("Usage: Rscript tools/verify_spz_corpus.R <corpus_dir> [n_sample]")
}
n_sample <- if (length(args) >= 2) as.integer(args[2]) else 100L

files <- list.files(corpus_dir, pattern = "\\.spz$", full.names = TRUE, recursive = TRUE)
cat(sprintf("Found %d .spz files in %s\n", length(files), corpus_dir))

sample_files <- sample(files, min(n_sample, length(files)))
n_ok <- 0L
n_fail <- 0L
for (f in sample_files) {
  tryCatch({
    info <- st_info(f)
    stopifnot(info$version == 2L, info$m > 0, info$n > 0, info$nnz > 0)
    n_ok <- n_ok + 1L
  }, error = function(e) {
    cat(sprintf("FAIL: %s — %s\n", f, conditionMessage(e)))
    n_fail <<- n_fail + 1L
  })
}
cat(sprintf("Passed: %d / %d sampled files\n", n_ok, n_ok + n_fail))
if (n_fail > 0) stop(sprintf("%d files failed compatibility check!", n_fail))
cat("All sampled files are compatible.\n")
```

Run it against the corpus found in Phase 0:
```bash
ssh c001 'cd /mnt/home/debruinz/RcppML-2 && module load r/4.5.2 && R --no-save -e "
  devtools::install(quick=TRUE)
" && Rscript tools/verify_spz_corpus.R /mnt/projects/debruinz_project/PATH_FROM_PHASE0 100'
```

### Step 7b: Unit tests

Create `tests/testthat/test_streampress_compat.R`:

```r
library(testthat)
library(RcppML)
library(Matrix)

test_that("st_write/st_read round-trip: sparse matrix", {
  set.seed(42)
  A <- Matrix::rsparsematrix(200, 100, density = 0.05, repr = "C")
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))
  st_write(A, f)
  B <- st_read(f)
  expect_equal(as.matrix(A), as.matrix(B), tolerance = 1e-6)
})

test_that("deprecated sp_write still works (but warns)", {
  A <- Matrix::rsparsematrix(50, 50, density = 0.1, repr = "C")
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))
  expect_warning(sp_write(A, f), "deprecated")
  B <- sp_read(f)
  expect_equal(as.matrix(A), as.matrix(B), tolerance = 1e-6)
})

test_that("st_info returns correct dimensions for v2 file", {
  A <- Matrix::rsparsematrix(300, 150, density = 0.03, repr = "C")
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))
  st_write(A, f)
  info <- st_info(f)
  expect_equal(info$m, 300L)
  expect_equal(info$n, 150L)
  expect_equal(info$version, 2L)
})

test_that("nmf from .spz file works end-to-end", {
  A <- Matrix::rsparsematrix(500, 200, density = 0.05, repr = "C")
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))
  st_write(A, f, include_transpose = TRUE)
  res <- nmf(f, k = 4, maxit = 5, tol = 1e-3)
  expect_s4_class(res, "nmf")
  expect_equal(nrow(res@w), 500L)
  expect_equal(ncol(res@h), 200L)
})

test_that("chunk_cols parameter accepted by st_write", {
  A <- Matrix::rsparsematrix(100, 200, density = 0.1, repr = "C")
  f <- tempfile(fileext = ".spz")
  on.exit(unlink(f))
  expect_no_error(st_write(A, f, chunk_cols = 64L))
  info <- st_info(f)
  expect_equal(info$m, 100L)
})
```

### Step 7c: k=64 benchmark script

Create `benchmarks/harness/streaming_spz/bench_geo_k64.R`:

```r
#!/usr/bin/env Rscript
# Flagship benchmark: GEO single-cell dataset, k=64, NB loss, no manual configuration.
# Run on a compute node with: Rscript benchmarks/harness/streaming_spz/bench_geo_k64.R <path.spz>
#
# Records: wall time, per-iteration breakdown, peak RAM, dispatch mode.

library(RcppML)
library(bench)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Usage: Rscript bench_geo_k64.R <path.spz>")
spz_path <- args[1]

stopifnot(file.exists(spz_path))

cat("System info:\n")
cat("  Node:", Sys.info()["nodename"], "\n")
cat("  R version:", R.version$version.string, "\n")
cat("  OMP_NUM_THREADS:", Sys.getenv("OMP_NUM_THREADS", "not set"), "\n")
cat("  GPU available:", RcppML::gpu_available(), "\n")
cat("  Available RAM (MB):", round(RcppML:::.get_available_ram_mb(), 1), "\n\n")

info <- st_info(spz_path)
cat(sprintf("Dataset: %d genes x %d cells, nnz=%d (%.1f%% density)\n",
            info$m, info$n, info$nnz,
            100 * info$nnz / (info$m * info$n)))
cat("File size:", round(file.info(spz_path)$size / 1e9, 2), "GB\n\n")

cat("Running nmf(k=64, loss='nb', maxit=20) ...\n")
t0 <- proc.time()
res <- nmf(
  spz_path,
  k     = 64,
  loss  = "nb",
  maxit = 20,
  tol   = 1e-4,
  verbose = TRUE
)
elapsed <- proc.time() - t0

cat(sprintf("\nTotal wall time: %.1f seconds (%.1f min)\n",
            elapsed["elapsed"], elapsed["elapsed"] / 60))
cat(sprintf("Final loss: %.4f\n", tail(res@loss_history, 1)))
cat(sprintf("Dispatch mode: %s\n", attr(res, "dispatch_mode") %||% "unknown"))

# Save results
results <- list(
  spz_path    = spz_path,
  info        = info,
  k           = 64L,
  loss        = "nb",
  maxit       = 20L,
  elapsed_sec = as.numeric(elapsed["elapsed"]),
  loss_history = res@loss_history,
  sys_info    = as.list(Sys.info()),
  timestamp   = Sys.time()
)
out_rds <- file.path("benchmarks/results",
                     paste0("geo_k64_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".rds"))
dir.create("benchmarks/results", showWarnings = FALSE)
saveRDS(results, out_rds)
cat("Results saved to:", out_rds, "\n")
```

---

## Final Validation

After all phases complete:

1. Full test suite:
   ```bash
   ssh c001 'cd /mnt/home/debruinz/RcppML-2 && module load r/4.5.2 && OMP_NUM_THREADS=4 R --no-save -e "devtools::test()" 2>&1 | tail -5'
   ```
   Target: `[ FAIL 0 ]`

2. StreamPress-specific tests:
   ```bash
   ssh c001 'cd /mnt/home/debruinz/RcppML-2 && module load r/4.5.2 && OMP_NUM_THREADS=4 R --no-save -e "devtools::test(filter=\"streampress\")"'
   ```

3. If GEO corpus was found in Phase 0, run corpus compatibility:
   ```bash
   ssh c001 'cd /mnt/home/debruinz/RcppML-2 && module load r/4.5.2 && Rscript tools/verify_spz_corpus.R <corpus_dir> 100'
   ```

4. Update `PRODUCTION_AUDIT.md §6` to mark each phase ✅.

5. Commit all changes:
   ```bash
   ssh c001 'cd /mnt/home/debruinz/RcppML-2 && git add -A && git commit -m "feat(wq3): StreamPress - seek loader, v3 compression, chunk_cols=2048, rename, transpose, auto-dispatch"'
   ```

---

## Completion Criteria

1. `devtools::test()` → `[ FAIL 0 ]` (no regressions)
2. **Phase 0**: GEO corpus directory path recorded in `PRODUCTION_AUDIT.md §6.1`
   (or noted as "not yet available" with search command documented)
3. **Phase 1**: `SpzLoader` and `DenseSpzLoader` no longer read entire file into RAM;
   `inst/include/FactorNet/io/file_reader.hpp` exists with platform-conditional pread
4. **Phase 2**: Dense v3 compression pipeline exists; `DenseCodec::FP16_RANS` works;
   old raw v3 files still read correctly (codec=0 path)
5. **Phase 3**: `DEFAULT_CHUNK_COLS = 2048`; `st_write(x, path, chunk_cols=N)` works
6. **Phase 4**: `inst/include/streampress/` exists with full renamed API;
   `inst/include/sparsepress/` contains shim headers; `R/streampress.R` with `st_*`;
   deprecated `sp_*` wrappers in `R/sparsepress.R` fire `.Deprecated()` warnings
7. **Phase 5**: `st_write(..., include_transpose=TRUE)` and `st_add_transpose(path)`
   both work correctly; streaming NMF W-update uses the transpose section
8. **Phase 6**: `nmf("file.spz", k=64)` auto-selects the correct dispatch mode;
   manual `dispatch=` override fires a non-suppressable message warning
9. **Phase 7**: `tests/testthat/test_streampress_compat.R` passes;
   `tools/verify_spz_corpus.R` exists; `benchmarks/harness/streaming_spz/bench_geo_k64.R` exists
10. `PRODUCTION_AUDIT.md §6` phases all marked ✅

**Do not modify** `inst/include/FactorNet/nmf/fit_chunked.hpp` — that is WORKQUEUE_2.
Do not modify any roxygen docs in `R/*.R` beyond adding new functions — that is
WORKQUEUE_1 territory (though adding new `st_*` functions with docs is expected).
