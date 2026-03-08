# WORKQUEUE_2 — Streaming Algorithm Bug Fixes & Performance

**Agent**: Claude Opus 4.6 — autonomous implementation mode  
**Parallel agents**: Two other agents (WORKQUEUE_1, WORKQUEUE_3) are running simultaneously.
Your scope is **C++ algorithm bug fixes in the streaming NMF engine and the R-level
solver dispatch**. You do NOT touch R documentation, vignettes, sparsepress format
headers, or the IoLoader infrastructure (that is WORKQUEUE_3).

---

## Project Context

**RcppML** is an R package (`/mnt/home/debruinz/RcppML-2`) for Non-negative Matrix
Factorization (NMF). The streaming NMF engine consists of:

- `inst/include/FactorNet/nmf/fit_chunked.hpp` — chunked ALS loop (CPU). This is
  the most important file in your scope. It processes data chunk-by-chunk from a
  `DataLoader<Scalar>&` interface.
- `inst/include/FactorNet/nmf/fit_streaming_spz.hpp` — entry point for `.spz` file
  streaming; sets up the `SpzLoader` and calls `fit_chunked`.
- `inst/include/FactorNet/nmf/fit_chunked_gpu.cuh` — GPU counterpart (read for
  context; some fixes may mirror here, but focus on CPU first).
- `inst/include/FactorNet/nmf/nnls_streaming.hpp` — the NNLS solver dispatch
  (Cholesky vs CD) for streaming context.
- `R/nmf_thin.R` — R dispatch; `solver_mode` encoding is at line ~605.

The authoritative work-tracking document is `PRODUCTION_AUDIT.md` in the repo root.
Read it **in full** before starting — especially §2.2 (Significant bugs), §2.3.2
(best practice comparison table in §3), and §2.4 (performance gaps).

Key design facts:
- `solver_mode = 0` → Coordinate Descent (CD) NNLS  
- `solver_mode = 1` → Cholesky+clip NNLS  
- Cholesky is faster for small k (< ~32). CD is better for k ≥ 32 or when L1 > 0  
  (Cholesky cannot enforce sparsity via L1).
- `use_irls = true` when `loss != "mse"` (GP, NB, Gamma, InvGauss, Tweedie)
- The forward pass = H-update (iterate over column chunks)  
- The transpose pass = W-update (iterate over row/transpose chunks)
- `nb_size_vec` = NB dispersion parameter vector (size m = num rows/genes)
- `LazySpeckledMask` = holdout mask using a hash function of `(row, col, seed)`

---

## Compute Environment (CRITICAL — READ CAREFULLY)

You are on a SLURM HPC cluster. The login node is `port` (hostname
`port.clipper.gvsu.edu`). **NEVER run R, Rscript, or any compute on `port`**.

Available compute nodes (SSH directly):
- **c001, c004** — CPU, use for R builds and tests
- **g051** — GPU (if you need to check GPU code compiles)

```bash
# All R/C++ work goes through SSH:
ssh c001 'cd /mnt/home/debruinz/RcppML-2 && module load r/4.5.2 && R --no-save -e "..."'
```

Pre-approved: `ssh`, `squeue`, `R`, `Rscript`, `hostname`, `cat`, `head`, `tail`,
`ls`, `git`, `module`. Use VS Code tools (`read_file`, `grep_search`, etc.) for
read-only operations.

### Build Workflow

```bash
# After any C++ change (devtools::document regenerates RcppExports.cpp):
ssh c001 'cd /mnt/home/debruinz/RcppML-2 && module load r/4.5.2 && R --no-save -e "devtools::document()"'
ssh c001 'cd /mnt/home/debruinz/RcppML-2 && bash tools/fix_rcpp_info_bug.sh'
ssh c001 'cd /mnt/home/debruinz/RcppML-2 && module load r/4.5.2 && R --no-save -e "devtools::install(quick=TRUE)"'
```

**CRITICAL**: After every `devtools::document()`, run `bash tools/fix_rcpp_info_bug.sh`
to patch the Rcpp 1.1.0 `info` symbol bug in `src/RcppExports.cpp`.

For header-only changes (no `// [[Rcpp::export]]` changes), just install directly:
```bash
ssh c001 'cd /mnt/home/debruinz/RcppML-2 && module load r/4.5.2 && R --no-save -e "devtools::install(quick=TRUE)"'
```

### Running Tests
```bash
ssh c001 'cd /mnt/home/debruinz/RcppML-2 && module load r/4.5.2 && OMP_NUM_THREADS=4 R --no-save -e "devtools::test()"'
```
Baseline: `[ FAIL 0 | WARN 15 | SKIP 148 | PASS 1987 ]`. Do not regress.

For targeted streaming tests only:
```bash
ssh c001 'cd /mnt/home/debruinz/RcppML-2 && module load r/4.5.2 && OMP_NUM_THREADS=4 R --no-save -e "
  devtools::test(filter=\"streaming\")
"'
```

---

## File Ownership — Do NOT Touch

| File | Owner |
|------|-------|
| `inst/include/FactorNet/io/spz_loader.hpp` | WORKQUEUE_3 |
| `inst/include/FactorNet/io/dense_spz_loader.hpp` | WORKQUEUE_3 |
| `inst/include/sparsepress/` (all) | WORKQUEUE_3 |
| `R/sparsepress.R` | WORKQUEUE_3 |
| `R/nmf_thin.R` lines for `.spz` dispatch / auto-dispatch | WORKQUEUE_3 |
| All files in `man/`, `NAMESPACE` | AUTO-GENERATED — never touch |

You MAY edit `R/nmf_thin.R` specifically for `solver_mode` dispatch (Task 7).
Coordinate so edits are at clearly different line numbers from WORKQUEUE_3 edits.

---

## Task 1 — NB Dispersion: Warning + Documentation (§2.2.1)

**File**: `inst/include/FactorNet/nmf/fit_chunked.hpp`  
**Context**: `nb_size_vec` is initialized from `config.nb_size_init` and never
updated during streaming NMF iteration. The in-memory path (`fit_unified.hpp`)
re-estimates dispersion each iteration. Streaming NB silently uses fixed dispersion.

**Minimum viable fix** (R-level warning):

In `R/nmf_thin.R`, find where `streaming` is set to `TRUE` and `loss="nb"`.
Add a warning there:
```r
if (isTRUE(streaming_active) && loss == "nb" && is.null(nb_size)) {
  warning(
    "Streaming NB NMF uses fixed dispersion throughout all iterations.\n",
    "  For correct results, supply a pre-estimated `nb_size` parameter.\n",
    "  Default nb_size=1 may give poor fits if your data has different overdispersion.",
    call. = FALSE
  )
}
```

Find the exact line in `nmf_thin.R` where streaming is activated. The warning
must fire before the C++ call.

**Better fix** (C++ streaming dispersion re-estimation, implement if feasible):

In `fit_chunked.hpp`, after the W-update transpose pass completes (end of
each full iteration), add a third mini-pass to re-estimate `nb_size_vec`
from residuals. The estimation uses method-of-moments: for each row i,
`r_i = mean_i² / (var_i - mean_i)` where mean and var are accumulated during
the existing forward pass. This requires adding mean and variance accumulators
to the forward pass and computing `nb_size_vec[i]` after each full iteration.

If the C++ fix is too risky without extensive testing, the R warning alone satisfies
this task. Mark which path you took in the PRODUCTION_AUDIT.md §2.2.1 status update.

---

## Task 2 — mask_zeros=FALSE CV: O(m·n) Scan (§2.2.2)

**File**: `inst/include/FactorNet/nmf/fit_chunked.hpp`  
**Context**: When `mask_zeros=FALSE`, finding holdout zero entries requires
iterating all `m` rows per column — O(m·n) total, defeating sparse streaming.

**Locate the loop**: Search `fit_chunked.hpp` for the loop that iterates
`for (uint32_t i = 0; i < m; ++i)` in the CV path and checks
`cv_mask->is_holdout(i, gj)` for zero entries when `nz_rows[ni] != i`.

**Fix strategy**: Add a method to `LazySpeckledMask` (in
`inst/include/FactorNet/nmf/speckled_cv.hpp` or wherever it is defined) that
generates holdout rows for a given column directly from the hash, without
iterating all m rows:

```cpp
// New method on LazySpeckledMask (or a free function):
// Returns a vector of holdout row indices for this column WITHOUT scanning all rows.
// Uses the same hash: hash(row, col, seed) % inv_density == 0
// Strategy: instead of testing all i in [0,m), use the observation that for
// a holdout fraction f = 1/inv_density, the expected count is f*m.
// Generate candidates by seeded random sampling rather than full scan.
std::vector<int> holdout_zeros_in_col(int col, int m,
                                       const std::vector<int>& nz_rows) const;
```

The implementation iterates only the expected ~f·m candidates using the
inverse of the hash function (or a seeded PRNG that replicates what
`is_holdout()` would return). This changes O(m) per column to O(f·m).

If the hash-inverse approach is too complex, a pragmatic middle ground:
pre-generate a sorted list of `holdout_zero_rows[col]` during `LazySpeckledMask`
construction for each chunk range, stored in a `std::unordered_map<int, std::vector<int>>`.
This trades memory for scan time.

Either way, the final test: for a synthetic matrix with m=50000, `mask_zeros=FALSE`,
confirm that CV still produces reasonable test error (correctness), and measure
whether the wall time is now sublinear in m*n.

---

## Task 3 — SVD Init Warning + Fallback (§2.2.3)

**File**: `inst/include/FactorNet/nmf/fit_streaming_spz.hpp`  
**Context**: When `init_mode=1` (Lanczos) or `init_mode=2` (IRLBA), the streaming
entry point decompresses the ENTIRE matrix into memory before SVD init. This OOMs
on any matrix that requires streaming precisely because it's too large.

**Find the code**: Search `fit_streaming_spz.hpp` for `A_full` or
`"Building temporary matrix for Lanczos"` or wherever the full decompression
occurs for SVD init.

**Fix**: Add a RAM availability check before attempting full decompression. If
the decompressed matrix won't fit in available RAM, warn and fall back to random
init:

```cpp
// Estimate decompressed size in bytes:
uint64_t est_bytes = (uint64_t)m * (uint64_t)n * sizeof(Scalar);

// Query available RAM:
uint64_t avail_bytes = get_available_ram_bytes();  // see below

if (init_mode != 0 && est_bytes > avail_bytes * 0.70) {
    // Warn and fall back
    Rcpp::warning(
        "SVD initialization requires decompressing the full matrix (%s GB), "
        "but only %s GB RAM is available. Falling back to random initialization. "
        "Pass init=\"random\" to suppress this warning.",
        to_gb_str(est_bytes), to_gb_str(avail_bytes)
    );
    init_mode = 0;  // random
}
```

**`get_available_ram_bytes()` implementation** (add to a shared utility header,
e.g., `inst/include/FactorNet/core/platform.hpp` or inline in the function):

```cpp
inline uint64_t get_available_ram_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return (uint64_t)status.ullAvailPhys;
#elif defined(__APPLE__)
    // macOS: sysctl vm.page_free_count * page_size
    int64_t page_size = sysconf(_SC_PAGE_SIZE);
    int64_t free_pages = 0;
    size_t len = sizeof(free_pages);
    sysctlbyname("vm.page_free_count", &free_pages, &len, nullptr, 0);
    return (uint64_t)(page_size * free_pages);
#else
    // Linux: read MemAvailable from /proc/meminfo
    std::ifstream f("/proc/meminfo");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("MemAvailable:", 0) == 0) {
            uint64_t kb = 0;
            sscanf(line.c_str(), "MemAvailable: %llu kB", &kb);
            return kb * 1024ULL;
        }
    }
    return 0;  // unknown — caller should treat as "insufficient"
#endif
}
```

This same utility will be used by WORKQUEUE_3 for auto-dispatch. Place it in a
shared header so WORKQUEUE_3 can include it without duplication.

---

## Task 4 — Replace std::async Per-Chunk with Persistent Thread (§2.2.4)

**File**: `inst/include/FactorNet/nmf/fit_chunked.hpp`  
**Context**: The current prefetch pattern spawns and joins a new OS thread per chunk:
```cpp
auto prefetch = std::async(std::launch::async, [&loader, &chunk_next]() {
    return loader.next_forward(*chunk_next);
});
// ... compute ...
bool has_next = prefetch.get();
```
This works but has non-trivial overhead (~10–50 µs per thread creation) that is
wasted for fast NVMe storage.

**Fix**: Replace with a persistent 2-slot ping-pong buffer using a dedicated
background I/O thread. This is a well-known pattern:

```cpp
// PingPongPrefetcher<ChunkType>: manages a background thread that pre-loads
// the next chunk while the current chunk is being processed.
template<typename ChunkType>
class PingPongPrefetcher {
    std::array<ChunkType, 2> slots_;
    int current_ = 0;
    std::thread worker_;
    std::mutex mu_;
    std::condition_variable cv_ready_, cv_free_;
    bool slot_ready_[2] = {false, false};
    bool slot_free_[2] = {true, true};
    bool done_ = false;
    std::function<bool(ChunkType&)> load_fn_;
public:
    PingPongPrefetcher(std::function<bool(ChunkType&)> load_fn);
    ~PingPongPrefetcher();  // signals done, joins worker_
    bool get_next(ChunkType& out);  // blocks until next chunk is ready
};
```

The background thread calls `load_fn_(slots_[next_slot])` while the main thread
processes `slots_[current_]`.

Replace both the forward pass and the transpose pass `std::async` calls with this
abstraction. The implementation should be a standalone header at
`inst/include/FactorNet/io/ping_pong_prefetch.hpp` so it can also be used
directly in the GPU streaming path.

After replacement, run the streaming tests to confirm no regression and check that
wall time is the same or better:
```bash
ssh c001 'cd /mnt/home/debruinz/RcppML-2 && module load r/4.5.2 && OMP_NUM_THREADS=4 Rscript -e "
  library(RcppML)
  data(monocle3_subset)
  f <- tempfile(fileext=\".spz\")
  sp_write(monocle3_subset, f)
  system.time(nmf(f, k=8, maxit=5))
  unlink(f)
"'
```

**Note**: `std::thread` requires linking `-lpthread`. RcppML should already link
this via SystemRequirements in `DESCRIPTION`. Check if `DESCRIPTION` has:
```
SystemRequirements: C++14, OpenMP
```
and add `-lpthread` to `src/Makevars` if thread tests fail to link:
```makefile
PKG_LIBS += -lpthread
```

---

## Task 5 — Replace `goto` in Symmetric NMF Path (§2.2.5)

**File**: `inst/include/FactorNet/nmf/fit_chunked.hpp`  
**Context**: The symmetric NMF path uses a `goto forward_done;` to skip ~200 lines
of forward pass code. Find lines containing `goto forward_done` and `forward_done:`.

**Fix**: Extract the forward pass as a lambda and call it conditionally:

```cpp
// Before the forward/W-update section, define:
auto run_forward_pass = [&]() -> void {
    // ... move the entire forward H-update block into this lambda ...
};

// Then replace the goto with:
if (!is_symmetric) {
    run_forward_pass();
}
// forward_done: label removed
```

This is a pure refactor — correctness is identical. Verify by running:
```bash
ssh c001 '... R --no-save -e "devtools::test(filter=\"nmf\")"'
```

---

## Task 6 — IRLS Loss in Streaming Loss History (§2.2.6)

**File**: `inst/include/FactorNet/nmf/fit_chunked.hpp`  
**Context**: When `use_irls=true` (GP, NB, etc.), `result.loss_history` stores MSE
values, not the actual IRLS objective (e.g., GP log-likelihood).

**Find**: Search for where `train_loss_accum` is accumulated in `fit_chunked.hpp`.
It should be something like `train_loss_accum += diff * diff` or a Gram-trick
`trAtA - 2*cross_term + recon_norm`.

**Fix**: When `use_irls=true`, replace the MSE accumulation with the correct
per-column distribution log-likelihood. Look at `fit_unified.hpp` for how it
computes the IRLS loss per-column (it uses a `compute_irls_loss()` function or
similar). Port that logic into the streaming path's per-column loop.

At minimum, add a comment when `use_irls=true` is active:
```
// NOTE: when use_irls=true, loss_history contains IRLS objective (not MSE)
```
and ensure the correct accumulation formula is used.

After fixing, verify with:
```bash
ssh c001 '... R --no-save -e "
  library(RcppML)
  data(monocle3_subset)
  f <- tempfile(fileext=\".spz\")
  sp_write(monocle3_subset, f)
  res <- nmf(f, k=4, maxit=5, loss=\"gp\")
  # Loss history should decrease and be of order log-likelihood, not MSE
  print(res@loss_history)
  unlink(f)
"'
```

---

## Task 7 — solver="auto" Adaptive CD↔Cholesky Crossover (§2.4)

**File**: `R/nmf_thin.R`  
**Context**: `solver="auto"` currently maps to `solver_mode=0L` (CD) via the
`switch` default at line ~605:
```r
solver_mode <- switch(solver, cd = 0L, cholesky = 1L, 0L)
```

**Fix**: Implement the advertised auto-selection logic. Empirically:
- **Cholesky** is faster for small k (k < 32) with MSE loss and no L1 regularization
- **CD** is required for L1 > 0 (Cholesky cannot enforce L1 sparsity)
- **CD** is better for large k (k ≥ 32) or IRLS losses

Replace the switch with:
```r
if (solver == "auto") {
  # Cholesky+clip is faster for small k with MSE and no L1 regularization
  if (k < 32 && loss == "mse" && l1_w == 0 && l1_h == 0) {
    solver_mode <- 1L  # Cholesky
  } else {
    solver_mode <- 0L  # CD
  }
} else {
  solver_mode <- switch(solver, cd = 0L, cholesky = 1L, 0L)
}
```

Find the exact variable names for `l1_w` and `l1_h` in `nmf_thin.R` — they may
be called `l1` or `l1_w`/`l1_h` depending on the function signature. Also find
where `loss` is normalized (it may be called `distribution` or `loss_type` by
this point in the function).

Add a test for `solver="auto"` behavior in `tests/testthat/test_solver.R` (create
if it doesn't exist) or add to an existing solver test file:
```r
test_that("solver='auto' selects Cholesky for small k MSE", {
  res <- nmf(monocle3_subset, k=8, solver="auto", loss="mse")
  expect_equal(res@diagnostics$solver_mode, 1L)  # 1 = Cholesky
})
test_that("solver='auto' selects CD when L1 > 0", {
  res <- nmf(monocle3_subset, k=8, solver="auto", loss="mse", L1=0.1)
  expect_equal(res@diagnostics$solver_mode, 0L)  # 0 = CD
})
```

The `solver_mode` value needs to be accessible in diagnostics — check if it's
already in `result@diagnostics` or add it.

---

## Task 8 — Final Validation

After all tasks are complete:

1. Run the full CPU test suite:
   ```bash
   ssh c001 'cd /mnt/home/debruinz/RcppML-2 && module load r/4.5.2 && OMP_NUM_THREADS=4 R --no-save -e "devtools::test()" 2>&1 | tail -20'
   ```
   Target: `[ FAIL 0 ]`. Fix any regressions before committing.

2. Update `PRODUCTION_AUDIT.md`: mark each completed item (§2.2.1–2.2.6, §2.4 items
   1 and 2) as ✅ with a brief note on what was done.

3. Commit all changes:
   ```bash
   ssh c001 'cd /mnt/home/debruinz/RcppML-2 && git add -A && git commit -m "fix(wq2): streaming algo correctness - NB dispersion, CV mask, SVD init, goto, IRLS loss, solver=auto"'
   ```

---

## Completion Criteria

All of the following must be true:

1. `devtools::test()` produces `[ FAIL 0 ]` (no regressions from baseline)
2. **§2.2.1**: Either (a) R-level warning fires when streaming + NB + no nb_size, OR
   (b) full streaming dispersion re-estimation is implemented
3. **§2.2.2**: mask_zeros=FALSE CV no longer scans all m rows per column
4. **§2.2.3**: SVD init with large streaming matrix warns and falls back to random init;
   `get_available_ram_bytes()` utility exists in a shared header
5. **§2.2.4**: `std::async` per-chunk replaced with `PingPongPrefetcher` (or equivalent
   persistent thread); `inst/include/FactorNet/io/ping_pong_prefetch.hpp` exists
6. **§2.2.5**: `goto` is gone from `fit_chunked.hpp`; all streaming NMF tests pass
7. **§2.2.6**: `loss_history` contains IRLS objective (not MSE) when `use_irls=true`
8. **§2.4/solver**: `solver="auto"` selects Cholesky for k<32 MSE no-L1, CD otherwise;
   test coverage added
9. `PRODUCTION_AUDIT.md` §2.2 and §2.4 items updated with ✅

**Do not modify** `inst/include/FactorNet/io/spz_loader.hpp` or
`inst/include/FactorNet/io/dense_spz_loader.hpp` — those are WORKQUEUE_3 property.
Do not modify anything in `inst/include/sparsepress/` — WORKQUEUE_3.
