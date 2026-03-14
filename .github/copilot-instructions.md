# Copilot Instructions for RcppML Development

## Project Overview

RcppML is an R package for high-performance Non-negative Matrix Factorization (NMF) using Rcpp and Eigen. The codebase includes both standard NMF and cross-validation NMF with support for sparse matrices.

## Lessons Learned from Testing & Benchmarking

### 1. Metric Definition Matters

**Critical Discovery**: Different packages may compute the same-named metric differently.

- **singlet's `test_error`**: Includes ALL matrix entries (zeros + nonzeros) in the test set
- **RcppML's `sparse=TRUE` test loss**: Only non-zero entries in test set
- **RcppML's `sparse=FALSE` test loss**: All matrix entries (matches singlet)

**Lesson**: When comparing packages, always verify the exact metric computation, not just the name. A 26x difference in "MSE" was traced to different test set definitions, not a bug.

### 2. Cross-Validation Test Set Semantics

For **recommendation systems** (e.g., MovieLens ratings):
- `sparse=TRUE` (mask_zeros=TRUE): Only predict existing ratings → more appropriate for recommendation
- `sparse=FALSE` (mask_zeros=FALSE): Predict all entries → measures full matrix reconstruction

For **dense data** (e.g., images, gene expression):
- Both modes are similar since most entries are non-zero
- `sparse=FALSE` is the standard choice

### 3. RNG Reproducibility Across Packages

Different packages use different random number generators:
- **singlet**: Custom hash-based RNG for speckled mask (`seed.draw(j, i, inv_density)`)
- **RcppML**: Unified RNG class with reproducible seeds

**Lesson**: Even with the same seed, different RNG implementations produce different masks. Focus on statistical equivalence, not exact replication.

### 4. Scalability Boundaries

- **singlet CV**: Crashes for k > 16 on most datasets
- **RcppML CV**: Works for all tested ranks (k up to 64)

**Lesson**: Always test edge cases (high rank, large datasets) during benchmarking.

### 5. Performance Profiling Strategy

When profiling NMF:
1. **NNLS solver**: O(k²m) or O(k²n) per update
2. **Gram matrix computation**: O(k²n) for H'H, O(k²m) for W'W
3. **Loss computation**: Can be O(nnz·k) or O(m·n·k) depending on mask_zeros
4. **Memory access patterns**: Column-major iteration is faster for sparse CSC matrices

### 6. Benchmarking Best Practices

1. **Force full iterations**: Use `tol=1e-10` to ensure maxit iterations run
2. **Multiple replicates**: At least 3 to capture variance
3. **Warm-up**: First run may be slower due to JIT/cache effects
4. **Separate timing**: Measure different phases (H update, W update, loss) independently

## Code Organization

### R Layer (`R/`)
- `nmf_thin.R`: Main user-facing `nmf()` function with parameter validation and dispatch
- `nmf_validation.R`: Parameter validation helpers
- `nmf_methods.R`: S4 methods for nmf class (subset, align, summary, etc.)
- `nmf_plots.R`: biplot, plot.nmfSummary
- `plot_nmf.R`: plot.nmf, compare_nmf, plot.nmfCrossValidate
- `consensus.R`: consensus_nmf, plot/summary methods
- `sparsepress.R`: SPZ file I/O wrappers

### C++ Layer (`src/` and `inst/include/`)
- `RcppFunctions.cpp`: R-callable wrapper functions
- `inst/include/RcppML/gateway/nmf_full.hpp`: Unified NMF gateway (dispatches to all paths)
- `inst/include/RcppML/nmf/`: Core NMF algorithms
  - `fit_unified.hpp`: Standard NMF fitting (CPU, sparse/dense, all features)
  - `fit_cv_unified.hpp`: Cross-validation NMF with lazy mask + IRLS
  - `fit_streaming_spz.hpp`: Streaming out-of-core NMF for SPZ files
- `inst/include/RcppML/core/`: Config and result types
- `inst/include/RcppML/primitives/cpu/`: NNLS batch solvers (MSE + IRLS)

## Auto-Generated Files — NEVER Modify Manually

The following files are **auto-generated** by `roxygen2` and `Rcpp::compileAttributes()`. **NEVER edit them by hand** — any manual changes will be silently overwritten.

| File(s) | Generator | How to change |
|---------|-----------|---------------|
| **`NAMESPACE`** | `roxygen2` via `devtools::document()` | Edit `@export`, `@importFrom`, `@useDynLib` tags in `R/*.R` source files |
| **`man/*.Rd`** | `roxygen2` via `devtools::document()` | Edit `@param`, `@return`, `@examples` etc. in roxygen comments in `R/*.R` |
| **`src/RcppExports.cpp`** | `Rcpp::compileAttributes()` via `devtools::document()` | Edit `// [[Rcpp::export]]` annotations in C++ source (`src/` or `inst/include/`) |
| **`R/RcppExports.R`** | `Rcpp::compileAttributes()` via `devtools::document()` | Same as above — add/change exported C++ functions in the C++ source |

**Workflow**: After any changes to roxygen comments or `// [[Rcpp::export]]` annotations, regenerate **all** auto-generated files:
```r
devtools::document()
```
Then run the Rcpp bug fix script:
```bash
bash tools/fix_rcpp_info_bug.sh
```

**CRITICAL**: If you need to change exports, documentation, or Rcpp bindings, always edit the **source** (R/*.R or C++ files), never the generated output. This applies to both human developers and AI assistants.

## Common Pitfalls

1. **Sparse matrix format**: RcppML expects dgCMatrix (CSC format). Convert with `as(x, "dgCMatrix")`.

2. **Factor scaling**: RcppML uses W·diag(d)·H factorization. Don't forget the diagonal scaling.

3. **Transposed storage**: Internal W is stored as k×m (transposed) for cache efficiency.

4. **NA handling**: Use `mask="NA"` parameter; C++ cannot handle R's NA directly.

5. **Thread safety**: OpenMP parallelization requires proper reduction clauses for shared variables.

6. **C++ API indexing**: `bipartiteMatch()` returns 0-indexed `$assignment` (not `$pairs`). `dclust()` returns 0-indexed `$samples` and character `$id` (binary path string encoding split hierarchy, e.g. "01" means root→left→right). Always check C++ struct definitions for actual field names and types.

7. **Rcpp 1.1.0 info bug**: After every `roxygenise()`, run `bash tools/fix_rcpp_info_bug.sh`. The bug inserts an undefined `info` symbol into `RcppExports.cpp`. Never use `roxygenise(".", clean = TRUE)` — it fails because it tries to dyn.load the .so with the bug.

8. **GCC on cluster**: `using RcppML::LossConfig;` declarations don't work — must use full namespace qualification (`RcppML::LossConfig<Scalar> loss_config;`).

9. **Streaming NMF supports all losses**: Non-MSE losses (GP/NB/Gamma/etc.) work in the streaming SPZ path via `weight_zeros=true` in the IRLS solver, which computes correct distribution-derived weights at zero entries instead of defaulting to unit weights. The chunked paths in `fit_chunked.hpp` and `fit_chunked_gpu.cuh` pass `weight_zeros=true` to `irls_nnls_col_sparse()`. The in-memory batch IRLS path uses the default `weight_zeros=false` (sparse approximation) for performance.

## Testing Guidelines

1. **Unit tests**: Focus on numerical correctness with known ground truth
2. **Integration tests**: Test full pipeline with real datasets
3. **Benchmark tests**: Compare against singlet for performance regression
4. **Edge cases**: k=1, k=max_rank-1, empty columns, all-zero rows

## Documentation Standards

1. **Roxygen**: All exported functions need complete documentation
2. **Examples**: Include runnable examples with package datasets
3. **Vignettes**: Tutorial-style documentation for complex features
4. **Benchmark reports**: Include methodology, datasets, and reproducibility instructions

## Compute Environment

This project runs on a SLURM-managed HPC cluster.

## CRITICAL: NEVER Execute Commands on the Login Node — MANDATORY

> **THIS IS THE SINGLE MOST IMPORTANT RULE. VIOLATING IT WILL DISRUPT THE ENTIRE CLUSTER FOR ALL USERS.**

**The login/submit node is called `port` (hostname: `port.clipper.gvsu.edu`).** You must **NEVER, UNDER ANY CIRCUMSTANCES**, execute **ANY** command on `port` other than:
- `sbatch` (job submission)
- `squeue` / `sacct` (job status)
- `srun` (to obtain an interactive session — not to run compute)
- `cat` / `head` / `tail` on small log files
- `ls` on small directories
- File editing (creating/editing scripts)

**EVERYTHING ELSE IS FORBIDDEN ON `port`.** This includes but is not limited to:
- Any `python` or `python3` invocation (even "quick" one-liners)
- Any `R` or `Rscript` invocation
- Reading/writing parquet files, CSVs, or any data processing
- Network I/O (API calls, downloads, `curl`, `wget`)
- `find`, `grep`, or `ls` over large directory trees on NFS
- Running compiled binaries (simpleaf, salmon, etc.)
- Any script execution whatsoever

### MANDATORY PRE-FLIGHT CHECK — Every Single Time

**Before executing ANY command (other than the allowed list above), you MUST run `hostname` and verify the output is NOT `port`.** If the hostname is `port` or `port.clipper.gvsu.edu`, you are on the login node and **MUST stop immediately**. Do NOT proceed. Obtain a compute node first (see SSH method below).

```bash
# ALWAYS run this before ANY work:
hostname
# If output contains "port" → STOP. SSH to a compute node first.
# If output is like "b004", "c001", "g001" → You are on a compute node. Proceed.
```

**This check is not optional. It must happen every time, even if you "think" you're on a compute node.** Terminal sessions can disconnect, time out, or be reassigned. Always verify.

---

## CURRENT HPC STATE: All Partitions at Quota — SSH-Only Execution

### The Problem

The user has a large number of long-running array jobs consuming their full per-user node quota on **every partition** (cpu, bigmem, gpu). SLURM reports `QOSMaxNodePerUserLimit` for pending jobs on all partitions. This means:

- **`sbatch` will NOT work** — new batch jobs will sit in the queue indefinitely with reason `(Priority)` or `(QOSMaxNodePerUserLimit)`. Do NOT submit batch jobs expecting them to run.
- **`srun --exclusive` will NOT work** — interactive session requests will also queue indefinitely because there are no free node slots in the user's quota.
- **Existing jobs are NOT fully utilizing their nodes** — each job typically uses only a fraction of the node's CPUs (e.g., 22 of 48 on bigmem, 18 of 64 on cpu), leaving significant idle resources.

### The Solution: SSH Directly to Occupied Nodes

Since the user already has SLURM allocations on many nodes, you can **SSH directly** to any node the user is already running on. The shared filesystem (NFS) means all project files are accessible from every node.

**Step 1: Identify available nodes**
```bash
# From port (login node), check which nodes have running jobs:
squeue -u debruinz -t R -o "%N %C %P %j" | sort -u
```

**Step 2: Choose a node based on task type**
- **CPU work** (R package install, benchmarks, tests): SSH to a `c***` node (cpu partition) or `b***` node (bigmem)
- **GPU work** (CUDA tests, GPU benchmarks): SSH to a `g***` node (gpu partition)
- **Prefer nodes with fewer running jobs** to minimize resource contention

**Step 3: SSH and set up environment**
```bash
# From port, SSH to a compute node you're already on:
ssh c001   # or b004, g052, etc.

# Verify you landed correctly:
hostname   # Must NOT be "port"

# Set up environment:
module load r/4.5.2
export OMP_NUM_THREADS=4   # Use a modest thread count to coexist with running jobs
```

**Important SSH caveats:**
- You are sharing the node with existing jobs. Be respectful of resources — don't launch CPU-intensive work that would starve the existing job.
- Use a modest `OMP_NUM_THREADS` (4–8) rather than claiming all cores.
- SSH sessions are not SLURM-managed — they won't show in `squeue` and have no time limit, but will lose access if the underlying SLURM job ends.
- Monitor with `top` or `htop` on the node if unsure about resource pressure.

### Quick Reference: Currently Occupied Nodes

Run this from `port` to get an up-to-date list at any time:
```bash
squeue -u debruinz -t R -o "%N %P" | tail -n+2 | sort -u
```

Typical nodes (subject to change as jobs complete/start):
- **bigmem**: b002, b003, b004
- **cpu**: c001, c003, c004, c005, c006, c100
- **gpu**: g002, g003, g004, g005, g051, g052

### When Batch Jobs ARE Possible

Batch jobs can still work if they are configured to **share** a node the user is already on (i.e., `--nodelist=<node> --oversubscribe` or similar), but this is fragile and not recommended. Prefer the SSH approach.

If a running job finishes and frees a node slot, `sbatch` jobs may start running. Periodically check with:
```bash
squeue -u debruinz -t PD -o "%.10i %.9P %.30j %R"
```

---

## CRITICAL: Zero-Interruption Terminal Execution

### How VS Code Agent Terminal Approval Works

When the agent runs a terminal command, VS Code checks `chat.tools.terminal.autoApprove` in `.vscode/settings.json`. The setting maps **the first word of the command** to an approval rule:

- `"command": true` → **silently auto-approved**, no prompt ever shown
- `"command": false` → **always blocked**, requires manual approval
- **No matching rule** → user prompted once; "Always Allow" creates a session-scoped auto-approve

Pattern matching is on the **first token** of the command string:
- `ssh g051 'anything'` → matches `ssh`
- `cd /mnt/... && git status` → matches `cd` (NOT `git`)
- `R --no-save -e '...'` → matches `R`

**Every unique first word that lacks a `true` rule = one user interruption.** An agent that uses `ssh`, `hostname`, `squeue`, `git`, `ls`, `cat`, `diff`, `find`, `wc`, `md5sum` directly causes **10 separate approval prompts**.

### Pre-Approved Commands (`.vscode/settings.json`)

The workspace includes a `.vscode/settings.json` that pre-approves safe commands:

```json
"chat.tools.terminal.autoApprove": {
    "ssh": true,        // primary execution vehicle for compute nodes
    "squeue": true,     // SLURM job status
    "sacct": true,      // SLURM accounting
    "srun": true,       // SLURM interactive
    "sbatch": true,     // SLURM batch submit
    "scancel": true,    // SLURM cancel
    "R": true,          // R interpreter
    "Rscript": true,    // R script runner
    "hostname": true,   // node verification
    "cat": true,        // log reading
    "head": true,       // log reading
    "tail": true,       // log reading
    "ls": true,         // directory listing
    "git": true,        // version control
    "module": true,     // environment modules
    "rm": false,        // BLOCKED — destructive
    "kill": false,      // BLOCKED — destructive
    ...
}
```

With this configuration, **zero approval prompts** are needed for normal agent workflows.

### Agent Rules for Zero-Interruption Sessions

**Rule 1: Use VS Code tools instead of terminal for read-only operations.**

These require ZERO terminal commands and ZERO approvals:

| Instead of terminal... | Use this tool | Why |
|------------------------|---------------|-----|
| `cat file` / `head file` / `tail file` | `read_file` | Built-in, instant |
| `ls directory` | `list_dir` | Built-in, instant |
| `grep pattern files` | `grep_search` | Built-in, workspace-aware |
| `find . -name '*.R'` | `file_search` | Built-in, glob support |
| `wc -l file` | `read_file` (count lines) | No terminal needed |
| `diff file1 file2` | `read_file` both | Compare in your context |

**Rule 2: For compute-node work, start every command with `ssh`.**

Since the agent terminal runs on `port` (login node), ALL compute work must go through SSH:

```bash
# GOOD — uses pre-approved "ssh":
ssh c004 'cd /mnt/home/debruinz/RcppML-2 && module load r/4.5.2 && Rscript test.R'
ssh c004 'squeue -u debruinz -t R -o "%N %P" | tail -n+2 | sort -u'
ssh c004 'cd /mnt/home/debruinz/RcppML-2 && git status --short | head -20'
ssh c004 'hostname && whoami'
ssh g051 'cd /mnt/home/debruinz/RcppML-2 && module load r/4.5.2 cuda/12.8.1 && Rscript bench.R'

# BAD — uses multiple executables, each could trigger approval if not pre-approved:
cd /mnt/... && git status   # "cd" as first word
for f in ...; do ... done   # "for" — NOT pre-approved, triggers prompt
diff file1 file2            # "diff" — NOT pre-approved, triggers prompt
md5sum file1 file2          # "md5sum" — NOT pre-approved, triggers prompt
wc -l file                  # "wc" — NOT pre-approved, triggers prompt
```

**Rule 3: For commands whose first word is NOT in the pre-approved list, wrap in `ssh`.**

If you must use `diff`, `md5sum`, `wc`, `find`, `for`, `make`, `python`, or any other executable:

```bash
# Wrap in ssh so the first word is always "ssh" (pre-approved):
ssh c004 'diff file1 file2'
ssh c004 'md5sum file1 file2'
ssh c004 'wc -l file'
ssh c004 'for f in *.R; do echo "$f"; done'
```

Since all nodes share NFS, this works for any file under `/mnt/home/`.

**Rule 4: The `hostname` check is unnecessary with SSH.**

The old instruction said "run `hostname` before any command." With SSH, you inherently know which node you're on:
- `ssh c004 'command'` → runs on c004, guaranteed
- No need for a separate hostname verification step

**Rule 5: R wrappers for multi-step compute workflows.**

For build/install/test sequences, use R as the orchestrator inside SSH:

```bash
ssh c004 'cd /mnt/home/debruinz/RcppML-2 && module load r/4.5.2 && R --no-save -e "
  devtools::document()
  system(\"bash tools/fix_rcpp_info_bug.sh\")
  devtools::install(quick = TRUE)
  devtools::test()
"'
```

Or create an `.R` script and run via `ssh <node> 'Rscript path/to/script.R'`.

### Maintaining the Pre-Approved List

If a new command pattern is needed frequently, add it to `.vscode/settings.json` under `chat.tools.terminal.autoApprove`. Never add destructive commands (`rm`, `kill`, etc.) as `true`. The file is version-controlled so all collaborators benefit.

### Quick Reference

| Scenario | Method | Approvals |
|----------|--------|-----------|
| Read a file | `read_file` tool | 0 |
| List a directory | `list_dir` tool | 0 |
| Search files | `grep_search` / `file_search` tool | 0 |
| Run R on compute node | `ssh <node> 'Rscript ...'` | 0 (ssh pre-approved) |
| Build GPU library | `ssh <node> 'cd ... && make ...'` | 0 (ssh pre-approved) |
| Check SLURM queue | `squeue -u debruinz ...` | 0 (squeue pre-approved) |
| Check node identity | Not needed with SSH | 0 |
| Run `diff`/`wc`/`md5sum` | `ssh <node> 'diff ...'` or use tools | 0 |
| Destructive operations | Always require manual approval | Per-command |

---

### Available Partitions
- **debug** (2h max) — Quick iteration
- **short** (1d max) — Medium jobs
- **cpu** (5d max, default) — General compute
- **bigmem** (5d max, high-mem) — Large memory operations
- **gpu** (5d max, GPU nodes) — GPU-accelerated work

### Module Setup
- `module load r/4.5.2` on compute nodes.
- **OpenMP**: Set `OMP_NUM_THREADS=4` to `8` when sharing nodes with existing jobs (not `$SLURM_CPUS_PER_TASK`, since you are SSH'd, not in a SLURM allocation).

### Viewing Rendered HTML in VS Code on the HPC

To view rendered HTML files (e.g. vignettes, pkgdown output) in VS Code's Simple Browser:

1. Start a Python HTTP server **on the login node** (serving static files is lightweight and safe):
   ```bash
   cd /path/to/html/files && python3 -m http.server 8899 --bind 127.0.0.1
   ```
2. VS Code Remote SSH auto-forwards the port. Open in Simple Browser:
   ```
   http://localhost:8899/filename.html
   ```

Do **not** use `file://` URIs — VS Code's Simple Browser cannot resolve remote filesystem paths. Do **not** run the server on a compute node — VS Code only auto-forwards ports from the host it's connected to (the login node).
