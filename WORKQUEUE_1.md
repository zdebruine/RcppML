# WORKQUEUE_1 — CRAN Readiness, Documentation & Paper Outlines

**Agent**: Claude Opus 4.6 — autonomous implementation mode  
**Parallel agents**: Two other agents (WORKQUEUE_2, WORKQUEUE_3) are running simultaneously
on different source files. Your scope is **R-level code, vignettes, documentation,
and CRAN compliance only**. You do NOT touch any C++ files under `inst/include/`.
File overlap is safe: none of the files you edit are the same as WORKQUEUE_2 or
WORKQUEUE_3.

---

## Project Context

**RcppML** is an R package (`/mnt/home/debruinz/RcppML-2`) for Non-negative Matrix
Factorization (NMF) using Rcpp and Eigen. It includes:

- Standard NMF with statistical distributions (MSE, Gaussian-Poisson, NB, Gamma,
  InvGauss, Tweedie, Zero-Inflated variants)
- Cross-validation NMF (`cv_nmf()`)
- GPU-accelerated NMF via CUDA
- Out-of-core streaming NMF from `.spz` sparse/dense files (SparsePress format)
- FactorNet: a graph DSL for multi-modal NMF (`factor_net()`, `link_nmf()`, etc.)
- SVD variants (deflation, Krylov, Lanczos, IRLBA, randomized)
- Guided NMF with classifiers

Key directories:
- `R/` — R source files (user-facing functions, roxygen comments)  
- `man/` — **AUTO-GENERATED** Rd files — NEVER edit directly  
- `NAMESPACE` — **AUTO-GENERATED** — NEVER edit directly  
- `src/RcppExports.cpp`, `R/RcppExports.R` — **AUTO-GENERATED** — NEVER edit  
- `tests/testthat/` — test suite  
- `vignettes/` — package vignettes  
- `docs/dev/` — development docs: `COVERAGE_MATRIX.yaml`, archived plans  
- `docs/papers/` — create this directory for paper outlines (may not exist yet)

The authoritative work-tracking document is `PRODUCTION_AUDIT.md` in the repo root.
Read it in full at the start — it contains the complete rationale for every task.

---

## Compute Environment (CRITICAL — READ CAREFULLY)

You are on a SLURM HPC cluster. The login node is `port` (hostname
`port.clipper.gvsu.edu`). **NEVER run R, Rscript, Python, or any compute on
`port`**. Always SSH to a compute node.

Available compute nodes (SSH directly — you already have SLURM allocations there):
- **c001, c004** — CPU (general)  
- **b004** — bigmem  
- **g051** — GPU (only needed if testing GPU features)

```bash
# Preferred pattern — wrap all compute in a single ssh call:
ssh c001 'cd /mnt/home/debruinz/RcppML-2 && module load r/4.5.2 && R --no-save -e "
  devtools::document()
"'
# Then immediately:
ssh c001 'cd /mnt/home/debruinz/RcppML-2 && bash tools/fix_rcpp_info_bug.sh'
```

Pre-approved terminal commands (no approval prompt): `ssh`, `squeue`, `sacct`,
`R`, `Rscript`, `hostname`, `cat`, `head`, `tail`, `ls`, `git`, `module`.

Use VS Code file tools (`read_file`, `grep_search`, `file_search`, `list_dir`)
for all read-only operations — no terminal needed.

### Build Workflow

After editing any R source file with roxygen comments:
```bash
ssh c001 'cd /mnt/home/debruinz/RcppML-2 && module load r/4.5.2 && R --no-save -e "devtools::document()"'
ssh c001 'cd /mnt/home/debruinz/RcppML-2 && bash tools/fix_rcpp_info_bug.sh'
ssh c001 'cd /mnt/home/debruinz/RcppML-2 && module load r/4.5.2 && R --no-save -e "devtools::install(quick=TRUE)"'
```

**CRITICAL**: After *every* `devtools::document()` call, always run
`bash tools/fix_rcpp_info_bug.sh`. There is a known Rcpp 1.1.0 bug that
inserts an undefined `info` symbol into `src/RcppExports.cpp`. The script
patches it. Failure to do this causes the package to fail to load.

### Running Tests
```bash
ssh c001 'cd /mnt/home/debruinz/RcppML-2 && module load r/4.5.2 && OMP_NUM_THREADS=4 R --no-save -e "devtools::test()"'
```
Current baseline: `[ FAIL 0 | WARN 15 | SKIP 148 | PASS 1987 ]`. Do not regress this.

---

## Task 1 — Audit "Planned" Entries in COVERAGE_MATRIX (§2.1.1)

**File**: `docs/dev/COVERAGE_MATRIX.yaml`  
**Risk**: LOW (R-level guards only; no C++ changes)

The coverage matrix has ~123 entries with `status: planned`. Every user-reachable
path must either be implemented or blocked with a clear `stop()`.

**Steps**:
1. Read `docs/dev/COVERAGE_MATRIX.yaml` in full.
2. For each `status: planned` entry, determine if it is:
   - **(a) Already implemented** but the YAML wasn't updated → update YAML to
     `status: complete` with a note.
   - **(b) User-reachable and unimplemented** → add a `stop("...", call.=FALSE)`
     guard in `R/nmf_thin.R` (or the appropriate R function) with a clear message
     like `"'semi_nmf' is not yet implemented. Use nonneg_W=TRUE."`.
   - **(c) Internal/test-only** → mark `status: internal`, no guard needed.
3. Update `docs/dev/COVERAGE_MATRIX.yaml` with audited statuses.

**Key paths to audit** (from `PRODUCTION_AUDIT.md §2.1.1`):
- Dense streaming (SPZ v3): `nmf("file.spz", ...)` with a v3 dense `.spz` — check
  whether `R/nmf_thin.R` routes to an implemented branch.
- Semi-NMF (`nonneg_W = FALSE`): Is this parameter exposed? Does it work or silently
  produce bad results? Find where `nonneg_W` is used in `nmf_thin.R`.
- GPU dense + non-MSE: `resource="gpu"` with dense matrix + `loss="gp"` — does it
  work, fall back gracefully, or crash?
- Standalone GPU NNLS: Is `nnls()` exposed for GPU acceleration? If parameter
  documentation implies it but it isn't implemented, add a guard.

---

## Task 2 — Fix Vignettes (§2.1.2)

**Files**: `vignettes/sparsepress.Rmd`, `vignettes/nmf-deep-dive.Rmd`  
Both are currently excluded in `.Rbuildignore`.

**Steps**:
1. Try to render each vignette:
   ```bash
   ssh c001 'cd /mnt/home/debruinz/RcppML-2 && module load r/4.5.2 && OMP_NUM_THREADS=4 R --no-save -e "
     rmarkdown::render(\"vignettes/sparsepress.Rmd\", quiet=FALSE)
   "'
   ```
2. For any build failure: fix the error. Common issues:
   - Missing package (add to `Suggests:` in `DESCRIPTION`)
   - Slow code block → wrap in `{r, eval=FALSE}` or use precomputed results
   - SPZ file paths that no longer exist → use `inst/extdata/pbmc3k.spz`
3. If a vignette builds cleanly: remove its exclusion line from `.Rbuildignore`.
4. If a vignette is genuinely too slow for CRAN (> 60 seconds total): leave it
   excluded and add a comment to `.Rbuildignore` explaining why. Make sure
   `vignettes/` still contains the source for pkgdown to pick up.
5. Update CRAN checklist item in `PRODUCTION_AUDIT.md §4` accordingly.

---

## Task 3 — Example Runtimes (§2.1.3)

**Goal**: All `man/*.Rd` examples run in under 5 seconds on CRAN hardware.

**Steps**:
1. Run example checks:
   ```bash
   ssh c001 'cd /mnt/home/debruinz/RcppML-2 && module load r/4.5.2 && OMP_NUM_THREADS=4 R --no-save -e "
     devtools::check_examples(run_donttest = FALSE)
   "'
   ```
2. Identify which examples are slow (look for `>5s` in output).
3. For each slow example: wrap the slow part in `\donttest{...}` in the **R source
   file** (the roxygen `@examples` block in `R/*.R`), NOT in `man/*.Rd`. Then
   rebuild docs (`devtools::document()`).
4. Likely offenders: `nmf()` with `k > 4` on movielens/monocle3_subset, all
   `cv_nmf()` examples, SPZ streaming examples, GPU examples.
5. Re-run `check_examples()` to confirm no example exceeds 5s.

---

## Task 4 — Package Tarball Size (§2.1.4)

**Goal**: CRAN tarball < 5 MB.

**Steps**:
1. Build the tarball and check size:
   ```bash
   ssh c001 'cd /mnt/home/debruinz/RcppML-2 && module load r/4.5.2 && R CMD build . 2>&1 | tail -5'
   ssh c001 'ls -lh /mnt/home/debruinz/RcppML-2/RcppML_*.tar.gz 2>/dev/null || ls -lh /tmp/RcppML_*.tar.gz 2>/dev/null'
   ```
2. If > 5 MB, identify the largest data objects:
   ```bash
   ssh c001 'cd /mnt/home/debruinz/RcppML-2 && module load r/4.5.2 && R --no-save -e "
     fns <- list.files(\"data\", full.names=TRUE)
     sort(sapply(fns, file.size) / 1e6, decreasing=TRUE)
   "'
   ```
3. If `data/digits_full.rda` or `data/olivetti.rda` are large, trim to a 1000-row
   subsample. Document the trim in `R/data.R` roxygen comments.
4. If `inst/extdata/pbmc3k.spz` is large, check if it can be replaced with a
   smaller synthetic dataset for examples (preserve the real pbmc3k for NOT_CRAN tests).
5. Rebuild and verify tarball is < 5 MB.

---

## Task 5 — README.md Rewrite (§2.3.1)

The current `README.md` is outdated. Write a comprehensive replacement.

**Source material to read first**:
- `docs/factornet/ARCHITECTURE.md` — architectural overview
- `docs/factornet/API_REFERENCE.md` — complete API reference
- `docs/factornet/algorithms/nmf.md`, `distributions.md`, `sparsepress.md` — algorithmic detail
- `R/nmf_thin.R` lines 1–350 — the complete `nmf()` function signature and docs

**Sections the new README must contain** (maintain this order):

1. **Header**: Package name, one-line description, badges (CRAN status, R-CMD-check, license)
2. **Overview**: 3–5 sentence description covering NMF, distributions, GPU, streaming, FactorNet
3. **Installation**: `install.packages("RcppML")` and `devtools::install_github("zdebruine/RcppML")`
4. **Quick Start**: 5–10 line runnable example using `monocle3_subset` or `pbmc3k` from the package:
   ```r
   library(RcppML)
   data(monocle3_subset)
   result <- nmf(monocle3_subset, k = 10)
   plot(result)
   ```
5. **Statistical Distributions**: Brief table of supported `loss=` values with use-cases
6. **Cross-Validation**: 3–5 line `cv_nmf()` example
7. **GPU Acceleration**: Brief overview, `gpu_available()`, `resource="gpu"` parameter
8. **Streaming Large Data** (`.spz` files): Brief `sp_write()` + `nmf("file.spz")` example;
   note that the format will be renamed to StreamPress in an upcoming release
9. **FactorNet Graph API**: 5-line `factor_net()` example
10. **Contributing**: link to `CONTRIBUTING.md`
11. **Citation**: `citation("RcppML")`

Length: 400–700 lines. Runnable examples only use built-in package data
(`monocle3_subset`, `pbmc3k`, etc.). No benchmark tables.

---

## Task 6 — Roxygen Coverage Gaps (§2.3.2)

**Steps**:
1. Run documentation check:
   ```bash
   ssh c001 'cd /mnt/home/debruinz/RcppML-2 && module load r/4.5.2 && R --no-save -e "devtools::check_man()"'
   ```
2. Add missing `@param`, `@return`, `@examples` to all functions that fail. Focus on:
   - `R/factor_net.R`: `factor_net()`, `link_nmf()`, `link_svd()`, `link_graph()`
   - `R/auto_distribution.R`: `auto_nmf_distribution()`, `diagnose_dispersion()`,
     `diagnose_zero_inflation()`
   - `R/gpu_backend.R`: `gpu_available()`, `gpu_info()`
   - `R/sp_gpu.R`: `sp_read_gpu()`, `sp_free_gpu()`
   - `R/classifier_metrics.R` (or wherever): `classify_embedding()`,
     `classify_logistic()`, `classify_rf()`
   - `R/training_log.R`: `training_logger()`, `export_log()`
3. Each function needs at minimum:
   - `@param name type. Description.` for every parameter
   - `@return Description of return value`  
   - `@examples` with at least one runnable line (wrap slow/GPU in `\donttest{}`)
4. Rebuild and re-run `check_man()` until it is clean.

---

## Task 7 — CRAN Submission Checklist (§4)

After completing Tasks 1–6, run the full CRAN check:
```bash
ssh c001 'cd /mnt/home/debruinz/RcppML-2 && module load r/4.5.2 && OMP_NUM_THREADS=4 R CMD check --as-cran . 2>&1 | tail -30'
```

Target: `Status: OK` or `Status: 1 NOTE` (environment artifacts acceptable).  
Zero `ERROR`s and zero `WARNING`s required.

Update `PRODUCTION_AUDIT.md §4` checklist: mark each completed item with ✅.
Use `str_replace` edits to the file.

**NEWS.md**: Add a `## RcppML 1.0.1` section at the top of `NEWS.md` summarizing
the changes made in this WORKQUEUE (documentation, CRAN compliance, etc.) if one
doesn't already exist.

---

## Task 8 — Paper Outlines (§5)

Create `docs/papers/` directory (if it doesn't exist) and write outline stubs for
all six planned publications. Each outline should be 100–200 lines.

**Files to create**:

### `docs/papers/P1_rcppml_overview.md`
Paper: *RcppML: High-performance NMF for R* (target: JSS or R Journal)
Include: abstract draft, 5 key contributions, related packages table (RcppML vs
NMF r-package vs singlet vs NNLM), benchmark requirements (time vs k, time vs n,
memory scaling), figure list, recommended venue.

### `docs/papers/P2_gpu_cv_gram.md`
Paper: *GPU-accelerated cross-validation for NMF via per-column Gram correction*
(target: Bioinformatics or JCGS)
Include: the mathematical derivation of the per-column Gram correction for
lazy holdout masks, GPU implementation sketch, benchmark design (wall time speedup
vs CPU, quality of CV curve), figure list.

### `docs/papers/P3_irls_framework.md`
Paper: *IRLS-based NMF for exponential family distributions* (target: Biostatistics)
Include: IRLS algorithm for NMF, convergence analysis outline, distribution support
table, benchmark vs standard Poisson NMF packages.

### `docs/papers/P4_streampress.md`
Paper: *StreamPress: a streaming format for out-of-core matrix factorization*
(target: SoftwareX)
Include: format spec summary (v2 sparse, v3 dense compression), loader architecture,
auto-dispatch algorithm, benchmark design — **the flagship benchmark is
`nmf(geo_spz_path, k=64, loss="nb")` on a GEO reprocessed single-cell corpus
running on HPC without user configuration** — figure list, memory scaling chart.

### `docs/papers/P5_factornet_graph.md`
Paper: *FactorNet: a graph DSL for multi-modal matrix factorization* (target: JMLR or NeurIPS)
Include: graph DSL design, multi-modal integration approach, comparison to MOFA/LIGER,
benchmark on multi-modal datasets, figure list.

### `docs/papers/P6_constrained_svd.md`
Paper: *Efficient constrained SVD variants for non-negative matrix analysis*
(target: Computational Statistics)
Include: SVD algorithm catalog (deflation, Krylov, Lanczos, IRLBA, randomized),
constraints handled, benchmark vs irlba/rsvd packages, stability analysis.

---

## Task 9 — Benchmark Harness Skeleton (§2.4)

Create a structured benchmark harness directory. This is a tooling/organizational
task, not a performance fix.

**Create** `benchmarks/harness/README.md` describing the harness organization:
- Purpose: reproducible, versioned benchmarks for all six papers
- Directory structure: `benchmarks/harness/{nmf_speed, cv_quality, gpu_compare, streaming_spz}/`
- Template `bench_template.R`: load data, run benchmark with `microbenchmark` or
  `bench::mark`, record system info, save results to `benchmarks/results/`

**Create** `benchmarks/harness/bench_template.R` — a template that:
1. Captures system metadata (`sessionInfo()`, `Sys.time()`, node hostname,
   `OMP_NUM_THREADS`, GPU if available)
2. Loads a dataset (parameterized by `DATA_PATH` env var)
3. Runs timed NMF with `Sys.time()` around iterations
4. Saves results as a named list to an RDS file in `benchmarks/results/`

Create the `benchmarks/results/` directory with a `.gitignore` excluding `*.rds`
(results are large and machine-specific, not version-controlled).

---

## Completion Criteria

Your work is done when ALL of the following are true:

1. `R CMD check --as-cran` produces `Status: OK` or `Status: 1 NOTE` only (0 ERRORS, 0 WARNINGS)
2. `devtools::test()` produces `[ FAIL 0 ]` (do not regress the baseline)
3. `devtools::check_man()` is clean (all exported functions documented)
4. `devtools::check_examples(run_donttest=FALSE)` completes without examples timing out
5. Package tarball is < 5 MB
6. README.md is rewritten and comprehensive
7. `docs/papers/P{1..6}_*.md` all exist with outline content
8. `docs/dev/COVERAGE_MATRIX.yaml` has all "planned" entries audited
9. `PRODUCTION_AUDIT.md §4` checklist is updated with ✅ for completed items
10. All changes committed: `git add -A && git commit -m "feat(wq1): CRAN readiness, docs, vignettes, paper outlines"`

**Do not modify** any file under `inst/include/` — that is WORKQUEUE_2 and
WORKQUEUE_3 territory. Do not modify `src/RcppFunctions.cpp` unless adding an
R-level guard that calls `stop()` (no Rcpp compilation needed for that).
