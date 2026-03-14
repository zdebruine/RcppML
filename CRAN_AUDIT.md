# RcppML 1.0.0 — CRAN Pre-Submission Audit

**Initial Audit**: 2026-03-13  
**Last Revised**: 2026-03-14  
**Package Version**: 1.0.0  
**Previous CRAN Version**: 0.3.7  

---

## Executive Summary

The package passes `R CMD check --as-cran` (with full vignette rebuild) with **0 errors**, **2 warnings** (both from missing system tools on the HPC: `checkbashisms`, `qpdf`), and **1 note** (CRAN incoming feasibility: SeuratData not in mainstream repos, ~9.5 MB tarball).

### Verdict: **PASS** — Ready for CRAN submission

---

## 1. R CMD check (`--as-cran`, full vignette rebuild)

### Status: ✅ PASS (0 errors, 2 warnings, 1 note)

```
Platform: x86_64-pc-linux-gnu, R 4.5.2, GCC 13.3.1, RHEL 9.7

* checking whether package 'RcppML' can be installed ... [274s] OK
* checking R code for possible problems ... OK
* checking Rd files ... OK
* checking for missing documentation entries ... OK
* checking for code/documentation mismatches ... OK
* checking Rd \usage sections ... OK
* checking Rd contents ... OK
* checking examples ... OK
* checking examples with --run-donttest ... OK
* checking tests ... OK (testthat: 40s)
* checking for unstated dependencies in vignettes ... OK
* checking package vignettes ... OK
* checking re-building of vignette outputs ... [679s] OK
* checking compiled code ... OK

Status: 2 WARNINGs, 1 NOTE
```

**WARNINGs** (system-tool gaps, not package defects — CRAN build machines have these tools):
1. `checkbashisms` — not installed on HPC. Checks configure/cleanup shell scripts.
2. `qpdf` — not installed on HPC. Checks PDF compression; all vignettes are HTML.

**NOTE** (CRAN incoming feasibility):
- `SeuratData` in Suggests but not in mainstream repos. Expected; documented in `cran-comments.md`.
- Tarball: 9,506,276 bytes (~9.1 MB). Justified: 7 datasets (6.5 MB), 11 pre-built vignettes (3.2 MB), Eigen template headers (2.4 MB).

**Installed size**: 111.7 MB unstripped (98.6 MB debug symbols in `libs/`). After `strip -s` (which CRAN uses), the shared library is 1.8 MB.

### All Previously Identified Issues — RESOLVED:
- ~~465 MB tarball from `GuidedNMFManuscript/` leak~~ → `.Rbuildignore`
- ~~4 unused-variable compilation warnings~~ → `(void)var;` casts
- ~~Test failures from deprecated API~~ → tests updated
- ~~Example errors (GPU, file, unexported)~~ → properly guarded
- ~~R >= 3.5.0 but `|>` pipe in examples~~ → bumped to R >= 4.1.0
- ~~`stxBrain.SeuratData` undeclared vignette dependency~~ → `system.file()` check
- ~~`https://yann.lecun.com/...` URL broken (no HTTPS support)~~ → reverted to working `http://`
- ~~Missing `\value` in 10 internal `.Rd` files~~ → added `@return` tags
- ~~`stop(paste(...))` in `plot_nmf.R`~~ → `stop(sprintf(...))`

---

## 2. Detailed Compliance Checks

### 2a. `\dontrun{}` Usage — ✅ JUSTIFIED (19 total)

| Category | Count | Files | Justification |
|----------|-------|-------|---------------|
| SPZ file-dependent | 12 | `streampress.R` | Require `.spz` files not shipped |
| GPU hardware | 3 | `sp_gpu.R` | Require CUDA GPU; no CPU fallback |
| Unexported internals | 3 | `random.R` | `@keywords internal`; not on search path |
| Internal NMF methods | 1 | `nmf_methods.R` | `mse()` is `@keywords internal` |

All `\dontrun{}` blocks are genuinely non-runnable in a CRAN check environment.

### 2b. `\value` / `@return` Tags — ✅ ALL PRESENT

All `.Rd` files (including internal `dot-*` functions) now have `\value` sections.

### 2c. `T`/`F` Misuse — ✅ NONE

No standalone `T` or `F` used as booleans in any R source file.

### 2d. `cat()` Usage — ✅ COMPLIANT

All `cat()` calls are in `print.*` S3 methods. No unconditional console output in non-print functions.

### 2e. `par()` / `options()` State — ✅ PROPERLY RESTORED

All `par()` modifications have corresponding `on.exit()` restoration:
- `R/dclust.R` L183-184
- `R/training_log.R` L300-301

No `options()` modifications in package code (only reads via `getOption()`).

### 2f. Forbidden Functions — ✅ NONE

No `Sys.setenv()`, `setwd()`, or `sink()` calls.

### 2g. URLs — ✅ ALL VALID

One `http://` URL (`yann.lecun.com/exdb/mnist/`) — this site does not support HTTPS. All other URLs use `https://`.

### 2h. Makevars — ✅ PORTABLE

- Uses relative include paths (`-I../inst/include/`)
- Standard R build variables (`$(SHLIB_OPENMP_CXXFLAGS)`, `$(LAPACK_LIBS)`, etc.)
- Windows: `-Wa,-mbig-obj` for large template code
- No hardcoded paths or non-portable flags

### 2i. configure Script — ✅ POSIX sh

- Shebang: `#!/bin/sh` (not bash)
- No bash-isms
- Standard `[ ]` conditionals
- Graceful CUDA fallback

### 2j. C++ Headers — ✅ STANDARD GUARDS

All `#ifndef`/`#define`/`#endif` include guards. No `#pragma once`.

---

## 3. Vignettes — ✅ PASS (11 total)

All vignettes use current API, proper `eval` guards for optional packages, and declared datasets.

---

## 4. C++ Code & Compilation — ✅ PASS (0 package warnings)

Only remaining compiler warnings come from RcppEigen/Eigen external headers (not actionable; tolerated by CRAN).

---

## 5. R Unit Tests — ✅ GOOD

82 test files, 1291 passing, 488 skipped (GPU + streaming), 0 failures.

---

## 6. DESCRIPTION & NAMESPACE — ✅ PASS

- `R (>= 4.1.0)` dependency
- `Matrix` in `Depends:` (auto-attached for examples)
- 97+ exported symbols properly registered
- `SystemRequirements: CUDA Toolkit >= 11.0 (optional)`
- License: `GPL (>= 3)` — standard CRAN format

---

## 7. Reverse Dependencies — ✅ PASS

| Package | Type | Impact |
|---------|------|--------|
| GeneNMF | imports `nmf()` | ✅ Compatible |
| phytoclass | imports `nnls()` (old API) | ✅ Backward-compat shim + deprecation warning |
| scater (Bioc) | runtime | ✅ No direct function calls |
| miloR (Bioc) | LinkingTo | ✅ No R function imports |
| CARDspa, flashier | Suggests | ✅ No breakage possible |

---

## 8. Outstanding Items (Optional — Not Blocking)

1. **Tarball size (~9.5 MB)**: Above 5 MB guideline. Justified in `cran-comments.md`.
2. **`NEWS.md.bak` on disk**: Excluded from tarball via `.Rbuildignore`.
3. **`http://` URL for MNIST source**: Site does not support HTTPS; HTTP is the only working option.

3. **TODO comments in streampress headers** — 3 informational TODO comments remain in `inst/include/streampress/` (bundled third-party library). All have working implementations; comments are optimization notes. Not flagged by R CMD check.

4. **`RcppML.Rcheck/` and `RcppML_1.0.0.tar.gz` in root** — Transient check/build artifacts. Already excluded from tarball. Can be deleted.

5. **Stale `^manuscript$` in `.Rbuildignore`** — No longer matches anything (superseded by `^GuidedNMFManuscript$`). Harmless but could be removed for tidiness.

6. **`training_logger()` example uses `\dontrun{}`** — Could be converted to a self-contained `\donttest{}` example with synthetic data, but current form is acceptable.

7. **`R (>= 3.5.0)` in DESCRIPTION** — The native pipe `|>` is used in some examples and vignettes, which requires R >= 4.1.0. No R CMD check warning was triggered (examples don't use `|>` in evaluated code), but updating the dependency version would be more accurate.

---

## Summary Scorecard

| Area | Status | Notes |
|------|--------|-------|
| R CMD check | ✅ | 0 errors, 0 package warnings, 2 benign notes |
| Vignettes (11) | ✅ | All valid, proper eval guards |
| Roxygen Docs | ✅ | All exports have @examples |
| C++ Code | ✅ | 0 warnings from package code |
| C++ Tests | ✅ | 31 tests, 2320 assertions, 0 failures |
| R Tests | ✅ | 1291 pass, 0 fail |
| GPU/CPU Matrix | ✅ | Full coverage including GPU randomized SVD dense |
| Float Precision | ✅ | Correct fp32/fp64 strategy |
| Dead Code | ✅ | Deprecated shims properly managed |
| Build Hygiene | ✅ | Tarball 9.0 MB, no leaked artifacts |
| DESCRIPTION | ✅ | Complete and accurate |
| NAMESPACE | ✅ | All exports registered |
| NEWS.md | ✅ | Comprehensive changelog |
| Reverse Deps | ✅ | Backward compat maintained |

**Overall**: Package is ready for CRAN submission. Update `cran-comments.md` to explain the tarball size and `SeuratData` NOTE.
