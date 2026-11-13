# RcppML

R package for non-negative matrix factorization, SVD/PCA and divisive clustering on large sparse and dense matrices. GitHub: `zdebruine/RcppML`. CRAN carries 0.3.7.1; `main` is 1.0.0 (a rewrite, not yet accepted by CRAN).

## Layout

- `R/` is the user API. `nmf_thin.R` holds `nmf()`; advanced options arrive through `...` and are parsed in `parse_dots.R`. `svd.R` holds `svd()` and `pca()` (the package masks `base::svd`). `solve.R` is `nnls()`. `factor_net.R` and `factor_methods.R` build multi-layer networks. `gpu_backend.R` finds and loads the optional GPU library.
- `src/RcppFunctions_*.cpp` are thin Rcpp bridges. `inst/gpu/*.cu` and `inst/gpu/Makefile.gpu` build the optional `RcppML_gpu.so` through `install_gpu()`.
- `inst/include/FactorNet/` is the header-only engine (about 55k lines): `nmf/` (fit paths), `primitives/cpu` and `primitives/gpu` (NNLS solvers, Gram and RHS), `features/` (penalties), `svd/` (five methods), `clustering/`, `graph/` (factor networks), `math/loss.hpp` (distributions). `inst/include/FactorNet/GUIDE.md` describes the design but is out of date in places.
- `inst/include/streampress/` is the `.spz` compressed sparse format.
- `tests/testthat/` has about 950 tests. The `test_ref_*.R` files compare against reference math written in plain R; the older files mostly compare the package with itself. Most older files are skipped on CRAN to fit the time limit.
- `docs/` is a committed pkgdown site. `man/`, `NAMESPACE` and `src/RcppExports.*` are generated.

## Fit paths in `nmf()`

One call can take any of these routes, and each applies a different subset of options. Check every route when changing a penalty, loss or solver.

- Sparse Gaussian "fused" path, dense or target "standard" path, masked path, IRLS path for non-Gaussian losses, zero-inflated path, cross-validation path (`fit_cv.hpp`), projective and symmetric variants, chunked/streaming path, GPU sparse, GPU dense, GPU cross-validation, and the zero-copy GPU path.
- All fits run in single precision.
- `solver = "auto"` picks `"cholesky"` for Gaussian CPU fits at every rank: a Cholesky solve, active-set refinement, then a coordinate-descent polish to the KKT conditions, so it is exact. `"cd"` is used for non-Gaussian and robust losses and on GPU. The rule lives in `.auto_solver()` in `R/nmf_validation.R`.
- The GPU runs only Gaussian-loss NMF with L1, L2 and non-negativity, and unconstrained SVD. Everything else is routed to CPU by the gateway in `inst/include/FactorNet/nmf/fit.hpp` with a message. Widen that list only after the GPU kernel passes the same reference checks as CPU.
- Options that a path cannot honor raise an error. Do not reintroduce silent drops.

## Build and test

There is no `configure` script any more; `R CMD INSTALL .` works from a plain checkout. GPU sources live in `inst/gpu/` and are built on demand with `RcppML::install_gpu()`; `gpu_status()` reports what is missing.

After changing roxygen comments or `// [[Rcpp::export]]`: run `devtools::document()`, then `bash tools/fix_rcpp_info_bug.sh` (works around a spurious `info` registration that `Rcpp::compileAttributes()` emits).

Heavy work runs on Clipper through the `clipper` skill, never on the laptop. Conventions used so far:

- Source checkout: `$WORK/src/RcppML/main` (cloned from GitHub inside a job).
- Job scripts and helpers: `$WORK/jobs/rcppml/` (`env.sh` loads `r/4.5.3`; GPU jobs add `cuda/12.9`).
- R libraries: `$WORK/rcppml/rlib/4.5.2` (current build and dependencies), `rlib/gpu-4.5.2` (build with the GPU library installed), `rlib/old037` and `rlib/old056` (earlier releases for regression comparisons).
- Results: `$WORK/rcppml/out/`.
- `short-bigmem` usually starts at once when `short-cpu` is queued. A full CPU test run takes about three minutes on 16 cores; the GPU build takes about three minutes.
- The working tree is pushed to Clipper without git by `bash tools/correctness/sync_build.sh` (builds into `rlib/fix`, regenerates `NAMESPACE`, `man/` and `RcppExports`, leaves them in `$WORK/rcppml/out/generated` to copy back). `bash tools/correctness/sync_verify.sh` then runs the test suite and the sweep against that build and prints a compact report. `bench.sbatch "lib:label ..."` times any installed library; `gpufix.sbatch` builds and tests the GPU library; `checkfix.sbatch` runs `R CMD check --as-cran` under the two-core limit.
- Windows checkouts have CRLF line endings; the Clipper scripts strip them before building. Edit files with tools that preserve the existing line endings.
- `sweep.R <section> <cpu|gpu>` in `tools/correctness/` is an independent correctness harness that compares the package against reference math written in plain R (solver exactness, each penalty on each fit path, each distribution, cross-validation, SVD/PCA against `base::svd` and `prcomp`, clustering and utilities). Use it, and extend it, before trusting a numerical change.

## Working rules

- Verify numerical claims against an independent reference, not against another path of the same package.
- CRAN limits examples, tests and vignettes to two cores; the package default is all cores.
- Keep pull requests narrow. Do not edit generated files by hand.
