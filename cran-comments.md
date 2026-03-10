## Test environments

* Local: Red Hat Enterprise Linux 9.7, R 4.5.2, GCC 13.3
* GitHub Actions: ubuntu-latest (R release, R devel), windows-latest (R release), macos-latest (R release)

## R CMD check results

0 errors | 0 warnings | 2 notes

* NOTE 1 (CRAN incoming feasibility): Package tarball 5.6 MB (slightly
  above 5 MB guideline). This is due to:
  - Seven compressed benchmark datasets (3.4 MB installed) used in
    vignettes and examples covering NMF, SVD, clustering, and
    recommendation systems.
  - C++ template library headers (2.3 MB installed) for the Eigen-based
    NNLS solvers, required at compile time by LinkingTo dependents.
  - Twelve pre-built vignettes (2.6 MB installed).

* NOTE 2 (Unstated vignette dependencies): Seurat, SeuratData,
  stxBrain.SeuratData appear in `library()` calls in one vignette
  (`factor-net-applications.Rmd`). These chunks are `eval`-guarded
  by `requireNamespace()` checks and will never execute on CRAN.
  SeuratData and stxBrain.SeuratData are not CRAN packages; they are
  optional interactive demo dependencies for Seurat integration
  examples.

## Reverse dependencies

Checked all CRAN/Bioconductor packages that depend on, import, or link to
RcppML 0.3.7:

* **GeneNMF**: imports `nmf()` only — passes with no changes.
* **phytoclass**: imports `nnls()` using the old positional API
  (`nnls(A, b, cd_maxit, cd_tol)`). The new `nnls()` includes a
  backward-compatibility shim that accepts the old calling convention
  and emits a deprecation notice. phytoclass will continue to work;
  we will coordinate with the maintainer to update to the new API.
* **scater** (Bioconductor): runtime dependency; does not directly call
  any RcppML functions via `importFrom()`.
* **miloR** (Bioconductor): `LinkingTo` only; does not import R functions.
* **CARDspa**, **flashier**: `Suggests` only — no breakage possible.

## Major changes since 0.3.7

This is a major version update (0.3.7 → 1.0.0). Key changes:

* Complete C++ backend rewrite using Eigen template metaprogramming
* S4 `nmf` class replacing the previous list output
* Built-in cross-validation (`nmf(..., cv = TRUE)`)
* Multiple distribution-based losses (Gaussian, Poisson, Generalized
  Poisson, Negative Binomial, Gamma, Inverse Gaussian, Tweedie)
* Optional GPU acceleration via CUDA (gracefully disabled when
  unavailable)
* SparsePress/StreamPress compressed sparse matrix I/O
* Factor networks for multi-layer and multi-modal factorization
* Backward-compatible shims for `project()` → `nnls()` and
  `crossValidate()` → `nmf(..., cv = TRUE)`
