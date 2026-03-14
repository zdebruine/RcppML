## Test environments

* Local: Red Hat Enterprise Linux 9.7, R 4.5.2, GCC 13.3
* GitHub Actions: ubuntu-latest (R release, R devel), windows-latest (R release), macos-latest (R release)

## R CMD check results

0 errors | 2 warnings | 1 note

* WARNING 1: `checkbashisms` script not installed. This is a system tool
  for checking shell scripts; not a package defect.

* WARNING 2: `qpdf` not installed. Used for PDF size reduction checks;
  all vignettes are HTML-only.

* NOTE 1 (CRAN incoming feasibility): Package tarball ~9 MB. This is
  due to:
  - Seven compressed benchmark datasets (6.5 MB installed) used in
    vignettes and examples covering NMF, SVD, clustering, and
    recommendation systems. The largest is `pbmc3k` (3.7 MB, stored
    as StreamPress-compressed raw bytes).
  - C++ template library headers (2.4 MB installed) for the Eigen-based
    NNLS solvers, required at compile time by LinkingTo dependents.
  - Eleven pre-built vignettes.
  - `SeuratData` is listed in Suggests but is not on CRAN; it is used
    only in optional `eval = FALSE` vignette examples for Seurat
    spatial transcriptomics integration.

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
* Built-in cross-validation (`nmf(..., test_fraction = ...)`)
* Multiple distribution-based losses (Gaussian, Generalized Poisson,
  Negative Binomial, Gamma, Inverse Gaussian, Tweedie)
* Optional GPU acceleration via CUDA (gracefully disabled when
  unavailable)
* SparsePress/StreamPress compressed sparse matrix I/O
* Factor networks for multi-layer and multi-modal factorization
* Backward-compatible shim for `nnls()` old positional API
