## Submission

This is a major update of RcppML (0.3.7.1 on CRAN to 1.0.0).

## Test environments

<!-- Fill in from the final check runs before submitting. -->

* Local: Red Hat Enterprise Linux 9, R 4.5.3, GCC 13
* GitHub Actions: ubuntu-latest (R release, devel, oldrel-1),
  windows-latest (R release), macos-latest (R release)

## R CMD check results

0 errors | 0 warnings | 1 note (R 4.5.3, GCC 13, `--as-cran`, two threads;
about four minutes in total, tests 16 s).

* NOTE (CRAN incoming feasibility): the tarball is about 7.2 MB. Seven
  compressed example datasets account for 5.0 MB; the examples, tests and
  vignettes use them to demonstrate NMF, SVD, clustering and recommendation
  on real data, and they are stored at maximum (xz) compression. The built
  vignettes with their figures account for most of the rest. The header-only
  C++ library in `inst/include`, which packages that link to RcppML compile
  against, is 2.6 MB installed.

## Notes for the reviewers

* Cores. Examples, tests and vignettes use at most two threads: tests and
  vignettes set `options(RcppML.threads = 2)`, examples pass `threads = 2`,
  and the package caps its thread count at 2 whenever
  `_R_CHECK_LIMIT_CORES_` is set and at `OMP_THREAD_LIMIT` when that is set.
* Check time. The long-running tests call `skip_on_cran()`; the tests that
  run on CRAN are a fast subset that includes the reference-correctness
  tests. Vignette chunks are evaluated only when `NOT_CRAN` is `"true"`; the
  vignettes in the tarball were built that way and carry their full output.
* The package has no `configure` script any more and requires C++17
  (`CXX_STD = CXX17`).
* All suggested packages are on CRAN and are used conditionally.
* GPU support is optional and is not built during installation. Without the
  separately built library every function runs on the CPU.
* The package exports `svd()`, which masks `base::svd()` when RcppML is
  attached. It returns a truncated, optionally constrained decomposition as
  an S4 object; `base::svd()` remains available by its qualified name, and
  the help page states this.
* `inst/COPYRIGHTS` lists the bundled BSD-licensed Hungarian algorithm
  (`inst/include/RcppHungarian.h`); its authors are in `Authors@R`.

## Reverse dependencies

<!-- Re-run the reverse dependency checks against the final tarball and
     replace this section with the results. -->

Changes in 1.0.0 that affect reverse dependencies, and how they are handled:

* `nmf()` returns an S4 object instead of a list. `model$w`, `model$d` and
  `model$h` keep working through a `$` method.
* `nnls(a, b, ...)`: the 0.3.7 calling convention (including positional
  `cd_maxit` and `cd_tol`) is still accepted and returns the same solution,
  with a deprecation warning.
* `project()` is kept as a deprecated wrapper. `nmf(mask_zeros = )` and
  `nmf(A = )` are accepted with a deprecation warning.
* The C++ headers were reorganized (`RcppML.h` now forwards to the FactorNet
  headers). Packages that use `LinkingTo: RcppML` need to be checked
  individually.
