// Prototype: parallel CSC cast (double->float) and parallel CSC transpose.
// Standalone benchmark against Eigen's serial cast<float>() and transpose().
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(openmp)]]
#include <RcppEigen.h>
#include <chrono>
#include <vector>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef Eigen::SparseMatrix<float> SpF;

// Parallel cast: copy structure + convert values, first-touch in parallel.
template <typename InScalar>
inline SpF parallel_cast(int m, int n, const int* p, const int* i,
                         const InScalar* x, int threads) {
  const int nnz = p[n];
  SpF out(m, n);
  out.resizeNonZeros(nnz);
  int* op = out.outerIndexPtr();
  int* oi = out.innerIndexPtr();
  float* ox = out.valuePtr();
  #pragma omp parallel for schedule(static) num_threads(threads)
  for (int j = 0; j <= n; ++j) op[j] = p[j];
  #pragma omp parallel for schedule(static) num_threads(threads)
  for (int k = 0; k < nnz; ++k) { oi[k] = i[k]; ox[k] = static_cast<float>(x[k]); }
  return out;
}

// Parallel transpose of CSC(A) [m x n] into CSC(A^T) [n x m].
// Rows of A are split into contiguous ranges of roughly equal nnz, one per
// thread. Each thread scans the column structure and writes only the entries
// whose row falls in its range, so writes never overlap and no atomics are
// needed. Column indices within each output column stay sorted.
template <typename InScalar>
inline SpF parallel_transpose(int m, int n, const int* p, const int* i,
                              const InScalar* x, int threads) {
  const int nnz = p[n];
  SpF out(n, m);
  out.resizeNonZeros(nnz);
  int* op = out.outerIndexPtr();   // size m + 1
  int* oi = out.innerIndexPtr();
  float* ox = out.valuePtr();

  // 1. Row counts with per-thread histograms.
  std::vector<int> cnt(static_cast<size_t>(m) + 1, 0);
  #pragma omp parallel num_threads(threads)
  {
    std::vector<int> local(m, 0);
    #pragma omp for schedule(static) nowait
    for (int k = 0; k < nnz; ++k) ++local[i[k]];
    #pragma omp critical
    for (int r = 0; r < m; ++r) cnt[r + 1] += local[r];
  }
  op[0] = 0;
  for (int r = 0; r < m; ++r) op[r + 1] = op[r] + cnt[r + 1];

  // 2. Split rows into ranges of roughly equal nnz.
  std::vector<int> bound(threads + 1, m);
  bound[0] = 0;
  for (int t = 1; t < threads; ++t) {
    const long long target = static_cast<long long>(nnz) * t / threads;
    bound[t] = static_cast<int>(std::lower_bound(op, op + m + 1, static_cast<int>(target)) - op);
    if (bound[t] > m) bound[t] = m;
    if (bound[t] < bound[t - 1]) bound[t] = bound[t - 1];
  }

  // 3. Each thread fills its own row range.
  #pragma omp parallel num_threads(threads)
  {
#ifdef _OPENMP
    const int t = omp_get_thread_num();
#else
    const int t = 0;
#endif
    const int lo = bound[t], hi = bound[t + 1];
    if (hi > lo) {
      std::vector<int> cur(op + lo, op + hi);
      for (int j = 0; j < n; ++j) {
        // rows within a column are sorted: binary-search the first row >= lo
        const int* b = std::lower_bound(i + p[j], i + p[j + 1], lo);
        for (int k = static_cast<int>(b - i); k < p[j + 1]; ++k) {
          const int r = i[k];
          if (r >= hi) break;
          const int dst = cur[r - lo]++;
          oi[dst] = j;
          ox[dst] = static_cast<float>(x[k]);
        }
      }
    }
  }
  return out;
}

static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// [[Rcpp::export]]
Rcpp::List bench_setup(const Eigen::Map<Eigen::SparseMatrix<double> > A, int threads) {
  const int m = A.rows(), n = A.cols();
  double t0 = now();
  SpF Af = A.cast<float>();
  double t1 = now();
  SpF At = Af.transpose();
  At.makeCompressed();
  double t2 = now();
  SpF Af2 = parallel_cast<double>(m, n, A.outerIndexPtr(), A.innerIndexPtr(), A.valuePtr(), threads);
  double t3 = now();
  SpF At2 = parallel_transpose<float>(m, n, Af2.outerIndexPtr(), Af2.innerIndexPtr(), Af2.valuePtr(), threads);
  double t4 = now();

  bool same = At.nonZeros() == At2.nonZeros() && Af.nonZeros() == Af2.nonZeros();
  for (int j = 0; same && j <= At.cols(); ++j) same = At.outerIndexPtr()[j] == At2.outerIndexPtr()[j];
  for (int k = 0; same && k < At.nonZeros(); ++k)
    same = At.innerIndexPtr()[k] == At2.innerIndexPtr()[k] && At.valuePtr()[k] == At2.valuePtr()[k];
  for (int k = 0; same && k < Af.nonZeros(); ++k)
    same = Af.innerIndexPtr()[k] == Af2.innerIndexPtr()[k] && Af.valuePtr()[k] == Af2.valuePtr()[k];

  return Rcpp::List::create(
    Rcpp::Named("eigen_cast") = t1 - t0, Rcpp::Named("eigen_transpose") = t2 - t1,
    Rcpp::Named("par_cast") = t3 - t2, Rcpp::Named("par_transpose") = t4 - t3,
    Rcpp::Named("identical") = same);
}
