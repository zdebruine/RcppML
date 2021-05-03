// 3/30/2021 Zach DeBruine <zach.debruine@vai.org
// Please raise issues on github.com/zdebruine/sparselm/issues
//
// AMF, an RcppEigen template library for fast matrix factorization by alternating least squares

//[[Rcpp::plugins(openmp)]]
#include <omp.h>

//[[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#define EIGEN_NO_DEBUG

// structure returned in all AMF functions
template <typename T>
struct WDH {
  Eigen::Matrix<T, -1, -1> W;
  Eigen::Matrix<T, -1, 1> D;
  Eigen::Matrix<T, -1, -1> H;
  Eigen::Matrix<T, -1, 1> tol;
  Eigen::Matrix<T, -1, 1> mse;
};

template <typename T>
struct udv {
  Eigen::Matrix<T, -1, -1> u;
  Eigen::Matrix<T, -1, 1> d;
  Eigen::Matrix<T, -1, -1> v;
};



// trim a vector to length n
template<typename T>
Eigen::Matrix<T, -1, 1> trim(Eigen::Matrix<T, -1, 1> x, const unsigned int n) {
  Eigen::Matrix<T, -1, 1> y(n);
  for (unsigned int i = 0; i < n; ++i) y(i) = x(i);
  return y;
}

// get indices of values in "x" greater than zero
template<typename Derived>
Eigen::VectorXi find_gtz(const Eigen::MatrixBase<Derived>& x) {
  unsigned int n_gtz = 0;
  for (unsigned int i = 0; i < x.size(); ++i) if (x[i] > 0) ++n_gtz;
  Eigen::VectorXi gtz(n_gtz);
  unsigned int j = 0;
  for (unsigned int i = 0; i < x.size(); ++i) {
    if (x[i] > 0) {
      gtz[j] = i;
      ++j;
    }
  }
  return gtz;
}

template<typename T, typename Derived>
Eigen::Matrix<T, -1, 1> nnls(const Eigen::Matrix<T, -1, -1>& a, const typename Eigen::MatrixBase<Derived>& b,
                             const Eigen::LLT<Eigen::Matrix<T, -1, -1>, 1>& a_llt, const unsigned int cd_maxit = 0,
                             const T cd_tol = 1e-8) {

  Eigen::Matrix<T, -1, 1> x = a_llt.solve(b);
  // iterative feasible set reduction while unconstrained least squares solutions at feasible indices contain negative values
  while ((x.array() < 0).any()) {
    // get indices in "x" greater than zero (the "feasible set")
    Eigen::VectorXi gtz_ind = find_gtz(x);
    // subset "a" and "b" to those indices in the feasible set
    Eigen::Matrix<T, -1, 1> bsub(gtz_ind.size());
    for (unsigned int i = 0; i < gtz_ind.size(); ++i) bsub(i) = b(gtz_ind(i));
    Eigen::Matrix<T, -1, -1> asub = submat(a, gtz_ind, gtz_ind);
    // solve for those indices in "x"
    Eigen::Matrix<T, -1, 1> xsub = asub.llt().solve(bsub);
    x.setZero();
    for (unsigned int i = 0; i < gtz_ind.size(); ++i) x(gtz_ind(i)) = xsub(i);
  }
  // coordinate descent least squares solver
  if (cd_maxit > 0) {
    Eigen::Matrix<T, -1, 1> b0 = a * x;
    for (unsigned int i = 0; i < b0.size(); ++i) b0(i) -= b(i);
    T cd_tol_xi, cd_tol_it = 0;
    for (unsigned int it = 0; it < cd_maxit && cd_tol_it > cd_tol; ++it) {
      cd_tol_it = 0;
      for (unsigned int i = 0; i < x.size(); ++i) {
        T xi = x(i) - b0(i) / a(i, i);
        if (xi < 0) xi = 0;
        if (xi != x(i)) {
          // update gradient
          b0 += a.col(i) * (xi - x(i));
          // calculate tolerance for this value, update iteration tolerance if needed
          cd_tol_xi = 2 * std::abs(x(i) - xi) / (xi + x(i) + 1e-16);
          if (cd_tol_xi > cd_tol_it) cd_tol_it = cd_tol_xi;
          x(i) = xi;
        }
      }
    }
  }
  return x;
}

// solve the least squares equation A = WH for H given A and W
template<typename T>
Eigen::Matrix<T, -1, -1> project(const Eigen::SparseMatrix<T>& A, Eigen::Matrix<T, -1, -1>& w, const bool nonneg = true,
                                 const unsigned int cd_maxit = 0, const T cd_tol = 1e-8, const T L1 = 0, const unsigned int threads = 0) {

  // compute left-hand side of linear system, "a"
  Eigen::Matrix<T, -1, -1> a;
  if (w.cols() == A.rows()) a = w * w.transpose();
  else a = w.transpose() * w;

  Eigen::LLT<Eigen::Matrix<T, -1, -1>> a_llt = a.llt();

  Eigen::Matrix<T, -1, -1> h(w.cols(), A.cols());
  #pragma omp parallel for num_threads(threads) schedule(dynamic)
  for (unsigned int i = 0; i < A.cols(); ++i) {
    // compute right-hand side of linear system, "b", in-place in h
    h.col(i) = w * A.col(i);
    if (L1 != 0) for (unsigned int j = 0; j < h.rows(); ++j) h(j, i) -= L1;

    // solve least squares
    if (!nonneg) h.col(i) = a_llt.solve(h.col(i));
    else h.col(i) = nnls(a, h.col(i), a_llt, cd_maxit, cd_tol);
  }
  return h;
}

// ********************************************************************************
// SECTION 3: Loss of matrix factorization model
// ********************************************************************************

// sparse "A"
template<typename T>
T mse(const Eigen::SparseMatrix<T>& A, Eigen::Matrix<T, -1, -1> w, const Eigen::Matrix<T, -1, 1>& d,
      const Eigen::Matrix<T, -1, -1>& h) {

  const unsigned int threads = Eigen::nbThreads();
  if (w.rows() == h.rows()) w.transposeInPlace();

  // multiply w by diagonal
  #pragma omp parallel for num_threads(threads) schedule(dynamic)
  for (unsigned int i = 0; i < w.cols(); ++i)
    for (unsigned int j = 0; j < w.rows(); ++j)
      w(j, i) *= d(i);

  // calculate total loss with parallelization across samples
  Eigen::Matrix<T, -1, 1> losses(A.cols());
  losses.setZero();
  #pragma omp parallel for num_threads(threads) schedule(dynamic)
  for (unsigned int i = 0; i < A.cols(); ++i) {
    Eigen::Matrix<T, -1, 1> wh_i = w * h.col(i);
    for (typename Eigen::SparseMatrix<T>::InnerIterator iter(A, i); iter; ++iter)
      wh_i(iter.row()) -= iter.value();
    for (unsigned int j = 0; j < wh_i.size(); ++j)
      losses(i) += std::pow(wh_i(j), 2);
  }
  return losses.sum() / (A.cols() * A.rows());
}

// ********************************************************************************
// SECTION 4: Generalized alternating matrix factorization base algorithm
// ********************************************************************************

template <typename T>
WDH<T> amf(const Eigen::SparseMatrix<T>& A, unsigned int k, const T tol = 1e-3, const bool nonneg_w = true,
           const bool nonneg_h = true, const T L1_w = 0, const T L1_h = 0, const unsigned int maxit = 100,
           const bool calc_mse = false, bool lowmem = false, const bool diag = true, const unsigned int cd_maxit = 0,
           const T cd_tol = 1e-8, const bool verbose = false) {

  const unsigned int threads = Eigen::nbThreads();
  const unsigned int n_features = A.rows(), n_samples = A.cols();
  const bool symmetric = ((n_samples == n_features) && !A.isApprox(A.transpose()));
  if (symmetric) lowmem = false;
  Eigen::SparseMatrix<T> At;
  if (!lowmem && !symmetric) At = A.transpose();
  Eigen::Matrix<T, -1, -1> h(k, n_samples), w_it(k, n_features), a(k, k), w(k, n_features);
  Eigen::Matrix<T, -1, 1> d(k), tols(maxit), losses(maxit);
  Eigen::LLT<Eigen::Matrix<T, -1, -1>> a_llt;
  d.setOnes(); tols.setOnes(); losses.setZero();

  w.setRandom();
  w.cwiseAbs();

  if (verbose) Rprintf("\n%4s | %8s \n---------------\n", "iter", "tol"); // REMOVE THIS LINE FOR C++ USE ONLY

  unsigned int it = 0;
  for (; it < maxit; ++it) {
    // update h
    a = w * w.transpose();
    a_llt.compute(a);
    #pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (unsigned int i = 0; i < A.cols(); ++i) {
      h.col(i) = w * A.col(i);
      if (L1_h != 0) for (unsigned int j = 0; j < h.rows(); ++j) h(j, i) -= L1_h;

      // solve (non-negative) least squares problem
      if (!nonneg_h) h.col(i) = a_llt.solve(h.col(i));
      else h.col(i) = nnls(a, h.col(i), a_llt, cd_maxit, cd_tol);
    }
    // calculate diagonal as rowsums of h, and normalize rows in h to l2 norm
    if (diag || symmetric) {
      #pragma omp parallel for num_threads(threads) schedule(dynamic)
      for (unsigned int i = 0; i < k; ++i) {
        d[i] = h.row(i).sum();
        for (unsigned int j = 0; j < A.cols(); ++j) h(i, j) /= d(i);
      }
    }

    // update w
    a = h * h.transpose();
    a_llt.compute(a);
    if (lowmem) {
      w_it = w;
      w.setZero();
      // avoid explicit storage of A.transpose() by first calculating "b" in-place in "w" 
      //   on a non-transposed copy of "A" (slower than default method)
      #pragma omp parallel for num_threads(threads) schedule(dynamic)
      for (unsigned int i = 0; i < A.cols(); ++i)
        for (typename Eigen::SparseMatrix<T>::InnerIterator iter(A, i); iter; ++iter)
          w.col(iter.row()) += iter.value() * h.col(i);
    }
    #pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (unsigned int i = 0; i < A.rows(); ++i) {
      if (!symmetric && !lowmem) w_it.col(i) = w.col(i);
      // update w by calculating "b" in-place in "w" and working on a transposed copy of "A"
      if (!lowmem) {
        if (symmetric) w.col(i) = h * A.col(i);
        else w.col(i) = h * At.col(i);
      }
      if (L1_w != 0) for (unsigned int j = 0; j < k; ++j) w(j, i) -= L1_w;
      if (!nonneg_w) w.col(i) = a_llt.solve(w.col(i));
      else w.col(i) = nnls(a, w.col(i), a_llt, cd_maxit, cd_tol);
    }

    // reset diagonal and scale w
    if (diag || symmetric) {
      #pragma omp parallel for num_threads(threads) schedule(dynamic)
      for (unsigned int i = 0; i < k; ++i) {
        d[i] = w.row(i).sum();
        for (unsigned int j = 0; j < A.rows(); ++j) w(i, j) /= d(i);
      }
    }

    // calculate tolerance, assuming convergence is unlikely before the third iteration
    if (tol > 0) {
      tols(it) = symmetric ? cor(w, h) : cor(w, w_it);
      if (verbose) {
        Rcpp::checkUserInterrupt(); // REMOVE THIS LINE FOR C++ USE ONLY
        Rprintf("%4d | %8.2e\n", it + 1, tols(it)); // REMOVE THIS LINE FOR C++ USE ONLY
      }
    }
    if (calc_mse) losses(it) = mse(A, w, d, h);
    if (tols(it) < tol) {
      tols = trim(tols, it);
      losses = trim(losses, it);
      break;
    }
  }
  if (diag) {
    std::vector<int> indx = sort_index(d);
    w = reorder_rows(w, indx);
    d = reorder(d, indx);
    h = reorder_rows(h, indx);
  }
  return WDH<T> {w.transpose(), d, h, tols, losses};
}

// ********************************************************************************
// SECTION 5: Rank-2 alternating matrix factorization base algorithm
// ********************************************************************************

template <typename T>
WDH<T> amf2(const Eigen::SparseMatrix<T>& A, const T tol = 1e-3, const bool nonneg_w = true,
            const bool nonneg_h = true, const unsigned int maxit = 100, const bool calc_mse = false,
            const bool diag = false, const bool verbose = false) {

  const unsigned int threads = Eigen::nbThreads();
  const unsigned int n_features = A.rows(), n_samples = A.cols();
  const bool symmetric = ((n_samples == n_features) && !A.isApprox(A.transpose()));
  Eigen::Matrix<T, -1, -1> h(2, n_samples), h_it(2, n_samples), a(2, 2), w(2, n_features), wb(2, n_features);
  Eigen::Matrix<T, -1, 1> d(2), tols(maxit), losses(maxit);
  d.setOnes(); tols.setOnes(); losses.setZero();

  h.setRandom();
  h = h.cwiseAbs();

  if (verbose) Rprintf("\n%4s | %8s \n---------------\n", "iter", "tol"); // REMOVE THIS LINE FOR C++ USE ONLY

  for (unsigned int it = 0; it < maxit; ++it) {
    // update w
    a = h * h.transpose();
    T denom = a(0, 0) * a(1, 1) - a(0, 1) * a(0, 1);
    wb.setZero();
    #pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (unsigned int i = 0; i < n_samples; ++i)
      for (typename Eigen::SparseMatrix<T>::InnerIterator iter(A, i); iter; ++iter)
        wb.col(iter.row()) += iter.value() * h.col(i);
    #pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (unsigned int i = 0; i < n_features; ++i) {
      if (nonneg_w) {
        T a12b2 = a(0, 1) * wb(1, i);
        T a2b1 = a(1, 1) * wb(0, i);
        if (a2b1 < a12b2) {
          w(0, i) = 0;
          w(1, i) = wb(1, i) / a(1, 1);
        } else {
          T a12b1 = a(0, 1) * wb(0, i);
          T a1b2 = a(0, 0) * wb(1, i);
          if (a1b2 < a12b1) {
            w(0, i) = wb(0, i) / a(0, 0);
            w(1, i) = 0;
          } else {
            w(0, i) = (a2b1 - a12b2) / denom;
            w(1, i) = (a1b2 - a12b1) / denom;
          }
        }
      } else {
        w(0, i) = (a(1, 1) * wb(0, i) - a(0, 1) * wb(1, i)) / denom;
        w(1, i) = (a(0, 0) * wb(1, i) - a(0, 1) * wb(0, i)) / denom;
      }
    }

    // reset diagonal and scale w
    if (diag || symmetric) {
      d(0) = w.row(0).sum();
      d(1) = w.row(1).sum();
      for (unsigned int j = 0; j < n_features; ++j) {
        w(0, j) /= d(0);
        w(1, j) /= d(1);
      }
    }

    // update h
    a = w * w.transpose();
    denom = a(0, 0) * a(1, 1) - a(0, 1) * a(0, 1);
    #pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (unsigned int i = 0; i < n_samples; ++i) {
      h_it.col(i) = h.col(i);
      T b0 = w.row(0) * A.col(i);
      T b1 = w.row(1) * A.col(i);
      if (nonneg_h) {
        T a12b2 = a(0, 1) * b1;
        T a2b1 = a(1, 1) * b0;
        if (a2b1 < a12b2) {
          h(0, i) = 0;
          h(1, i) = b1 / a(1, 1);
        } else {
          T a12b1 = a(0, 1) * b0;
          T a1b2 = a(0, 0) * b1;
          if (a1b2 < a12b1) {
            h(0, i) = b0 / a(0, 0);
            h(1, i) = 0;
          } else {
            h(0, i) = (a2b1 - a12b2) / denom;
            h(1, i) = (a1b2 - a12b1) / denom;
          }
        }
      } else {
        h(0, i) = (a(1, 1) * b0 - a(0, 1) * b1) / denom;
        h(1, i) = (a(0, 0) * b1 - a(0, 1) * b0) / denom;
      }
    }
    // calculate diagonal as rowsums of h, and normalize rows in h to l2 norm
    if (diag || symmetric) {
      d(0) = h.row(0).sum();
      d(1) = h.row(1).sum();
      for (unsigned int j = 0; j < n_samples; ++j) {
        h(0, j) /= d(0);
        h(1, j) /= d(1);
      }
    }

    // calculate tolerance, assuming convergence is unlikely before the third iteration
    if (tol > 0) {
      tols(it) = symmetric ? cor(w, h) : cor(h, h_it);
      if (verbose) {
        Rcpp::checkUserInterrupt(); // REMOVE THIS LINE FOR C++ USE ONLY
        Rprintf("%4d | %8.2e\n", it + 1, tols(it)); // REMOVE THIS LINE FOR C++ USE ONLY
      }
    }
    if (calc_mse) losses(it) = mse(A, w, d, h);
    if (tols(it) < tol) {
      tols = trim(tols, it);
      losses = trim(losses, it);
      break;
    }
  }
  if (diag && d[0] < d[1]) {
    w.row(0).swap(w.row(1));
    h.row(0).swap(h.row(1));
    T d_temp = d(1);
    d(1) = d(0);
    d(0) = d_temp;
  }
  return WDH<T> {w.transpose(), d, h, tols, losses};
}

// return second singular vector using an alternating matrix factorization
template <typename T>
Eigen::Matrix<T, -1, 1> svd2(const Eigen::SparseMatrix<T>& A, const T tol = 1e-3, const unsigned int maxit = 100,
                             const bool return_v = true) {

  WDH<T> model = amf2(A, tol, false, false, maxit, false, true, false);
  if (return_v) return model.H.row(0) - model.H.row(1);
  else return model.W.row(0) - model.W.row(1);
}

// ********************************************************************************
// SECTION 6: Rank-1 alternating matrix factorization base algorithm
// ********************************************************************************

template <typename T>
WDH<T> amf1(const Eigen::SparseMatrix<T>& A, const T tol = 1e-3, const unsigned int maxit = 100, const bool diag = false,
            const bool verbose = false) {

  const unsigned int threads = Eigen::nbThreads();
  const unsigned int n_features = A.rows(), n_samples = A.cols();
  const bool symmetric = ((n_samples == n_features) && !A.isApprox(A.transpose()));
  Eigen::Matrix<T, -1, 1> h(n_samples), h_it(n_samples), w(n_features);
  T d = 1;
  Eigen::Matrix<T, -1, 1> tols(maxit), losses(maxit);
  tols.setOnes(); losses.setZero();
  T a = 0;

  h.setRandom();
  h = h.cwiseAbs();

  if (verbose) Rprintf("\n%4s | %8s \n---------------\n", "iter", "tol"); // REMOVE THIS LINE FOR C++ USE ONLY

  for (unsigned int it = 0; it < maxit; ++it) {
    // update w
    w.setZero();
    a = 0;
    for (unsigned int i = 0; i < n_samples; ++i) a += h(i) * h(i);
    #pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (unsigned int sample = 0; sample < n_samples; ++sample)
      for (typename Eigen::SparseMatrix<T>::InnerIterator iter(A, sample); iter; ++iter)
        w(iter.row()) += iter.value() * h(sample);
    for (unsigned int i = 0; i < n_features; ++i) w(i) /= a;

    if (diag) {
      d = w.sum();
      for (unsigned int i = 0; i < n_features; ++i) w(i) /= d;
    }

    // update h
    h_it = h;
    h.setZero();
    a = 0;
    for (unsigned int i = 0; i < n_features; ++i) a += w(i) * w(i);
    #pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (unsigned int sample = 0; sample < n_samples; ++sample) {
      for (typename Eigen::SparseMatrix<T>::InnerIterator iter(A, sample); iter; ++iter)
        h(sample) += iter.value() * w(iter.row());
      h(sample) /= a;
    }
    if (diag) {
      d = h.sum();
      for (unsigned int i = 0; i < n_samples; ++i) h(i) /= d;
    }

    // calculate tolerance, assuming convergence is unlikely before the third iteration
    if (tol > 0) {
      tols(it) = symmetric ? cor(w, h) : cor(h, h_it);
      if (verbose) {
        Rcpp::checkUserInterrupt(); // REMOVE THIS LINE FOR C++ USE ONLY
        Rprintf("%4d | %8.2e\n", it + 1, tols(it)); // REMOVE THIS LINE FOR C++ USE ONLY
      }
    }
    if (tols(it) < tol) {
      tols = trim(tols, it);
      losses = trim(losses, it);
      break;
    }
  }
  Eigen::Matrix<T, -1, 1> dmat(1);
  dmat(0) = d;
  return WDH<T> {w.transpose(), dmat, h, tols, losses};
}

template <typename T>
Eigen::Matrix<T, -1, 1> svd1(const Eigen::SparseMatrix<T>& A, const T tol = 1e-3, const unsigned int maxit = 100,
                             const bool return_v = true) {

  const unsigned int threads = Eigen::nbThreads();
  const unsigned int n_features = A.rows(), n_samples = A.cols();
  Eigen::Matrix<T, -1, 1> h(n_samples), h_it(n_samples), w(n_features);
  h.setRandom();
  h = h.cwiseAbs();
  for (unsigned int it = 0; it < maxit; ++it) {
    // update w
    w.setZero();
    T a = 0;
    for (unsigned int i = 0; i < n_samples; ++i) a += h(i) * h(i);
    #pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (unsigned int sample = 0; sample < n_samples; ++sample)
      for (typename Eigen::SparseMatrix<T>::InnerIterator iter(A, sample); iter; ++iter)
        w(iter.row()) += iter.value() * h(sample);
    for (unsigned int i = 0; i < n_features; ++i) w(i) /= a;
    // update h
    h_it = h;
    h.setZero();
    a = 0;
    for (unsigned int i = 0; i < n_features; ++i) a += w(i) * w(i);
    #pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (unsigned int sample = 0; sample < n_samples; ++sample) {
      for (typename Eigen::SparseMatrix<T>::InnerIterator iter(A, sample); iter; ++iter)
        h(sample) += iter.value() * w(iter.row());
      h(sample) /= a;
    }
    if (it > 0 && cor(h, h_it) < tol) break;
  }
  return return_v ? h : w;
}

template <typename T>
udv<T> als_svd(const Eigen::SparseMatrix<T>& A, const unsigned int k, const T tol = 1e-5,
               const unsigned int maxit = 100, const bool verbose = false) {

  const unsigned int threads = Eigen::nbThreads();
  const unsigned int m = A.rows(), n = A.cols();
  Eigen::Matrix<T, -1, 1> v_it(n), v_curr(n), d(k), a(k);
  Eigen::Matrix<T, -1, -1> u(m, k), v(n, k);
  u.setZero();
  d.setZero();
  v.setRandom();
  v = v.cwiseAbs();
  T tol_it = 1, a_rank, u_new_sum, v_new_sum;

  for (unsigned int rank = 0; rank < k; ++rank) {
    // alternating updates of u/v vector pair to be added until convergence is satisfied
    for (unsigned int it = 0; it < maxit; ++it) {
      // update u_rank
      a.setZero();
      for (unsigned int j = 0; j <= rank; ++j)
        for (unsigned int i = 0; i < n; ++i)
          a(j) += v(i, j) * v(i, rank);
      u.col(rank).setZero();
      #pragma omp parallel for num_threads(threads) schedule(dynamic)
      for (unsigned int i = 0; i < n; ++i)
        for (typename Eigen::SparseMatrix<T>::InnerIterator iter(A, i); iter; ++iter)
          u(iter.row(), rank) += iter.value() * v(i, rank);
      #pragma omp parallel for num_threads(threads) schedule(dynamic)
      for (unsigned int i = 0; i < m; ++i) {
        T au = 0;
        for (unsigned int j = 0; j < rank; ++j) au += u(i, j) * a(j);
        u(i, rank) = (u(i, rank) - au) / a(rank);
      }
      v_it = v.col(rank);
      // update v_new
      a.setZero();
      for (unsigned int j = 0; j <= rank; ++j)
        for (unsigned int i = 0; i < m; ++i)
          a(j) += u(i, j) * u(i, rank);
      v.col(rank).setZero();
      #pragma omp parallel for num_threads(threads) schedule(dynamic)
      for (unsigned int i = 0; i < n; ++i) {
        for (typename Eigen::SparseMatrix<T>::InnerIterator iter(A, i); iter; ++iter)
          v(i, rank) += iter.value() * u(iter.row(), rank);
        T av = 0;
        for (unsigned int j = 0; j < rank; ++j) av += v(i, j) * a(j);
        v(i, rank) = (v(i, rank) - av) / a(rank);
      }
      v_curr = v.col(rank);
      tol_it = cor(v_curr, v_it);
      if (verbose) { // ************************** BEGIN REMOVE THESE LINES FOR C++ USE ONLY ***
        Rcpp::checkUserInterrupt();
        Rprintf("%2d | %4d | %8.2e\n", rank + 1, it + 1, tol_it);
      } // ***************************************** END REMOVE THESE LINES FOR C++ USE ONLY ***
      if (tol_it < tol) break;
    }
  }
  return udv<T> {u, d, v};
}

//[[Rcpp::export]]
Rcpp::List ltsvd(const Eigen::SparseMatrix<double>& A, const unsigned int k, const double tol = 1e-5,
                 const unsigned int maxit = 100, const bool verbose = true) {

  udv<double> model = als_svd(A, k, tol, maxit, verbose);

  return(Rcpp::List::create(
    Rcpp::Named("u") = model.u,
    Rcpp::Named("d") = model.d,
    Rcpp::Named("v") = model.v));
}

//[[Rcpp::export]]
Rcpp::List Rcpp_amf_double(const Eigen::SparseMatrix<double>& A, const unsigned int k, const unsigned int seed = 0,
                           const double tol = 1e-3, const bool nonneg_w = true, const bool nonneg_h = true, const double L1_w = 0,
                           const double L1_h = 0, const unsigned int maxit = 25, const unsigned int threads = 0,
                           const bool verbose = true, const bool calc_mse = false, const bool lowmem = false, const bool diag = true,
                           const unsigned int cd_maxit = 0, const double cd_tol = 1e-8) {

  if (seed == 0) srand((unsigned int)time(0));
  else srand(seed);

  Eigen::setNbThreads(threads);

  WDH<double> model;
  if (k == 1) model = amf1(A, tol, maxit, diag, verbose);
  else if (k == 2) model = amf2(A, tol, nonneg_w, nonneg_h, maxit, calc_mse, diag, verbose);
  else model = amf(A, k, tol, nonneg_w, nonneg_h, L1_w, L1_h, maxit, calc_mse, lowmem, diag, cd_maxit, cd_tol, verbose);
  return(Rcpp::List::create(
    Rcpp::Named("w") = model.W,
    Rcpp::Named("d") = model.D,
    Rcpp::Named("h") = model.H,
    Rcpp::Named("tol") = model.tol,
    Rcpp::Named("mse") = model.mse));
}

//[[Rcpp::export]]
Rcpp::List Rcpp_amf_float(const Eigen::SparseMatrix<float>& A, const unsigned int k, const unsigned int seed = 0,
                          const float tol = 1e-3, const bool nonneg_w = true, const bool nonneg_h = true,
                          const float L1_w = 0, const float L1_h = 0, const unsigned int maxit = 100, const unsigned int threads = 0,
                          const bool verbose = false, const bool calc_mse = false, const bool lowmem = false, const bool diag = true,
                          const unsigned int cd_maxit = 0, const float cd_tol = 1e-8) {

  if (seed == 0) srand((unsigned int)time(0));
  else srand(seed);

  Eigen::setNbThreads(threads);

  WDH<float> model;
  if (k == 1) model = amf1(A, tol, maxit, diag, verbose);
  else if (k == 2) model = amf2(A, tol, nonneg_w, nonneg_h, maxit, calc_mse, diag, verbose);
  else model = amf(A, k, tol, nonneg_w, nonneg_h, L1_w, L1_h, maxit, calc_mse, lowmem, diag, cd_maxit, cd_tol, verbose);

  return(Rcpp::List::create(
    Rcpp::Named("w") = model.W,
    Rcpp::Named("d") = model.D,
    Rcpp::Named("h") = model.H,
    Rcpp::Named("tol") = model.tol,
    Rcpp::Named("mse") = model.mse));
}

//[[Rcpp::export]]
Eigen::VectorXd Rcpp_svd1(const Eigen::SparseMatrix<double>& A, const double tol = 1e-3, const unsigned int maxit = 100,
                          const bool return_v = true, const unsigned int threads = 0) {

  Eigen::setNbThreads(threads);
  return svd1(A, tol, maxit, return_v);
}

//[[Rcpp::export]]
Eigen::VectorXd Rcpp_nnls_double(const Eigen::MatrixXd& a, Eigen::MatrixXd b, const unsigned int cd_maxit = 0, const double cd_tol = 1e-8) {

  Eigen::LLT<Eigen::MatrixXd> a_llt = a.llt();
  for (unsigned int i = 0; i < b.cols(); ++i)
    b.col(i) = nnls(a, b.col(i), a.llt(), cd_maxit, cd_tol);
  return b;
}

//[[Rcpp::export]]
Eigen::VectorXf Rcpp_nnls_float(const Eigen::MatrixXf& a, Eigen::MatrixXf b, const unsigned int cd_maxit = 0, const float cd_tol = 1e-8) {
  Eigen::LLT<Eigen::MatrixXf> a_llt = a.llt();
  for (unsigned int i = 0; i < b.cols(); ++i)
    b.col(i) = nnls(a, b.col(i), a.llt(), cd_maxit, cd_tol);
  return b;
}

//[[Rcpp::export]]
Eigen::MatrixXd Rcpp_project_double(const Eigen::SparseMatrix<double>& A, Eigen::MatrixXd& w, const bool nonneg = true,
                                    const unsigned int cd_maxit = 0, const double cd_tol = 1e-8, const double L1 = 0, const unsigned int threads = 0) {
  return project(A, w, nonneg, cd_maxit, cd_tol, L1, threads);
}

//[[Rcpp::export]]
Eigen::MatrixXf Rcpp_project_float(const Eigen::SparseMatrix<float>& A, Eigen::MatrixXf& w, const bool nonneg = true,
                                   const unsigned int cd_maxit = 0, const float cd_tol = 1e-8, const float L1 = 0, const unsigned int threads = 0) {
  return project(A, w, nonneg, cd_maxit, cd_tol, L1, threads);
}

//[[Rcpp::export]]
double Rcpp_mse_double(const Eigen::SparseMatrix<double>& A, const Eigen::MatrixXd& w, const Eigen::VectorXd& d, const Eigen::MatrixXd& h) {
  return mse(A, w, d, h);
}

//[[Rcpp::export]]
float Rcpp_mse_float(const Eigen::SparseMatrix<float>& A, const Eigen::MatrixXf& w, const Eigen::VectorXf& d, const Eigen::MatrixXf& h) {
  return mse(A, w, d, h);
}