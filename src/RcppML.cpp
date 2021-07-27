// This file is part of RcppML, an Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
// github.com/zdebruine/RcppML

#define EIGEN_NO_DEBUG
#define EIGEN_INITIALIZE_MATRICES_BY_ZERO

//[[Rcpp::plugins(openmp)]]
#ifdef _OPENMP
#include <omp.h>
#endif

//[[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

// CONTENTS
// section 1: helper subroutines
// section 2: non-negative least squares
// section 3: Rcpp dgCMatrix matrix class
// section 4: projecting linear factor models
// section 5: mean squared error loss of a factor model
// section 6: matrix factorization by alternating least squares
// section 7: Rcpp wrapper functions

// ********************************************************************************
// SECTION 1: Helper subroutines
// ********************************************************************************

// these functions for matrix subsetting are documented here:
// http://eigen.tuxfamily.org/dox-devel/TopicCustomizing_NullaryExpr.html#title1
// official support will likely appear in Eigen 4.0, this is a patch in the meantime
template<class ArgType, class RowIndexType, class ColIndexType>
class indexing_functor {
  const ArgType& m_arg;
  const RowIndexType& m_rowIndices;
  const ColIndexType& m_colIndices;
public:
  typedef Eigen::Matrix<typename ArgType::Scalar,
    RowIndexType::SizeAtCompileTime,
    ColIndexType::SizeAtCompileTime,
    ArgType::Flags& Eigen::RowMajorBit ? Eigen::RowMajor : Eigen::ColMajor,
    RowIndexType::MaxSizeAtCompileTime,
    ColIndexType::MaxSizeAtCompileTime> MatrixType;

  indexing_functor(const ArgType& arg, const RowIndexType& row_indices, const ColIndexType& col_indices)
    : m_arg(arg), m_rowIndices(row_indices), m_colIndices(col_indices) {}

  const typename ArgType::Scalar& operator() (Eigen::Index row, Eigen::Index col) const {
    return m_arg(m_rowIndices[row], m_colIndices[col]);
  }
};

template <class ArgType, class RowIndexType, class ColIndexType>
Eigen::CwiseNullaryOp<indexing_functor<ArgType, RowIndexType, ColIndexType>,
  typename indexing_functor<ArgType, RowIndexType, ColIndexType>::MatrixType>
  submat(const Eigen::MatrixBase<ArgType>& arg, const RowIndexType& row_indices, const ColIndexType& col_indices) {
  typedef indexing_functor<ArgType, RowIndexType, ColIndexType> Func;
  typedef typename Func::MatrixType MatrixType;
  return MatrixType::NullaryExpr(row_indices.size(), col_indices.size(), Func(arg.derived(), row_indices, col_indices));
}

// calculate sort index of vector "d" in decreasing order
std::vector<int> sort_index(const Eigen::VectorXd& d) {
  std::vector<int> idx(d.size());
  std::iota(idx.begin(), idx.end(), 0);
  sort(idx.begin(), idx.end(), [&d](size_t i1, size_t i2) {return d[i1] > d[i2];});
  return idx;
}

// reorder rows in dynamic matrix "x" by integer vector "ind"
Eigen::MatrixXd reorder_rows(const Eigen::MatrixXd& x, const std::vector<int>& ind) {
  Eigen::MatrixXd x_reordered(x.rows(), x.cols());
  for (unsigned int i = 0; i < ind.size(); ++i)
    x_reordered.row(i) = x.row(ind[i]);
  return x_reordered;
}

// reorder elements in vector "x" by integer vector "ind"
Eigen::VectorXd reorder(const Eigen::VectorXd& x, const std::vector<int>& ind) {
  Eigen::VectorXd x_reordered(x.size());
  for (unsigned int i = 0; i < ind.size(); ++i)
    x_reordered(i) = x(ind[i]);
  return x_reordered;
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

double cor(Eigen::MatrixXd& x, Eigen::MatrixXd& y) {
  double x_i, y_i, sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0, sum_y2 = 0;
  const unsigned int n = x.size();
  for (unsigned int i = 0; i < n; ++i) {
    x_i = (*(x.data() + i));
    y_i = (*(y.data() + i));
    sum_x += x_i;
    sum_y += y_i;
    sum_xy += x_i * y_i;
    sum_x2 += x_i * x_i;
    sum_y2 += y_i * y_i;
  }
  return 1 - (n * sum_xy - sum_x * sum_y) / std::sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));
}

// ********************************************************************************
// SECTION 2: Non-negative least squares
// ********************************************************************************

// coordinate descent (CD) least squares given an initial 'x'
template <typename Derived>
Eigen::VectorXd c_cdnnls(
  const Eigen::MatrixXd& a,
  const typename Eigen::MatrixBase<Derived>& b,
  Eigen::VectorXd x,
  const unsigned int cd_maxit,
  const double cd_tol,
  const bool nonneg) {

  Eigen::VectorXd b0 = a * x;
  for (unsigned int i = 0; i < b0.size(); ++i) b0(i) -= b(i);
  double cd_tol_xi, cd_tol_it = 1 + cd_tol;
  for (unsigned int it = 0; it < cd_maxit && cd_tol_it > cd_tol; ++it) {
    cd_tol_it = 0;
    for (unsigned int i = 0; i < x.size(); ++i) {
      double xi = x(i) - b0(i) / a(i, i);
      if (nonneg && xi < 0) xi = 0;
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
  return x;
}

// Fast active set tuning (FAST) least squares given only 'a', 'b', and a pre-conditioned LLT decomposition of 'a'
template<typename Derived>
Eigen::VectorXd c_nnls(
  const Eigen::MatrixXd& a,
  const typename Eigen::MatrixBase<Derived>& b,
  const Eigen::LLT<Eigen::MatrixXd, 1>& a_llt,
  const unsigned int fast_maxit,
  const unsigned int cd_maxit,
  const double cd_tol,
  const bool nonneg) {

  // initialize with unconstrained least squares solution
  Eigen::VectorXd x = a_llt.solve(b);
  // iterative feasible set reduction while unconstrained least squares solutions at feasible indices contain negative values
  for (unsigned int it = 0; nonneg && it < fast_maxit && (x.array() < 0).any(); ++it) {
    // get indices in "x" greater than zero (the "feasible set")
    Eigen::VectorXi gtz_ind = find_gtz(x);
    // subset "a" and "b" to those indices in the feasible set
    Eigen::VectorXd bsub(gtz_ind.size());
    for (unsigned int i = 0; i < gtz_ind.size(); ++i) bsub(i) = b(gtz_ind(i));
    Eigen::MatrixXd asub = submat(a, gtz_ind, gtz_ind);
    // solve for those indices in "x"
    Eigen::VectorXd xsub = asub.llt().solve(bsub);
    x.setZero();
    for (unsigned int i = 0; i < gtz_ind.size(); ++i) x(gtz_ind(i)) = xsub(i);
  }

  if (cd_maxit == 0 && nonneg) return x;
  else return c_cdnnls(a, b, x, cd_maxit, cd_tol, nonneg);
}

// ********************************************************************************
// SECTION 3: Rcpp dgCMatrix class
// ********************************************************************************

// zero-copy sparse matrix class for access by reference to R objects already in memory
// note that Eigen::SparseMatrix<double> requires a deep copy of R objects for use in C++

namespace Rcpp {
  class dgCMatrix {
  public:
    Rcpp::IntegerVector i, p, Dim;
    Rcpp::NumericVector x;

    // Constructor from Rcpp::S4 object
    dgCMatrix(Rcpp::S4 m) : i(m.slot("i")), p(m.slot("p")), Dim(m.slot("Dim")), x(m.slot("x")) {}

    int rows() { return Dim[0]; }
    int cols() { return Dim[1]; }

    class InnerIterator {
    public:
      InnerIterator(dgCMatrix& ptr, int col) : ptr(ptr) { index = ptr.p[col]; max_index = ptr.p[col + 1]; }
      operator bool() const { return (index < max_index); }
      InnerIterator& operator++() { ++index; return *this; }
      const double& value() const { return ptr.x[index]; }
      int row() const { return ptr.i[index]; }
    private:
      dgCMatrix& ptr;
      int index, max_index;
    };
  };
}

// ********************************************************************************
// SECTION 4: Projecting linear factor models
// ********************************************************************************

// solve the least squares equation A = wh for h given A and w

Eigen::MatrixXd c_project(
  Rcpp::dgCMatrix& A,
  Eigen::MatrixXd& w,
  const bool nonneg,
  const unsigned int fast_maxit,
  const unsigned int cd_maxit,
  const double cd_tol,
  const double L1,
  const unsigned int threads) {

  // "w" must be in "wide" format, where w.cols() == A.rows()

  // compute left-hand side of linear system, "a"
  Eigen::MatrixXd a = w * w.transpose();
  Eigen::LLT<Eigen::MatrixXd, 1> a_llt = a.llt();
  unsigned int k = a.rows();
  Eigen::MatrixXd h(k, A.cols());

  #pragma omp parallel for num_threads(threads) schedule(dynamic)
  for (unsigned int i = 0; i < A.cols(); ++i) {
    // compute right-hand side of linear system, "b", in-place in h
    Eigen::VectorXd b = Eigen::VectorXd::Zero(k);
    for (Rcpp::dgCMatrix::InnerIterator it(A, i); it; ++it)
      for (unsigned int j = 0; j < k; ++j) b(j) += it.value() * w(j, it.row());

    if (L1 != 0) for (unsigned int j = 0; j < k; ++j) b(j) -= L1;

    // solve least squares
    h.col(i) = c_nnls(a, b, a_llt, fast_maxit, cd_maxit, cd_tol, nonneg);
  }
  return h;
}

// ********************************************************************************
// SECTION 5: Mean squared error loss of a matrix factorization model
// ********************************************************************************

//[[Rcpp::export]]
double Rcpp_mse(
  const Rcpp::S4& A_S4,
  Eigen::MatrixXd w,
  const Eigen::VectorXd& d,
  const Eigen::MatrixXd& h,
  const unsigned int threads) {

  Rcpp::dgCMatrix A(A_S4);
  if (w.rows() == h.rows()) w.transposeInPlace();

  // multiply w by diagonal
  #pragma omp parallel for num_threads(threads) schedule(dynamic)
  for (unsigned int i = 0; i < w.cols(); ++i)
    for (unsigned int j = 0; j < w.rows(); ++j)
      w(j, i) *= d(i);

  // calculate total loss with parallelization across samples
  Eigen::VectorXd losses(A.cols());
  losses.setZero();
  #pragma omp parallel for num_threads(threads) schedule(dynamic)
  for (unsigned int i = 0; i < A.cols(); ++i) {
    Eigen::VectorXd wh_i = w * h.col(i);
    for (Rcpp::dgCMatrix::InnerIterator iter(A, i); iter; ++iter)
      wh_i(iter.row()) -= iter.value();
    for (unsigned int j = 0; j < wh_i.size(); ++j)
      losses(i) += std::pow(wh_i(j), 2);
  }
  return losses.sum() / (A.cols() * A.rows());
}

// ********************************************************************************
// SECTION 6: Matrix Factorization by Alternating Least Squares
// ********************************************************************************

// structure returned in all ALSMF functions

//[[Rcpp::export]]
Rcpp::List Rcpp_nmf(
  const Rcpp::S4& A_S4,
  const Rcpp::S4& At_S4,
  Eigen::MatrixXd w,
  const double tol = 1e-3,
  const bool nonneg = true,
  const double L1_w = 0,
  const double L1_h = 0,
  const unsigned int maxit = 100,
  const bool diag = true,
  const unsigned int fast_maxit = 10,
  const unsigned int cd_maxit = 100,
  const double cd_tol = 1e-8,
  const bool verbose = false,
  const unsigned int threads = 0) {

  unsigned int k = w.rows();
  Rcpp::dgCMatrix A(A_S4), At(At_S4);
  Eigen::MatrixXd h(k, A.cols());
  Eigen::VectorXd d(k);
  d = d.setOnes();

  if (verbose) Rprintf("\n%4s | %8s \n---------------\n", "iter", "tol");

  for (unsigned int it = 0; it < maxit; ++it) {
    // update h
    h = c_project(A, w, nonneg, fast_maxit, cd_maxit, cd_tol, L1_h, threads);

    // reset diagonal and scale "h"
    for (unsigned int i = 0; diag && i < k; ++i) {
      d[i] = h.row(i).sum();
      for (unsigned int j = 0; j < A.cols(); ++j) h(i, j) /= d(i);
    }

    // update w
    Eigen::MatrixXd w_it = w;
    w = c_project(At, h, nonneg, fast_maxit, cd_maxit, cd_tol, L1_w, threads);

    // reset diagonal and scale "w"
    for (unsigned int i = 0; diag && i < k; ++i) {
      d[i] = w.row(i).sum();
      for (unsigned int j = 0; j < A.rows(); ++j) w(i, j) /= d(i);
    }

    // calculate tolerance
    double tol_ = cor(w, w_it);
    Rcpp::checkUserInterrupt();
    if (verbose) Rprintf("%4d | %8.2e\n", it + 1, tol_);

    // check for convergence
    if (tol_ < tol) break;
  }

  // reorder factors by diagonal
  if (diag) {
    std::vector<int> indx = sort_index(d);
    w = reorder_rows(w, indx);
    d = reorder(d, indx);
    h = reorder_rows(h, indx);
  }

  return Rcpp::List::create(Rcpp::Named("w") = w.transpose(), Rcpp::Named("d") = d, Rcpp::Named("h") = h);
}

// ********************************************************************************
// SECTION 7: Rcpp wrapper functions
// ********************************************************************************

// Non-Negative Least Squares

//[[Rcpp::export]]
Eigen::MatrixXd Rcpp_nnls(
  const Eigen::MatrixXd& a,
  Eigen::MatrixXd b,
  const unsigned int fast_maxit = 10,
  const unsigned int cd_maxit = 100,
  const double cd_tol = 1e-8,
  const bool nonneg = true) {

  Eigen::LLT<Eigen::MatrixXd> a_llt = a.llt();
  for (unsigned int i = 0; i < b.cols(); ++i)
    b.col(i) = c_nnls(a, b.col(i), a.llt(), fast_maxit, cd_maxit, cd_tol, nonneg);
  return b;
}

//[[Rcpp::export]]
Eigen::MatrixXd Rcpp_cdnnls(
  const Eigen::MatrixXd& a,
  Eigen::MatrixXd& b,
  Eigen::MatrixXd x,
  const unsigned int cd_maxit = 100,
  const double cd_tol = 1e-8,
  const bool nonneg = true) {

  for (unsigned int i = 0; i < b.cols(); ++i) {
    Eigen::VectorXd x_i = x.col(i);
    x.col(i) = c_cdnnls(a, b.col(i), x_i, cd_maxit, cd_tol, nonneg);
  }
  return x;
}

//[[Rcpp::export]]
Eigen::MatrixXd Rcpp_project(
  const Rcpp::S4& A_S4,
  Eigen::MatrixXd& w,
  const bool nonneg,
  const unsigned int fast_maxit,
  const unsigned int cd_maxit,
  const double cd_tol,
  const double L1,
  const unsigned int threads) {
    
  Rcpp::dgCMatrix A(A_S4);
  return c_project(A, w, nonneg, fast_maxit, cd_maxit, cd_tol, L1, threads);
}

