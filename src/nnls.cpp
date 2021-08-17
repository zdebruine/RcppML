// This file is part of RcppML, an Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
// github.com/zdebruine/RcppML

#include "RcppML.h"

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
Eigen::VectorXd c_nnls(
  const Eigen::MatrixXd& a,
  const typename Eigen::VectorXd& b,
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

//[[Rcpp::export]]
Eigen::MatrixXd Rcpp_nnls(
  const Eigen::MatrixXd& a,
  Eigen::MatrixXd b,
  const unsigned int fast_maxit = 10,
  const unsigned int cd_maxit = 100,
  const double cd_tol = 1e-8,
  const bool nonneg = true) {

  Eigen::LLT<Eigen::MatrixXd> a_llt = a.llt();
  for (unsigned int i = 0; i < b.cols(); ++i){
    Eigen::VectorXd b_i = b.col(i);
    b.col(i) = c_nnls(a, b_i, a.llt(), fast_maxit, cd_maxit, cd_tol, nonneg);
  }
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
    Eigen::VectorXd b_i = b.col(i);
    x.col(i) = c_cdnnls(a, b_i, x_i, cd_maxit, cd_tol, nonneg);
  }
  return x;
}

