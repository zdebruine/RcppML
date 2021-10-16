// This file is part of RcppML, a Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU
// Public License v. 2.0.

#ifndef RcppML_bits
#define RcppML_bits

#ifndef RcppML_common
#include <RcppMLCommon.hpp>
#endif

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

Eigen::MatrixXd submat(const Eigen::MatrixXd& x, const Eigen::VectorXi& col_indices){
  Eigen::MatrixXd x_(x.rows(), col_indices.size());
  for(unsigned int i = 0; i < col_indices.size(); ++i)
    x_.col(i) = x.col(col_indices(i));
  return x_;
}

inline Eigen::VectorXd subvec(const Eigen::MatrixXd& b, const Eigen::VectorXi& ind, const unsigned int col) {
  Eigen::VectorXd bsub(ind.size());
  for (unsigned int i = 0; i < ind.size(); ++i) bsub(i) = b(ind(i), col);
  return bsub;
}

inline Eigen::VectorXi find_gtz(const Eigen::MatrixXd& x, const unsigned int col) {
  unsigned int n_gtz = 0;
  for (unsigned int i = 0; i < x.rows(); ++i) if (x(i, col) > 0) ++n_gtz;
  Eigen::VectorXi gtz(n_gtz);
  unsigned int j = 0;
  for (unsigned int i = 0; i < x.rows(); ++i) {
    if (x(i, col) > 0) {
      gtz(j) = i;
      ++j;
    }
  }
  return gtz;
}

// Pearson correlation between two matrices
inline double cor(Eigen::MatrixXd& x, Eigen::MatrixXd& y) {
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

// calculate sort index of vector "d" in decreasing order
inline std::vector<int> sort_index(const Eigen::VectorXd& d) {
  std::vector<int> idx(d.size());
  std::iota(idx.begin(), idx.end(), 0);
  sort(idx.begin(), idx.end(), [&d](size_t i1, size_t i2) {return d[i1] > d[i2];});
  return idx;
}

// reorder rows in dynamic matrix "x" by integer vector "ind"
inline Eigen::MatrixXd reorder_rows(const Eigen::MatrixXd& x, const std::vector<int>& ind) {
  Eigen::MatrixXd x_reordered(x.rows(), x.cols());
  for (unsigned int i = 0; i < ind.size(); ++i)
    x_reordered.row(i) = x.row(ind[i]);
  return x_reordered;
}

// reorder elements in vector "x" by integer vector "ind"
inline Eigen::VectorXd reorder(const Eigen::VectorXd& x, const std::vector<int>& ind) {
  Eigen::VectorXd x_reordered(x.size());
  for (unsigned int i = 0; i < ind.size(); ++i)
    x_reordered(i) = x(ind[i]);
  return x_reordered;
}

std::vector<double> getRandomValues(const unsigned int len, const unsigned int seed){
  if(seed > 0){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(std::floor(std::fabs(seed)));
  }
  Rcpp::NumericVector R_RNG_random_values = Rcpp::runif(len);
  std::vector<double> random_values = Rcpp::as<std::vector<double> >(R_RNG_random_values);
  return random_values;
}

Eigen::MatrixXd randomMatrix(const unsigned int nrow, const unsigned int ncol, const unsigned int seed){
  std::vector<double> random_values = getRandomValues(nrow * ncol, seed);
  Eigen::MatrixXd x(nrow, ncol);
  unsigned int indx = 0;
  for(unsigned int r = 0; r < nrow; ++r)
    for(unsigned int c = 0; c < ncol; ++c, ++indx)
      x(r, c) = random_values[indx];
  return x;
}

inline bool isAppxSymmetric(Eigen::MatrixXd& A) {
  if (A.rows() == A.cols()) {
    for (int i = 0; i < A.cols(); ++i)
      if (A(i, 0) != A(0, i))
        return false;
    return true;
  } else return false;
}

inline std::vector<unsigned int> nonzeroRowsInCol(const Eigen::MatrixXd& x, const unsigned int i){
  std::vector<unsigned int> nonzeros(x.rows());
  unsigned int it_nz = 0;
  for(unsigned int it = 0; it < x.rows(); ++it){
    if(x(it, i) != 0){
      nonzeros[it_nz] = it;
      ++it_nz;
    }
  }
  nonzeros.resize(it_nz);
  return nonzeros;
}

inline unsigned int n_nonzeros(const Eigen::MatrixXd& x) {
  unsigned int nz = 0;
  for (unsigned int i = 0, size = x.size(); i < size; ++i)
    if(*(x.data() + i) == 0) ++nz;
  return nz;
}

inline Eigen::MatrixXd addL2(Eigen::MatrixXd a, double L2, const std::string& scale_L2){
  if(scale_L2 == "mean") L2 *= a.diagonal().array().mean();
  else if(scale_L2 == "sum") L2 *= a.diagonal().array().sum();
  else if(scale_L2 == "max") L2 *= a.diagonal().array().maxCoeff();
  a.diagonal().array() += L2;
  return a;
}

inline Eigen::MatrixXd addPE(Eigen::MatrixXd a, double PE, const std::string& scale_PE){
  if(scale_PE == "mean") PE *= a.diagonal().array().mean();
  else if(scale_PE == "sum") PE *= a.diagonal().array().sum();
  else if(scale_PE == "max") PE *= a.diagonal().array().maxCoeff();
  a.array() += PE;
  a.diagonal().array() -= PE;
  return a;
}

#endif