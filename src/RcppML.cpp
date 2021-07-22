// This file is part of RcppML, an Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
// github.com/zdebruine/RcppML

#define EIGEN_NO_DEBUG
#define EIGEN_INITIALIZE_MATRICES_BY_ZERO

#pragma GCC diagnostic ignored "-Wignored-attributes"

//[[Rcpp::plugins(openmp)]]
#ifdef _OPENMP
#include <omp.h>
#endif

//[[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

// CONTENTS
// section 1: helper functions
// section 2: non-negative least squares
// section 3: projecting linear factor models
// section 4: mean squared error loss of a factor model
// section 5: matrix factorization by alternating least squares
// section 6: Rcpp wrapper functions

// ********************************************************************************
// SECTION 1: Helper functions
// ********************************************************************************

// these template functions for matrix subsetting are documented here:
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

// trim a vector to length n
template<typename T>
Eigen::Matrix<T, -1, 1> trim(Eigen::Matrix<T, -1, 1> x, const unsigned int n) {
  Eigen::Matrix<T, -1, 1> y(n);
  for (unsigned int i = 0; i < n; ++i) y(i) = x(i);
  return y;
}

// calculate sort index of vector "d" in decreasing order
template <typename T>
std::vector<int> sort_index(const Eigen::Matrix<T, -1, 1>& d) {
  std::vector<int> idx(d.size());
  std::iota(idx.begin(), idx.end(), 0);
  sort(idx.begin(), idx.end(), [&d](size_t i1, size_t i2) {return d[i1] > d[i2];});
  return idx;
}

// reorder rows in dynamic matrix "x" by integer vector "ind"
template <typename T>
Eigen::Matrix<T, -1, -1> reorder_rows(const Eigen::Matrix<T, -1, -1>& x, const std::vector<int>& ind) {
  Eigen::Matrix<T, -1, -1> x_reordered(x.rows(), x.cols());
  for (unsigned int i = 0; i < ind.size(); ++i)
    x_reordered.row(i) = x.row(ind[i]);
  return x_reordered;
}

// reorder elements in vector "x" by integer vector "ind"
template <typename T>
Eigen::Matrix<T, -1, 1> reorder(const Eigen::Matrix<T, -1, 1>& x, const std::vector<int>& ind) {
  Eigen::Matrix<T, -1, 1> x_reordered(x.size());
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

template<typename T>
T cor(Eigen::Matrix<T, -1, -1>& x, Eigen::Matrix<T, -1, -1>& y) {
  T x_i, y_i, sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0, sum_y2 = 0;
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
template<typename T, typename Derived>
Eigen::Matrix<T, -1, 1> c_cdnnls(const Eigen::Matrix<T, -1, -1>& a, const typename Eigen::MatrixBase<Derived>& b,
                                 Eigen::Matrix<T, -1, 1> x, const unsigned int cd_maxit, const T cd_tol, const bool nonneg) {
  
  Eigen::Matrix<T, -1, 1> b0 = a * x;
  for (unsigned int i = 0; i < b0.size(); ++i) b0(i) -= b(i);
  T cd_tol_xi, cd_tol_it = 1 + cd_tol;
  for (unsigned int it = 0; it < cd_maxit && cd_tol_it > cd_tol; ++it) {
    cd_tol_it = 0;
    for (unsigned int i = 0; i < x.size(); ++i) {
      T xi = x(i) - b0(i) / a(i, i);
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
template<typename T, typename Derived>
Eigen::Matrix<T, -1, 1> c_nnls(const Eigen::Matrix<T, -1, -1>& a, const typename Eigen::MatrixBase<Derived>& b,
                               const Eigen::LLT<Eigen::Matrix<T, -1, -1>, 1>& a_llt, const unsigned int fast_maxit, 
                               const unsigned int cd_maxit, const T cd_tol, const bool nonneg) {
  
  // initialize with unconstrained least squares solution
  Eigen::Matrix<T, -1, 1> x = a_llt.solve(b);
  // iterative feasible set reduction while unconstrained least squares solutions at feasible indices contain negative values
  for (unsigned int it = 0; nonneg && it < fast_maxit && (x.array() < 0).any(); ++it) {
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

  return c_cdnnls(a, b, x, cd_maxit, cd_tol, nonneg);
}

// ********************************************************************************
// SECTION 3: Projecting linear factor models
// ********************************************************************************

// solve the least squares equation A = WH for H given A and W
template<typename T>
Eigen::Matrix<T, -1, -1> c_project(const Eigen::SparseMatrix<T>& A, Eigen::Matrix<T, -1, -1>& w, const bool nonneg = true,
                                   const unsigned int fast_maxit = 10, const unsigned int cd_maxit = 100, const T cd_tol = 1e-8, 
                                   const T L1 = 0, const unsigned int threads = 0) {
  
  // "w" must be in "wide" format, where w.cols() == A.rows()
  
  // compute left-hand side of linear system, "a"
  Eigen::Matrix<T, -1, -1> a = w * w.transpose();
  Eigen::LLT<Eigen::Matrix<T, -1, -1>, 1> a_llt = a.llt();

  unsigned int k = a.rows();
  Eigen::Matrix<T, -1, -1> h(k, A.cols());
  
  #pragma omp parallel for num_threads(threads) schedule(dynamic)
  for (unsigned int i = 0; i < A.cols(); ++i) {
    // compute right-hand side of linear system, "b", in-place in h
    Eigen::Matrix<T, -1, 1> b(k);
    for (typename Eigen::SparseMatrix<T>::InnerIterator it(A, i); it; ++it)
      for (unsigned int j = 0; j < k; ++j) b(j) += it.value() * w(j, it.row());
    
    if (L1 != 0) for (unsigned int j = 0; j < k; ++j) b(j) -= L1;
    
    // solve least squares
    h.col(i) = c_nnls(a, b, a_llt, fast_maxit, cd_maxit, cd_tol, nonneg);
  }
  return h;
}

// ********************************************************************************
// SECTION 4: Mean squared error loss of a matrix factorization model
// ********************************************************************************

// sparse "A"
template<typename T>
T c_mse(const Eigen::SparseMatrix<T>& A, Eigen::Matrix<T, -1, -1> w, const Eigen::Matrix<T, -1, 1>& d,
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
// SECTION 5: Matrix Factorization by Alternating Least Squares
// ********************************************************************************

// structure returned in all ALSMF functions
template <typename T>
struct WDH {
  Eigen::Matrix<T, -1, -1> W;
  Eigen::Matrix<T, -1, 1> D;
  Eigen::Matrix<T, -1, -1> H;
  Eigen::Matrix<T, -1, 1> tol;
  Eigen::Matrix<T, -1, 1> mse;
};

template <typename T>
WDH<T> c_nmf(const Eigen::SparseMatrix<T>& A, unsigned int k, const T tol = 1e-3, const bool nonneg_w = true,
             const bool nonneg_h = true, const T L1_w = 0, const T L1_h = 0, const unsigned int maxit = 100,
             const bool calc_mse = false, bool symmetric = false, const bool diag = true, const unsigned int fast_maxit = 10, const unsigned int cd_maxit = 100,
             const T cd_tol = 1e-8, const bool verbose = false) {
  
  const unsigned int threads = Eigen::nbThreads();
  const unsigned int n_features = A.rows(), n_samples = A.cols();
  Eigen::SparseMatrix<T> At;
  if(!symmetric) At = A.transpose();
  Eigen::Matrix<T, -1, -1> h(k, n_samples), w_it(k, n_features), a(k, k), w(k, n_features);
  Eigen::Matrix<T, -1, 1> d(k), tols(maxit), losses(maxit);
  Eigen::LLT<Eigen::Matrix<T, -1, -1>> a_llt;
  d.setOnes(); tols.setOnes(); losses.setZero();
  
  w = w.setRandom();
  w = w.cwiseAbs();
  
  if (verbose) Rprintf("\n%4s | %8s \n---------------\n", "iter", "tol"); // REMOVE THIS LINE FOR C++ USE ONLY
  
  for (unsigned int it = 0; it < maxit; ++it) {
    // update h
    if(fast_maxit > 0){
      // fast-cd update of 'h'
      h = c_project(A, w, nonneg_h, fast_maxit, cd_maxit, cd_tol, L1_h, threads);
    } else {
      // coordinate descent least squares update of 'h'
      a = w * w.transpose();
      #pragma omp parallel for num_threads(threads) schedule(dynamic)
      for (unsigned int i = 0; i < A.cols(); ++i) {
        Eigen::Matrix<T, -1, 1> b(k);
        for (typename Eigen::SparseMatrix<T>::InnerIterator it(A, i); it; ++it)
          for (unsigned int j = 0; j < k; ++j) b(j) += it.value() * w(j, it.row());
        if (L1_h != 0) for (unsigned int j = 0; j < k; ++j) b(j) -= L1_h;
        Eigen::Matrix<T, -1, 1> x = h.col(i);
        h.col(i) = c_cdnnls(a, b, x, cd_maxit, cd_tol, nonneg_h);
      }
    }
    
    // reset diagonal and scale "h"
    if (diag || symmetric) {
      for (unsigned int i = 0; i < k; ++i) {
        d[i] = h.row(i).sum();
        for (unsigned int j = 0; j < A.cols(); ++j) h(i, j) /= d(i);
      }
    }
    
    // update w
    w_it = w;
    if(fast_maxit > 0){
      if(!symmetric) w = c_project(At, h, nonneg_w, fast_maxit, cd_maxit, cd_tol, L1_w, threads);
      else if(symmetric) w = c_project(A, h, nonneg_w, fast_maxit, cd_maxit, cd_tol, L1_w, threads);
    } else {
      // coordinate descent least squares update of 'w'
      a = h * h.transpose();
      #pragma omp parallel for num_threads(threads) schedule(dynamic)
      for (unsigned int i = 0; i < A.rows(); ++i) {
        Eigen::Matrix<T, -1, 1> b(k);
        if(!symmetric){
          for (typename Eigen::SparseMatrix<T>::InnerIterator it(At, i); it; ++it)
            for (unsigned int j = 0; j < k; ++j) b(j) += it.value() * h(j, it.row());
        } else {
          for (typename Eigen::SparseMatrix<T>::InnerIterator it(A, i); it; ++it)
            for (unsigned int j = 0; j < k; ++j) b(j) += it.value() * h(j, it.row());
        }
        if (L1_w != 0) for (unsigned int j = 0; j < k; ++j) b(j) -= L1_w;    
        Eigen::Matrix<T, -1, 1> x = w.col(i);
        w.col(i) = c_cdnnls(a, b, x, cd_maxit, cd_tol, nonneg_w);
      }
    }
    
    // reset diagonal and scale "w"
    if (diag || symmetric) {
      for (unsigned int i = 0; i < k; ++i) {
        d[i] = w.row(i).sum();
        for (unsigned int j = 0; j < A.rows(); ++j) w(i, j) /= d(i);
      }
    }
    
    // calculate mean squared error loss
    if (calc_mse) losses(it) = c_mse(A, w, d, h);
    
    // calculate tolerance
    if (tol > 0) {
      tols(it) = symmetric ? cor(h, w) : cor(w, w_it);
      if (verbose) {
        Rcpp::checkUserInterrupt(); // REMOVE THIS LINE FOR C++ USE ONLY
        Rprintf("%4d | %8.2e\n", it + 1, tols(it)); // REMOVE THIS LINE FOR C++ USE ONLY
      }
    }
    
    // check for convergence
    if (tols(it) < tol) {
      tols = trim(tols, it);
      losses = trim(losses, it);
      break;
    }
  }
  
  // reorder diagonal if needed
  if (diag) {
    std::vector<int> indx = sort_index(d);
    w = reorder_rows(w, indx);
    d = reorder(d, indx);
    h = reorder_rows(h, indx);
  }
  
  return WDH<T> {w.transpose(), d, h, tols, losses};
}

// ********************************************************************************
// SECTION 6: Rcpp wrapper functions
// ********************************************************************************

// Non-Negative Least Squares

//[[Rcpp::export]]
Eigen::MatrixXd Rcpp_nnls_double(const Eigen::MatrixXd& a, Eigen::MatrixXd b, const unsigned int fast_maxit = 10, const unsigned int cd_maxit = 100, const double cd_tol = 1e-8, const bool nonneg = true) {
  
  Eigen::LLT<Eigen::MatrixXd> a_llt = a.llt();
  for (unsigned int i = 0; i < b.cols(); ++i)
    b.col(i) = c_nnls(a, b.col(i), a.llt(), fast_maxit, cd_maxit, cd_tol, nonneg);
  return b;
}

//[[Rcpp::export]]
Eigen::MatrixXf Rcpp_nnls_float(const Eigen::MatrixXf& a, Eigen::MatrixXf b, const unsigned int fast_maxit = 10, const unsigned int cd_maxit = 100, const float cd_tol = 1e-8, const bool nonneg = true) {
  
  Eigen::LLT<Eigen::MatrixXf> a_llt = a.llt();
  for (unsigned int i = 0; i < b.cols(); ++i)
    b.col(i) = c_nnls(a, b.col(i), a.llt(), fast_maxit, cd_maxit, cd_tol, nonneg);
  return b;
}

//[[Rcpp::export]]
Eigen::MatrixXd Rcpp_cdnnls_double(const Eigen::MatrixXd& a, Eigen::MatrixXd& b, Eigen::MatrixXd x, const unsigned int cd_maxit = 100, const double cd_tol = 1e-8, const bool nonneg = true){
  
  for(unsigned int i = 0; i < b.cols(); ++i){
    Eigen::VectorXd x_i = x.col(i);
    x.col(i) = c_cdnnls(a, b.col(i), x_i, cd_maxit, cd_tol, nonneg);
  }
  return x;
}

//[[Rcpp::export]]
Eigen::MatrixXf Rcpp_cdnnls_float(const Eigen::MatrixXf& a, Eigen::MatrixXf& b, Eigen::MatrixXf x, const unsigned int cd_maxit = 100, const float cd_tol = 1e-8, const bool nonneg = true){
  
  for(unsigned int i = 0; i < b.cols(); ++i){
    Eigen::VectorXf x_i = x.col(i);
    x.col(i) = c_cdnnls(a, b.col(i), x_i, cd_maxit, cd_tol, nonneg);
  }
  return x;
}

// Projecting linear models

//[[Rcpp::export]]
Eigen::MatrixXd Rcpp_project_double(const Eigen::SparseMatrix<double>& A, Eigen::MatrixXd& w, const bool nonneg = true,
                                    const unsigned int fast_maxit = 10, const unsigned int cd_maxit = 100, const double cd_tol = 1e-8, const double L1 = 0, const unsigned int threads = 0) {
  
  return c_project(A, w, nonneg, fast_maxit, cd_maxit, cd_tol, L1, threads);
}

//[[Rcpp::export]]
Eigen::MatrixXf Rcpp_project_float(const Eigen::SparseMatrix<float>& A, Eigen::MatrixXf& w, const bool nonneg = true,
                                   const unsigned int fast_maxit = 10, const unsigned int cd_maxit = 100, const float cd_tol = 1e-8, const float L1 = 0, const unsigned int threads = 0) {
  
  return c_project(A, w, nonneg, fast_maxit, cd_maxit, cd_tol, L1, threads);
}

// Calculating mean squared error of a factorization

//[[Rcpp::export]]
double Rcpp_mse_double(const Eigen::SparseMatrix<double>& A, const Eigen::MatrixXd& w, const Eigen::VectorXd& d, const Eigen::MatrixXd& h) {
  
  return c_mse(A, w, d, h);
}

//[[Rcpp::export]]
float Rcpp_mse_float(const Eigen::SparseMatrix<float>& A, const Eigen::MatrixXf& w, const Eigen::VectorXf& d, const Eigen::MatrixXf& h) {
  
  return c_mse(A, w, d, h);
}

// Matrix factorization by alternating least squares

//[[Rcpp::export]]
Rcpp::List Rcpp_nmf_double(const Eigen::SparseMatrix<double>& A, const unsigned int k, const unsigned int seed = 0,
                           const double tol = 1e-3, const bool nonneg_w = true, const bool nonneg_h = true, const double L1_w = 0,
                           const double L1_h = 0, const unsigned int maxit = 25, const unsigned int threads = 0,
                           const bool verbose = true, const bool calc_mse = false, const bool symmetric = false, const bool diag = true,
                           const unsigned int fast_maxit = 10, const unsigned int cd_maxit = 100, const double cd_tol = 1e-8) {
  
  if (seed == 0) srand((unsigned int)time(0));
  else srand(seed);
  
  Eigen::setNbThreads(threads);
  
  WDH<double> model = c_nmf(A, k, tol, nonneg_w, nonneg_h, L1_w, L1_h, maxit, calc_mse, symmetric, diag, fast_maxit, cd_maxit, cd_tol, verbose);
  return(Rcpp::List::create(
      Rcpp::Named("w") = model.W,
      Rcpp::Named("d") = model.D,
      Rcpp::Named("h") = model.H,
      Rcpp::Named("tol") = model.tol,
      Rcpp::Named("mse") = model.mse));
}

//[[Rcpp::export]]
Rcpp::List Rcpp_nmf_float(const Eigen::SparseMatrix<float>& A, const unsigned int k, const unsigned int seed = 0,
                          const float tol = 1e-3, const bool nonneg_w = true, const bool nonneg_h = true, const float L1_w = 0,
                          const float L1_h = 0, const unsigned int maxit = 25, const unsigned int threads = 0,
                          const bool verbose = true, const bool calc_mse = false, const bool symmetric = false, const bool diag = true,
                          const unsigned int fast_maxit = 10, const unsigned int cd_maxit = 100, const float cd_tol = 1e-8) {
  
  if (seed == 0) srand((unsigned int)time(0));
  else srand(seed);
  
  Eigen::setNbThreads(threads);
  
  WDH<float> model = c_nmf(A, k, tol, nonneg_w, nonneg_h, L1_w, L1_h, maxit, calc_mse, symmetric, diag, fast_maxit, cd_maxit, cd_tol, verbose);
  return(Rcpp::List::create(
      Rcpp::Named("w") = model.W,
      Rcpp::Named("d") = model.D,
      Rcpp::Named("h") = model.H,
      Rcpp::Named("tol") = model.tol,
      Rcpp::Named("mse") = model.mse));
}
