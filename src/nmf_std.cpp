// This file is a standalone RcppEigen demo for basic minimal NMF

// hide the horrendous swamp of compile-time warnings about ignored attributes
#pragma GCC diagnostic ignored "-Wignored-attributes"

//[[Rcpp::depends(RcppSparse)]]
#include <RcppSparse.h>
//[[Rcpp::depends(RcppClock)]]
//[[Rcpp::depends(RcppEigen)]]
//[[Rcpp::plugins(openmp)]]
#include <RcppEigen.h>
#include <RcppClock.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define USING_RCPP
#define NNLS_MAX_ITER 100
#define NNLS_TOL 1e-8
#define TINY_NUM_FOR_STABILITY 1e-15

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

// Pearson correlation between two matrices
inline double cor(Eigen::MatrixXd& x, Eigen::MatrixXd& y) {
  double x_i, y_i, sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0, sum_y2 = 0;
  const int n = x.size();
  for (int i = 0; i < n; ++i) {
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

inline Eigen::VectorXi InnerNNZs(RcppSparse::Matrix& A, const int i) {
  const int begin = A.p[i], end = A.p[i + 1];
  Eigen::VectorXi nnz(end - begin);
  for (int it = begin; it < end; ++it)
    nnz(it) = A.i[it];
  return nnz;
}

// Non-Negative Least Squares solver
// solve ax = b given "a", "b", and h.col(sample) giving "x", subject to non-negativity. Coordinate descent.
inline void nnls(Eigen::MatrixXd& a, Eigen::VectorXd& b, Eigen::MatrixXd& h, const int sample) {
  double tol = 1;
  // tolerance is calculated for the entire solution as a whole. This is both for performance
  //    (fewer temporaries and comparisons) but also to stabilize the measure across the entire solution
  for (int it = 0; it < NNLS_MAX_ITER && (tol / b.size()) > NNLS_TOL; ++it) {
    tol = 0;
    for (int i = 0; i < h.rows(); ++i) {
      double diff = b(i) / a(i, i);
      // for efficiency, if...else based on whether h(i, sample) is to be set to zero or not
      //   this approach is rather unintuitive, but increases runtime ~15% for systems 10-20 variables in size
      if (-diff > h(i, sample)) {
        // if the current solution is positive, but the gradient requests it to become negative, h(i, sample)
        //   will be set to zero. So we update the gradient accordingly and mandate another iteration
        if (h(i, sample) != 0) {
          b -= a.col(i) * -h(i, sample);
          tol = 1;
          h(i, sample) = 0;
        }
      } else if (diff != 0) {
        // in this case, simply update the solution, gradient, and tolerance
        h(i, sample) += diff;
        b -= a.col(i) * diff;
        tol += std::abs(diff / (h(i, sample) + TINY_NUM_FOR_STABILITY));
      }
    }
  }
}

// solve for H without masking zeros in A
inline void update(RcppSparse::Matrix A, const Eigen::MatrixXd& w, Eigen::MatrixXd& h, const double L1, const double L2,
                   const int threads, const bool masking_h, const Eigen::MatrixXd& mask_h, const bool masking_A,
                   RcppSparse::Matrix mask_A) {

  // calculate "a"
  //  * calculate "a" for updates of all columns of "h"
  //  * if masking is applied to "A", we will subtract away the contributions of masked
  //       values in each column update
  Eigen::MatrixXd a = w * w.transpose();
  a.diagonal().array() += L2 + TINY_NUM_FOR_STABILITY;

  #ifdef _OPENMP
  #pragma omp parallel for num_threads(threads)
  #endif
  for (int i = 0; i < h.cols(); ++i) {

    // if there are no nonzeros in this column of "A", no need to solve anything
    if (A.p[i] == A.p[i + 1]) continue;

    // find the number of masked values in "A.col(i)"
    int num_masked = 0;
    if (masking_A)
      num_masked = mask_A.p[i + 1] - mask_A.p[i];

    Eigen::MatrixXd a_i;

    // calculate "b"
    Eigen::VectorXd b = Eigen::VectorXd::Zero(h.rows());
    if (num_masked == 0) {
      // calculate "b" without masking on "A"
      for (RcppSparse::Matrix::InnerIterator it(A, i); it; ++it)
        b += it.value() * w.col(it.row());
    } else {
      // calculate "b" with weighted masking on "A"
      //  * traverse both A.col(i) and mask_A.col(i) similar to a boost ForwardTraversalIterator
      RcppSparse::Matrix::InnerIterator it_A(A, i), it_mask(mask_A, i);
      while (it_A) {
        if (!it_mask || it_A.row() < it_mask.row()) {
          b += it_A.value() * w.col(it_A.row());
          ++it_A;
        } else if (it_mask && it_A.row() == it_mask.row()) {
          if (it_mask.value() < 1)
            b += ((it_A.value() * (1 - it_mask.value())) * w.col(it_A.row()));
          ++it_mask; ++it_A;
        } else if (it_mask) {
          ++it_mask;
        }
      }
      // if masking values in A.col(i), subtract contributions of masked indices from "a"
      //  * we only need to consider columns in "w" that correspond to non-zero rows in "A" because
      //      this code block does not consider the masked_zeros case
      Eigen::MatrixXd w_(w.rows(), num_masked);
      int j = 0;
      for (RcppSparse::Matrix::InnerIterator it(mask_A, i); it; ++it, ++j)
        w_.col(j) = w.col(it.row()) * it.value();
      Eigen::MatrixXd a_ = w_ * w_.transpose();
      a_i = a - a_;
    }

    // apply L1 penalty on "b"
    if (L1 != 0) b.array() -= L1;

    // apply masking on "h"
    if (masking_h) b.array() *= mask_h.col(i).array();

    // solve nnls equations
    (num_masked == 0) ? nnls(a, b, h, i) : nnls(a_i, b, h, i);
  }
}

// update for H with masking zeros in A
inline void update_mask_zeros(RcppSparse::Matrix A, const Eigen::MatrixXd& w, Eigen::MatrixXd& h, const double L1, const double L2,
                              const int threads, const bool masking_h, const Eigen::MatrixXd& mask_h, const bool masking_A,
                              RcppSparse::Matrix mask_A) {

  #ifdef _OPENMP
  #pragma omp parallel for num_threads(threads)
  #endif
  for (int i = 0; i < h.cols(); ++i) {
    if (A.p[i] == A.p[i + 1]) continue;

    int num_masked = 0;
    if (masking_A) num_masked = mask_A.p[i + 1] - mask_A.p[i];

    Eigen::VectorXi nnz(A.p[i + 1] - A.p[i]);
    for (int ind = A.p[i], j = 0; ind < A.p[i + 1]; ++ind, ++j)
      nnz(j) = A.i[ind];
    Eigen::MatrixXd w_ = submat(w, nnz, nnz);

    Eigen::VectorXd b = Eigen::VectorXd::Zero(h.rows());
    if (num_masked == 0) {
      for (RcppSparse::Matrix::InnerIterator it(A, i); it; ++it)
        b += it.value() * w.col(it.row());
    } else {
      // subset "w" at non-masked indices in A.col(i) to calculate "a"
      RcppSparse::Matrix::InnerIterator it_mask(mask_A, i), it_A(A, i);
      int j = 0;
      while (it_mask && it_A) {
        if (it_mask.row() == it_A.row()) {
          w_.col(j) *= (1 - it_mask.value());
          ++it_mask; ++it_A; ++j;
        } else if (it_mask.row() < it_A.row()) {
          ++it_mask;
        } else {
          ++it_A;
        }
      }

      // calculate "b" with masking on "A"
      RcppSparse::Matrix::InnerIterator it_A(A, i), it_mask(mask_A, i);
      while (it_A) {
        if (!it_mask || it_A.row() < it_mask.row()) {
          b += it_A.value() * w.col(it_A.row());
          ++it_A;
        } else if (it_mask && it_A.row() == it_mask.row()) {
          if (it_mask.value() < 1)
            b += ((it_A.value() * (1 - it_mask.value())) * w.col(it_A.row()));
          ++it_mask; ++it_A;
        } else if (it_mask) {
          ++it_mask;
        }
      }
    }
    Eigen::MatrixXd a = w_ * w_.transpose();

    if (L1 != 0) b.array() -= L1;
    a.diagonal().array() += L2 + TINY_NUM_FOR_STABILITY;
    if (masking_h) b.array() *= mask_h.col(i).array();
    nnls(a, b, h, i);
  }
}

inline void scale(Eigen::MatrixXd& h, Eigen::VectorXd& d, std::string norm) {
  if (norm == "none") {
    d = Eigen::VectorXd::Ones(h.rows());
  } else {
    if (norm == "abs") {
      for (int i = 0; i < h.cols(); ++i)
        for (int j = 0; j < h.rows(); ++j)
          d(j) += h(j, i);
    } else if (norm == "2norm") {
      for (int i = 0; i < h.cols(); ++i)
        for (int j = 0; j < h.rows(); ++j)
          d(j) += h(j, i) * h(j, i);
      for (int i = 0; i < d.size(); ++i)
        d(i) = std::sqrt(d(i));
    }
    d.array() += TINY_NUM_FOR_STABILITY;
    for (int i = 0; i < h.rows(); ++i)
      for (int j = 0; j < h.cols(); ++j)
        h(i, j) /= d(i);
  }
}

namespace RcppML {
  template <typename T>
  class nmf {
  private:
    T& A, mask_A_;
    T At, mask_At_;
    Eigen::MatrixXd w, h, w_it, mask_w, mask_h;
    bool masking_w = false, masking_h = false, masking_A = false,
      mask_zeros = false, symmetric = false, verbose_ = false,
      is_initialized = false, is_transposed = false;
    Eigen::VectorXd d;
    double tol_ = 1, target_tol = 1e-4, L1_w = 0, L1_h = 0, L2_w = 0, L2_h = 0;
    int iter, max_iter_ = 100, k, rows, cols, threads_ = 0;
    std::string norm_ = "abs";

    void scaleH() { scale(h, d, norm_); }
    void scaleW() { scale(w, d, norm_); }

  public:
    // CONSTRUCTOR
    nmf(T& A) : A(A) {
      if (A.isAppxSymmetric()) symmetric = true;
      rows = A.rows();
      cols = A.cols();
    };

    // INITIALIZER
    void initRandom(int k, int seed = 0) : k(k) {
      #ifdef USING_RCPP
      // set seed
      if (seed > 0) {
        Rcpp::Environment base_env("package:base");
        Rcpp::Function set_seed_r = base_env["set.seed"];
        set_seed_r(std::floor(std::fabs(seed)));
      }
      // use R RNG to initialize W
      Rcpp::NumericVector R_RNG_random_values = Rcpp::runif(k * rows);
      w = Eigen::MatrixXd(k, nrow);
      for (int i = 0, index = 0; i < k; ++i)
        for (int j = 0; j < rows; ++j, ++index)
          w(i, j) = random_values[index];
      #else
      w = Eigen::MatrixXd::Random(k, rows);
      w.array().abs();
      #endif
    }

    // SETTERS: model matrices
    void W(Eigen::MatrixXd w_) {
      if (w_.cols() != rows)
        throw std::invalid_argument("number of columns in 'w' was not equal to number of rows in 'A'");
      if (is_initialized) {
        if (w_.rows() != h.rows())
          throw std::invalid_argument("cannot initialize 'w' when 'h' is initialized and contains a different number of rows.");
        if (d.size() != w_.rows())
          throw std::invalid_argument("cannot initialize 'w' when 'd' is initialized and contains a different rank.");
      } else {
        is_initialized = true;
        h = Eigen::MatrixXd::Zero(w_.rows(), cols);
        d = Eigen::VectorXd::Ones(w_.rows());
      }
      w = w_;
      k = w.rows();
    }
    void H(Eigen::MatrixXd& h_) {
      if (h_.cols() != cols)
        throw std::invalid_argument("number of columns in 'w' was not equal to number of rows in 'A'");
      if (is_initialized) {
        if (w.rows() != h_.rows())
          throw std::invalid_argument("cannot initialize 'h' when 'w' is initialized and contains a different number of rows.");
        if (d.size() != h_.rows())
          throw std::invalid_argument("cannot initialize 'h' when 'd' is initialized and contains a different rank.");
      } else {
        is_initialized = true;
        w = Eigen::MatrixXd::Zero(h_.rows(), rows);
        d = Eigen::VectorXd::Ones(h_.rows());
      }
      h = h_;
      k = h.rows();
    }
    void D(Eigen::VectorXd& d_) {
      if (is_initialized) {
        if (w.rows() != d_.size())
          throw std::invalid_argument("cannot initialize 'd' when 'w' is initialized and contains a different rank.");
        if (h.rows() != d_.size())
          throw std::invalid_argument("cannot initialize 'd' when 'h' is initialized and contains a different rank.");
      } else {
        is_initialized = true;
        h = Eigen::MatrixXd::Zero(d_.size(), cols);
        w = Eigen::MatrixXd::Zero(d_.size(), rows);
      }
      d = d_;
      k = d.size();
    }

    // SETTERS: parameters
    void tol(double tol) { target_tol = tol; }
    void max_iter(double max_iter) { max_iter_ = max_iter; }
    void verbose(bool verbose) {
      #ifndef USING_RCPP
      throw std::invalid_argument("Cannot use 'verbose' mode without '#define USING_RCPP'");
      #endif
      verbose_ = verbose;
    }
    void L1(Eigen::VectorXd alpha) {
      if (alpha.size() != 2)
        throw std::invalid_argument("L1 'alpha' must be a vector of length 2 for {w, h}");
      if (symmetric && alpha[0] != alpha[1])
        throw std::invalid_argument("for symmetric factorizations, L1 'alpha' must be the same for {w, h}");
      if (alpha[0] >= 1 || alpha[1] >= 1)
        throw std::invalid_argument("L1 'alpha' may not contain values >= 1");
      L1_w = alpha(0); L1_h = alpha(1);
    }
    void L2(Eigen::VectorXd alpha) {
      if (alpha.size() != 2)
        throw std::invalid_argument("L2 'alpha' must be a vector of length 2 for {w, h}");
      if (symmetric && alpha[0] != alpha[1])
        throw std::invalid_argument("for symmetric factorizations, L2 'alpha' must be the same for {w, h}");
      L2_w = alpha(0); L2_h = alpha(1);
    }
    void L1(double alpha) {
      if (alpha >= 1)
        throw std::invalid_argument("L1 'alpha' may not contain values >= 1");
      L1_w = alpha; L1_h = alpha;
    }
    void L2(double alpha) { L2_w = alpha; L2_h = alpha; }
    void mask_A(T& mask_matrix) {
      if (mask_matrix.rows() != rows || mask_matrix.cols() != cols)
        throw std::invalid_argument("masking matrix was not of the same dimensions as 'A'");
      if (symmetric)
        if (!mask_matrix.isAppxSymmetric())
          throw std::invalid_argument("since 'A' is symmetric the masking matrix must be symmetric.");
      // it may be useful to check that values in "mask_A" are strictly in the range (0, 1)
      // this would require splitting code out by typename T
      mask_A_ = mask_matrix;
      masking_A = true;
    }
    void mask_A() {
      masking_A = false;
    }
    void mask_A_zeros(bool mask_zeros) { mask_zeros = mask_zeros; }
    void mask_A_zeros() { mask_zeros = true; }
    void mask_W(Eigen::MatrixXd mask_matrix) {
      if (mask_matrix.cols() != rows || mask_matrix.rows() != k)
        throw std::invalid_argument("masking matrix was not of the same dimensions as 'w'");
      if (mask_matrix.maxCoeff() > 1 || mask_matrix.minCoeff() < 0)
        throw std::invalid_argument("masking matrix is not strictly in the range (0, 1).");
      mask_w = mask_matrix;
      if (symmetric) mask_h = mask_matrix;
      masking_w = true;
    }
    void mask_H(Eigen::MatrixXd mask_matrix) {
      if (mask_matrix.cols() != cols || mask_matrix.rows() != k)
        throw std::invalid_argument("masking matrix was not of the same dimensions as 'h'");
      if (mask_matrix.maxCoeff() > 1 || mask_matrix.minCoeff() < 0)
        throw std::invalid_argument("masking matrix is not strictly in the range (0, 1).");
      mask_h = mask_matrix;
      if (symmetric) mask_w = mask_h;
      masking_h = true;
    }
    void mask_W() { mask_w = Eigen::MatrixXd(); masking_w = false; }
    void mask_H() { mask_h = Eigen::MatrixXd(); masking_h = false; }
    void threads(int t) { threads_ = t; }
    void norm(std::string n) {
      if (n != "none" && n != "2norm" && n != "abs")
        throw std::invalid_argument("'norm' must be one of 'none', '2norm', or 'abs'");
      norm_ = n;
    }

    // GETTERS
    Eigen::MatrixXd W() { return w; }
    Eigen::MatrixXd H() { return h; }
    Eigen::VectorXd D() { return d; }
    double tol() { return tol_; }
    double n_iter() { return iter; }

    // OPERATIONS
    void order() {
      std::vector<int> indx = sort_index(d);
      w = reorder_rows(w, indx);
      d = reorder(d, indx);
      h = reorder_rows(h, indx);
    }

    // FITTERS
    void updateH() {
      if (!is_initialized)
        throw std::invalid_argument("Cannot update 'h' when the nmf model has not been initialized.");
      if (!mask_zeros)
        update(A, w, h, L1_h, L2_h, threads_, masking_h, mask_h, masking_A, mask_A_);
      else
        update_mask_zeros(A, w, h, L1_h, L2_h, threads_, masking_h, mask_h, masking_A, mask_A_);
    }

    void updateW() {
      if (!is_initialized)
        throw std::invalid_argument("Cannot update 'w' when the nmf model has not been initialized.");

      // if "A" is symmetric, update without transposition
      if (!symmetric) {
        if (!is_transposed) {
          At = A.transpose();
          if (masking_A) mask_At_ = mask_A_.transpose();
          is_transposed = true;
        }
        if (!mask_zeros)
          update(At, h, w, L1_w, L2_w, threads_, masking_w, mask_w, masking_A, mask_At_);
        else
          update_mask_zeros(At, h, w, L1_w, L2_w, threads_, masking_w, mask_w, masking_A, mask_At_);
      } else {
        if (!mask_zeros)
          update(A, h, w, L1_w, L2_w, threads_, masking_w, mask_w, masking_A, mask_At_);
        else
          update_mask_zeros(A, h, w, L1_w, L2_w, threads_, masking_w, mask_w, masking_A, mask_At_);
      }
    }

    void fit() {

      #ifdef USING_RCPP
      if (verbose_) Rprintf("\n%4s | %7s \n---------------\n", "iter", "tol");
      #endif

      if (!is_initialized)
        throw std::invalid_argument("Initialize the model prior to fitting.");

      tol_ = target_tol;
      if (w.isZero(1e-15)) {
        updateW();
        scaleW();
      }

      // alterating NNLS updates
      for (iter = 0; iter < max_iter_ && tol_ >= target_tol; ++iter) {
        w_it = w;
        updateH();
        scaleH();
        updateW();
        scaleW();
        tol_ = symmetric ? cor(w, h) : cor(w, w_it);
        #ifdef USING_RCPP
        if (verbose_) Rprintf("%4d | %7.2e\n", iter + 1, tol_);
        Rcpp::checkUserInterrupt();
        #endif
      }
      if (!masking_h && !masking_w) order();
    }

    // methods for crossValidate with initRandom
    //   methods for find_k using stochastic gradient descent

    // methods for multiple fits, pick best one

    // methods for calculating mean squared error
    double mse();
  };

  template <>
  double nmf<RcppSparse::Matrix>::mse() {
    Eigen::MatrixXd w0 = w.transpose();
    // multiply w by diagonal
    for (unsigned int i = 0; i < w0.cols(); ++i)
      for (unsigned int j = 0; j < w0.rows(); ++j)
        w0(j, i) *= d(i);

    // compute losses across all samples in parallel
    Eigen::ArrayXd losses(h.cols());
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(threads)
    #endif
    for (unsigned int i = 0; i < h.cols(); ++i) {
      Eigen::VectorXd wh_i = w0 * h.col(i);
      if (mask_zeros) {
        if (!masking_A) {
          for (RcppSparse::Matrix::InnerIterator iter(A, i); iter; ++iter)
            losses(i) += std::pow(wh_i(iter.row()) - iter.value(), 2);
        } else if (masking_A) {
          RcppSparse::Matrix::InnerIterator it_A(A, i), it_mask(mask_A_, i);
          while (it_A) {
            if (!it_mask || it_A.row() < it_mask.row()) {
              losses(i) += std::pow(wh_i(it_A.row()) - it_A.value(), 2);
              ++it_A;
            } else if (it_mask && it_A.row() == it_mask.row()) {
              if (it_mask.value() < 1)
                losses(i) += std::pow((wh_i(it_A.row()) - it_A.value()) * (1 - it_mask.value()), 2);
              ++it_mask; ++it_A;
            } else if (it_mask) {
              ++it_mask;
            }
          }
          for (RcppSparse::Matrix::InnerIterator iter(mask_A_, i); iter; ++iter)
            losses(i) -= (iter.row()) *= (1 - iter.value());
        }
      } else if (!mask_zeros) {
        for (RcppSparse::Matrix::InnerIterator iter(A, i); iter; ++iter)
          wh_i(iter.row()) -= iter.value();
        if (masking_A) {
          for (RcppSparse::Matrix::InnerIterator iter(mask_A_, i); iter; ++iter)
            wh_i(iter.row()) *= (1 - iter.value());
        }
        losses(i) += wh_i.array().square().sum();
      }
    }

    // divide total loss by number of applicable measurements
    if (masking_A) {
      double tot_mask = mask_A_.x.array().sum();
      return losses.sum() / ((h.cols() * w.cols()) - tot_mask);
    } else if (mask_zeros)
      return losses.sum() / A.x.size();
    return losses.sum() / ((h.cols() * w.cols()));
  };

  template <>
  double nmf<RcppSparse::Matrix>::mse() {
    if (!mask) Rcpp::stop("'mse_masked' can only be run when a masking matrix has been specified");

    Eigen::MatrixXd w0 = w.transpose();
    for (unsigned int i = 0; i < w0.cols(); ++i)
      for (unsigned int j = 0; j < w0.rows(); ++j)
        w0(j, i) *= d(i);

    Eigen::ArrayXd losses(h.cols());
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(threads) schedule(dynamic)
    #endif
    for (unsigned int i = 0; i < h.cols(); ++i) {
      std::vector<unsigned int> masked_rows = mask_matrix.nonzeroRowsInCol(i);
      if (masked_rows.size() > 0) {
        for (RcppSparse::MatrixInnerIteratorInRange iter(A, i, masked_rows); iter; ++iter) {
          losses(i) += std::pow((w0.row(iter.row()) * h.col(i)) - iter.value(), 2);
        }
        // get masked rows that are also zero in A.col(i)
        std::vector<unsigned int> zero_rows = A.zeroRowsInCol(i);
        std::vector<unsigned int> masked_zero_rows;
        std::set_intersection(zero_rows.begin(), zero_rows.end(),
                              masked_rows.begin(), masked_rows.end(),
                              std::back_inserter(masked_zero_rows));
        for (unsigned int it = 0; it < masked_zero_rows.size(); ++it)
          losses(i) += std::pow(w0.row(masked_zero_rows[it]) * h.col(i), 2);
      }
    }
    return losses.sum() / mask_matrix.i.size();
  };

}

//' Fast NMF
//'
//' A basic implementation of NMF
//'
//' @param sparse matrix of features in rows and samples in columns, of class \code{dgCMatrix}
//' @param w dense matrix of factors in rows and features in columns
//' @param tol tolerance of the fit
//' @param maxit maximum number of fitting iterations
//' @export
//[[Rcpp::export]]
Rcpp::List nmf_bench(RcppSparse::Matrix A, Eigen::MatrixXd w, const Eigen::VectorXd L1, const double tol = 1e-4, int maxit = 100, bool verbose = true) {
  RcppML::nmf<RcppSparse::Matrix> model(A);
  model.W(w);
  model.tol(tol);
  model.max_iter(maxit);
  model.verbose(verbose);
  model.fit();
  return Rcpp::List::create(Rcpp::Named("w") = model.W(), Rcpp::Named("d") = model.D(), Rcpp::Named("h") = model.H());
}