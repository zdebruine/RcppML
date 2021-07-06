// This file is part of RcppML, an RcppEigen template library
//  for machine learning.
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

// helper functions for Eigen submatrix views
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

template <typename T>
void push_back(Eigen::Matrix<T, -1, 1>& m, T value) {
  const unsigned int len = m.size() + 1;
  m.conservativeResize(len, Eigen::NoChange);
  m[len] = value;
}

namespace RcppML {

  template <typename T>
  class CoeffMatrix {

  public:
    // CLASS CONSTRUCTOR
    CoeffMatrix(Eigen::Matrix<T, -1, -1>& a, bool nonneg = false, int L0 = -1, T L1 = 0, T L2 = 0, T RL2 = 0, int maxit = 100, T tol = 1e-8, std::string L0_method = "ggperm") :
      nonneg(nonneg), rank(a.rows()), L0(L0), maxit(maxit), tol(tol), L1(L1), L2(L2), RL2(RL2), L0_method(L0_method), a(a) {

      if (L0 >= rank || L0 <= 0) L0 = -1;

      // add L2/Ridge regularization penalty
      if (L2 != 0)
        for (int i = 0; i < a.rows(); ++i)
          a(i, i) += L2;

      // add Pattern Extraction/inverse L2/trough/angular regularization penalty
      if (RL2 != 0) {
        for (int i = 0; i < a.rows(); ++i) {
          for (int j = 0; j < i; ++j) {
            if (i != j) {
              a(i, j) += RL2;
              a(j, i) += RL2;
            }
          }
        }
      }
      a_llt = a.llt();
    }

    // GENERAL SOLVER
    // Unconstrained or non-negative least squares solution for "x" in "a * x = b" where "a" is symmetric positive definite
    // generalized interface for private specialized subroutines
    template<typename Derived>
    Eigen::Matrix<T, -1, 1> solve(Eigen::MatrixBase<Derived>& b) {
      if (L1 != 0) for (int i = 0; i < rank; ++i) b[i] -= L1;

      if (L0 == 1) return solve_L0_1(b);

      // initialize with an unconstrained least squares solution
      Eigen::Matrix<T, -1, 1> x = a_llt.solve(b);

      // NNLS only: iterative feasible set selection while unconstrained solutions on the feasible set give any negative values
      if (nonneg) while ((x.array() < 0).any()) solve_nonzeros(b, x);

      // refine the solution by coordinate descent
      if (maxit > 0) solve_scd(b, x);

      if (L0 != -1 && cardinality(x) > L0) {
        if (L0_method == "exact") return solve_L0_exact(b);
        do reduce_rank(b, x); while (cardinality(x) > L0);
      }
      return x;
    }

    // COORDINATE DESCENT GENERAL SOLVER
    template<typename Derived, typename OtherDerived>
    void solve(Eigen::MatrixBase<Derived>& b, Eigen::MatrixBase<OtherDerived>& x) {
      if (L1 != 0) for (int i = 0; i < rank; ++i) b[i] -= L1;
      if (L0 == 1) x = solve_L0_1(b);
      else {
        solve_scd(b, x);
        if (L0 != -1 && cardinality(x) > L0) {
          if (L0_method == "exact") x = solve_L0_exact(b);
          do reduce_rank(b, x); while (cardinality(x) > L0);
        }
      }
    }

  private:
    // CLASS MEMBERS
    bool nonneg;
    int rank, L0, maxit;
    T tol, L1, L2, RL2;
    std::string L0_method;
    Eigen::Matrix<T, -1, -1> a;
    Eigen::LLT <Eigen::Matrix<T, -1, -1>> a_llt;

    // NUMBER OF NON-ZERO VALUES IN SOLUTION
    int cardinality(const Eigen::Matrix<T, -1, 1>& x) {
      int k = 0;
      for (int i = 0; i < rank; ++i) {
        if (x(i) != 0) ++k;
      }
      return k;
    }

    // ERROR/RESIDUALS OF SOLUTION
    // calculate the error of "a * x - b" for a given solution of "x" to the system "a * x = b"
    template <typename Derived>
    T calc_error(const Eigen::MatrixBase<Derived>& b, const Eigen::Matrix<T, -1, 1>& x) {
      T x_err = 0;
      Eigen::Matrix<T, -1, 1> b0 = a * x;
      for (int i = 0; i < x.size(); ++i) x_err += std::pow(b0[i] - b[i], 2);
      return x_err;
    }

    // ITERATIVE FEASIBLE SET SELECTION
    // find the best solution to the system "a * x = b" where only positive indices in initial "x" are considered
    template<typename Derived>
    void solve_nonzeros(const Eigen::MatrixBase<Derived>& b, Eigen::Matrix<T, -1, 1>& x) {
      // get indices of positive values in x (i.e. the "feasible set")
      int n_gtz = 0;
      for (int i = 0; i < x.size(); ++i) if (x(i) > 0) ++n_gtz;
      Eigen::VectorXi feasible_set(n_gtz);
      Eigen::Matrix<T, -1, 1> bsub(n_gtz);
      for (int i = 0, j = 0; i < x.size(); ++i) {
        if (x(i) > 0) {
          feasible_set(j) = i;
          bsub(j) = b(i);
          ++j;
        }
      }

      // subset "a" at feasible set indices
      Eigen::Matrix<T, -1, -1> asub = submat(a, feasible_set, feasible_set);

      // solve for x given "a" and "b" at feasible set indices and map the solution back to a full-length vector "x"
      Eigen::Matrix<T, -1, 1> xsub = asub.llt().solve(bsub);
      x.setZero();
      for (int i = 0; i < n_gtz; ++i) x(feasible_set(i)) = xsub(i);
    }

    // SEQUENTIAL COORDINATE DESCENT
    // Refines "x" in-place
    template<typename Derived, typename OtherDerived>
    void solve_scd(Eigen::MatrixBase<Derived>& b, Eigen::MatrixBase<OtherDerived>& x) {
      Eigen::Matrix<T, -1, 1> gradient = a * x;
      for (int i = 0; i < rank; ++i) gradient(i) -= b(i);

      T tol_ = tol + 1;
      for (int it = 0; it < maxit && tol_ > tol; ++it) {
        tol_ = 0;
        for (int i = 0; i < rank; ++i) {
          // update this solution coefficient to minimize difference to the gradient
          T tmp = x(i) - gradient(i) / a(i, i);

          // apply non-negativity constraint if applicable
          if (nonneg && tmp < 0) tmp = 0;
          T diff = tmp - x(i);

          if (tmp != x(i)) {
            // update the gradient based on the residuals between the current coefficient and the optimal coefficient
            for (int j = 0; j < rank; ++j) gradient(j) += a(j, i) * diff;

            // calculate tolerance of the update, used as stopping criteria
            T tmp_tol = 2 * std::abs(diff) / (tmp + x(i));
            if (tmp_tol > tol_) tol_ = tmp_tol;
            x(i) = tmp;
          }
        }
      }
    }

    // EXACT L0 = 1 SOLUTION
    template <typename Derived>
    Eigen::Matrix<T, -1, 1> solve_L0_1(const Eigen::MatrixBase<Derived>& b) {

      // find the exact least squares solution for each index if all other indices were zero
      Eigen::Matrix<T, -1, 1> solutions = b;
      for (int i = 0; i < solutions.size(); ++i) {
        if (nonneg && solutions(i) < 0) solutions(i) = 0;
        else solutions(i) /= a(i, i);
      }

      // calculate the squared error for each possible solution
      Eigen::Matrix<T, -1, 1> errors(solutions.size());
      for (int i = 0; i < solutions.size(); ++i)
        for (int j = 0; j < b.size(); ++j)
          errors(i) += std::pow(b(j) - a(j, i) * solutions(i), 2);

      // find the index of the solution with the least error (index_min)
      T min_val = errors(0);
      int index_min = 0;
      for (int i = 0; i < errors.size(); ++i) {
        if (errors(i) < min_val) {
          min_val = errors(i);
          index_min = i;
        }
      }
      Eigen::Matrix<T, -1, 1> x(b.size());
      x(index_min) = solutions(index_min);
      return x;
    }

    // EXACT L0 SOLUTION FOR THE GENERAL CASE
    template<typename Derived>
    Eigen::Matrix<T, -1, 1> solve_L0_exact(const Eigen::MatrixBase<Derived>& b) {

      int iter = 0;
      std::vector<bool> v(rank);
      std::fill(v.end() - L0, v.end(), true);
      Eigen::VectorXi feasible_set(L0);
      Eigen::Matrix<T, -1, 1> x_new(rank), x(rank), bsub(L0);
      T best_err = b.sum();
      do {
        ++iter;
        if (iter % 1000 == 0) Rcpp::checkUserInterrupt();

        // solve unconstrained least squares for the given feasible set permutation
        int n_gtz = 0;
        for (int i = 0; i < rank; ++i) {
          if (v[i]) {
            feasible_set(n_gtz) = i;
            bsub(n_gtz) = b(i);
            ++n_gtz;
          }
        }
        Eigen::Matrix<T, -1, -1> asub = submat(a, feasible_set, feasible_set);
        Eigen::Matrix<T, -1, 1> xsub = asub.llt().solve(bsub);
        x_new.setZero();
        for (int i = 0; i < n_gtz; ++i) x_new(feasible_set(i)) = xsub(i);

        // check error
        T curr_err = calc_error(b, x_new);
        if (curr_err < best_err) {
          best_err = curr_err;
          x = x_new;
        }
      } while (std::next_permutation(v.begin(), v.end()));
      return x;
    }

    // L0 SOLVER
    // reduce the cardinality/rank of the solution by 1
    template<typename Derived, typename OtherDerived>
    void reduce_rank(Eigen::MatrixBase<Derived>& b, Eigen::MatrixBase<OtherDerived>& x) {
      Eigen::Matrix<T, -1, 1> x_new(rank);

      // bind the non-zero coefficient to zero that increases error least when bound to zero
      T x_err = 0;
      for (int i = 0; i < rank; ++i) {
        if (x(i) != 0) {
          x_new = x;
          x_new(i) = 0;
          solve_nonzeros(b, x_new);
          if (nonneg) for (int j = 0; j < rank; ++j) if (x_new[j] < 0) x_new[j] = 0;
          Eigen::Matrix<T, -1, 1> b0 = a * x_new;
          T x_new_err = 0;
          for (int j = 0; j < rank; ++j) x_new_err += std::pow(b0[j] - b[j], 2);
          if (x_new_err < x_err || x_err == 0) {
            x_err = x_new_err;
            x = x_new;
          }
        }
      }

      // exhaustive permutation
      // try permuting feasible set var "i" for active set var "j" for all i/j pairs
      // repeat permutation until no permutations which reduce error are discovered
      if (L0_method == "eperm") {
        for (;;) {
          T x_new_err = x_err;
          for (int i = 0; i < rank; ++i) {
            if (x(i) != 0) { // variable i is in the feasible set
              for (int j = 0; j < rank; ++j) {
                if (x(j) == 0) { // variable j is not in the feasible set
                  Eigen::Matrix<T, -1, 1> x_permutation = x;
                  x_permutation(i) = 0;
                  x_permutation(j) = 1;
                  solve_nonzeros(b, x_permutation);
                  if (nonneg) for (int k = 0; k < rank; ++k) if (x_permutation[k] < 0) x_permutation[k] = 0;
                  T x_permutation_err = calc_error(b, x_permutation);
                  if (x_permutation_err < x_new_err) {
                    // i/j feasible set permutation reduced error
                    x_new = x_permutation;
                    x_new_err = x_permutation_err;
                  }
                }
              }
            }
          }
          if (x_new_err == x_err) break;
          x = x_new;
          x_err = calc_error(b, x);
        }
      } else if (L0_method == "ggperm") {
        // gradient-guided permutation
        for (;;) {
          T x_new_err = x_err;
          // consider removing each variable from the feasible set and finding the best possible substitution using coordinate descent
          Eigen::Matrix<T, -1, 1> x_new(rank);
          for (int i = 0; i < rank; ++i) {
            if (x(i) != 0) { // variable i is in the feasible set
              // compute gradient when removing variable i
              Eigen::Matrix<T, -1, 1> x_permutation = x;
              x_permutation(i) = 0;
              Eigen::Matrix<T, -1, 1> b0 = a * x_permutation;
              for (int j = 0; j < rank; ++j) b0[j] -= b[0];

              // find coefficient with the greatest value in the gradient
              int max_index = 0;
              T max_value = b0[0];
              for (int j = 0; j < rank; ++j) if (b0[j] > max_value) max_index = j;

              // if the variable with the largest error is not the one we just removed, substitute and calculate change in error
              if (max_index != i) {
                x_permutation(max_index) = 1;
                solve_nonzeros(b, x_permutation);
                T x_permutation_err = calc_error(b, x_permutation);

                // this substitution improved the error, keep it and set it as the new benchmark
                if (x_permutation_err < x_new_err) {
                  x_new = x_permutation;
                  x_new_err = x_permutation_err;
                }
              }
            }
          }
          if (x_new_err < x_err) {
            x = x_new;
            x_err = x_new_err;
          } else break;
        }
      }
    }
  };

  template <typename T>
  class CoeffMatrix2 {

  public:
    // CLASS CONSTRUCTOR
    CoeffMatrix2(Eigen::Matrix<T, 2, 2>& a, bool nonneg = false, int L0 = 2) :
      a(a), nonneg(nonneg), denom(a(0, 0)* a(1, 1) - a(0, 1) * a(0, 1)) {}

    // 2-variable systems of equations in the form "a * x = b" may be solved directly as follows:
    //
    // [a11 a12] [x1] = [b1]    x1 = (a22b1 - a12b2) / (a11a22 - a12a12)
    // [a12 a22] [x2] = [b2]    x2 = (a11b2 - a12b1) / (a11a22 - a12a12)
    //
    // Note that the denominator (a11a22 - a12a12) is constant and depends only on "a" and therefore can be calculated
    // just once for many systems in a linear model projection

    // 2-variable NNLS in-place in "b"
    template <typename Derived>
    void solveInPlace(Eigen::MatrixBase<Derived>& b) {
      if (nonneg) {
        T a01b1 = a(0, 1) * b(1);
        T a11b0 = a(1, 1) * b(0);
        if (a11b0 < a01b1) {
          b(0) = 0;
          b(1) /= a(1, 1);
        } else {
          T a01b0 = a(0, 1) * b(0);
          T a00b1 = a(0, 0) * b(1);
          if (a00b1 < a01b0) {
            b(0) /= a(0, 0);
            b(1) = 0;
          } else {
            b(0) = (a11b0 - a01b1) / denom;
            b(1) = (a00b1 - a01b0) / denom;
          }
        }
      } else {
        T b0 = b[0];
        b(0) = (a(1, 1) * b(0) - a(0, 1) * b(1)) / denom;
        b(1) = (a(0, 0) * b(1) - a(0, 1) * b(0)) / denom;
      }
    }

    template <typename Derived, typename OtherDerived>
    void solveInPlace(Eigen::MatrixBase<Derived>& b, Eigen::MatrixBase<OtherDerived>& x) {
      if (nonneg) {
        T a01b1 = a(0, 1) * b(1);
        T a11b0 = a(1, 1) * b(0);
        if (a11b0 < a01b1) {
          x(0) = 0;
          x(1) = b(1) / a(1, 1);
        } else {
          T a01b0 = a(0, 1) * b(0);
          T a00b1 = a(0, 0) * b(1);
          if (a00b1 < a01b0) {
            x(0) = b(0) / a(0, 0);
            x(1) = 0;
          } else {
            x(0) = (a11b0 - a01b1) / denom;
            x(1) = (a00b1 - a01b0) / denom;
          }
        }
      } else {
        x(0) = (a(1, 1) * b(0) - a(0, 1) * b(1)) / denom;
        x(1) = (a(0, 0) * b(1) - a(0, 1) * b(0)) / denom;
      }
    }

    template <typename Derived>
    Eigen::MatrixBase<Derived> solve(Eigen::MatrixBase<Derived>& b) {
      Eigen::MatrixBase<Derived> x = b;
      solveInPlace(x);
      return x;
    }

    // 2-variable least squares solution with L0 = 1
    template <typename Derived>
    Eigen::Matrix<T, 2, 1> solve_L0(Eigen::MatrixBase<Derived>& b) {
      Eigen::Matrix<T, 2, 1> x;
      x << b(0) / a(0, 0), b(1) / a(1, 1);
      if (nonneg) {
        if (x(0) < 0) x(0) = 0;
        if (x(1) < 0) x(1) = 0;
      }
      T err0 = std::pow(b(0) - a(0, 0) * x(0), 2) + std::pow(b(1) - a(1, 0) * x(0), 2);
      T err1 = std::pow(b(0) - a(0, 1) * x(1), 2) + std::pow(b(1) - a(1, 1) * x(1), 2);
      if (err0 < err1) x(1) = 0;
      else x(0) = 0;
      return x;
    }

    template <typename Derived>
    void solveInPlace_L0(Eigen::MatrixBase<Derived>& b) {
      Eigen::Matrix<T, 2, 1> x = solve_L0(b);
      b(0) = x(0);
      b(1) = x(1);
    }

  private:
    // CLASS MEMBERS
    Eigen::Matrix<T, 2, 2> a;
    bool nonneg;
    T denom;

  };

  template <typename T>
  class MatrixFactorization {
  public:
    MatrixFactorization(Eigen::SparseMatrix<T>& A, int rank) : A(A), k(rank) {
      W = Eigen::Matrix<T, -1, -1>(k, A.rows());
      H = Eigen::Matrix<T, -1, -1>(k, A.cols());
      H.setRandom();
      H.cwiseAbs();
      D = Eigen::Matrix<T, -1, 1>(k);
      D.setOnes();
      threads = Eigen::nbThreads();
    }
    void nonneg() { nonneg_W = true; nonneg_H = true; }
    void nonnegW() { nonneg_W = true; }
    void nonnegH() { nonneg_H = true; }
    void nonneg(bool n) { nonneg_W = n; nonneg_H = n; }
    void nonnegW(bool n) { nonneg_W = n; }
    void nonnegH(bool n) { nonneg_H = n; }
    void L0(int n) { L0_W = n; L0_H = n; if (n >= k || n <= 0) { L0_W = -1; L0_H = -1; } }
    void L0W(int n) { L0_W = n; if (n >= k || n <= 0) L0_W = -1; }
    void L0H(int n) { L0_H = n; if (n >= k || n <= 0) L0_H = -1; }
    void L1(T n) { L1_W = n; L1_H = n; }
    void L1W(T n) { L1_W = n; }
    void L1H(T n) { L1_H = n; }
    void L2(T n) { L2_W = n; L2_H = n; }
    void L2W(T n) { L2_W = n; }
    void L2H(T n) { L2_H = n; }
    void RL2(T n) { RL2_W = n; RL2_H = n; }
    void RL2W(T n) { RL2_W = n; }
    void RL2H(T n) { RL2_H = n; }
    void withDiagonalization() { fit_with_diag = true; }
    void withNoDiagonalization() { fit_with_diag = false; }
    void L0method(std::string s) { L0_method = s; }
    void solveMaxit(int n) { solve_maxit = n; }
    void solveTol(T n) { solve_tol = n; }
    void trackLoss() { calc_losses = true; }
    void trackLoss(bool n) { calc_losses = n; }

    int rank() { return k; }
    void MatrixW(Eigen::Matrix<T, -1, -1>& matrixW) { W = matrixW; }
    void MatrixH(Eigen::Matrix<T, -1, -1>& matrixH) { H = matrixH; }
    void VectorD(Eigen::Matrix<T, -1, 1>& vectorD) { D = vectorD; }
    Eigen::Matrix<T, -1, 1> VectorD() { return D; }
    Eigen::Matrix<T, -1, -1> MatrixW() { return W.transpose(); }
    Eigen::Matrix<T, -1, -1> MatrixH() { return H; }
    T loss() { if (loss == -1) calcLoss(); return model_loss; }

    // given A = WH and solving for "W", this is a slower alternative to solve for Wt but does not require transposition of A
    // thus, this solves A = WH for "W" rather than At = HWt for W.
    void solveWInPlace() {
      Eigen::Matrix<T, -1, -1> a_temp = H * H.transpose();
      CoeffMatrix<T> a(a_temp, nonneg_W, L0_W, L1_W, L2_W, RL2_W, solve_maxit, solve_tol, L0_method);
      // compute right-hand side of linear system, "b" in place in w as b = h * A.row(i)
      #pragma omp parallel for num_threads(threads) schedule(dynamic)
      for (int i = 0; i < A.cols(); ++i)
        for (typename Eigen::SparseMatrix<T>::InnerIterator it(A, i); it; ++it)
          for (int j = 0; j < k; ++j) W(j, it.row()) += it.value() * H(j, i);

      #pragma omp parallel for num_threads(threads) schedule(dynamic)
      for (int i = 0; i < A.rows(); ++i) {
        Eigen::Matrix<T, -1, 1> x = W.col(i);
        W.col(i) = a.solve(x);
      }
    }

    // Solves At = W * H for W, requires a transposition of sparse matrix "A"
    void solveW() {
      if (!At_) {
        At = A.transpose();
        At_ = true;
      }
      Eigen::Matrix<T, -1, -1> a_temp = H * H.transpose();
      CoeffMatrix<T> a(a_temp, nonneg_W, L0_W, L1_W, L2_W, RL2_W, solve_maxit, solve_tol, L0_method);
      #pragma omp parallel for num_threads(threads) schedule(dynamic)
      for (int i = 0; i < At.cols(); ++i) {
        Eigen::Matrix<T, -1, 1> b(k);
        for (typename Eigen::SparseMatrix<T>::InnerIterator it(At, i); it; ++it)
          for (int j = 0; j < k; ++j) b(j) += it.value() * H(j, it.row());
        W.col(i) = a.solve(b);
      }
    }

    // Solves A = Wt * H for H, where Wt.rows() and H.rows() = k
    // Equivalent to At = H * Wt
    void solveH() {
      Eigen::Matrix<T, -1, -1> a_temp = W * W.transpose();
      CoeffMatrix<T> a(a_temp, nonneg_H, L0_H, L1_H, L2_H, RL2_H, solve_maxit, solve_tol, L0_method);
      #pragma omp parallel for num_threads(threads) schedule(dynamic)
      for (int i = 0; i < A.cols(); ++i) {
        // compute right-hand side of linear system, "b" as b = w * A.col(i)
        Eigen::Matrix<T, -1, 1> b(k);
        for (typename Eigen::SparseMatrix<T>::InnerIterator it(A, i); it; ++it)
          for (int j = 0; j < k; ++j) b(j) += it.value() * W(j, it.row());
        H.col(i) = a.solve(b);
      }
    }

    double getLoss() {
      calcLoss();
      return model_loss;
    }

  private:
    Eigen::SparseMatrix<T>& A;
    Eigen::SparseMatrix<T> At;
    Eigen::Matrix<T, -1, -1> W, H;
    Eigen::Matrix<T, -1, 1> D;
    bool nonneg_W = false, nonneg_H = false, At_ = false, fit_with_diag = true, calc_losses = false;
    int k, solve_maxit = 100, L0_W = -1, L0_H = -1, threads;
    T solve_tol = 1e-8, L1_W = 0, L1_H = 0, L2_W = 0, L2_H = 0, RL2_W = 0, RL2_H = 0, model_loss = -1, model_tol = -1;
    Eigen::Matrix<T, -1, 1> tols, losses;
    std::string L0_method = "ggperm";

    void calcLoss() {
      T model_loss = 0;
      Eigen::Matrix<T, -1, -1> Wt = W.transpose();
      #pragma omp parallel for num_threads(threads) schedule(dynamic)
      for (int i = 0; i < A.cols(); ++i) {
        Eigen::Matrix<T, -1, 1> WH_i = Wt * H.col(i);
        for (typename Eigen::SparseMatrix<T>::InnerIterator it(A, i); it; ++it)
          WH_i(it.row()) -= it.value();
        for (unsigned int j = 0; j < WH_i.size(); ++j)
          model_loss += std::pow(WH_i(j), 2);
      }
      model_loss /= (A.cols() * A.rows());
    }

  };

} // end namespace RcppML

//[[Rcpp::export]]
Eigen::VectorXd Rcpp_solve_vector_double(Eigen::MatrixXd& a, Eigen::VectorXd& b, bool nonneg = false, int L0 = -1, double L1 = 0,
                                         double L2 = 0, double RL2 = 0, int maxit = 100, double tol = 1e-9, std::string L0_method = "ggperm") {
  RcppML::CoeffMatrix<double> model(a, nonneg, L0, L1, L2, RL2, maxit, tol, L0_method);
  return model.solve(b);
}

//[[Rcpp::export]]
Eigen::MatrixXd Rcpp_solve_matrix_double(Eigen::MatrixXd& a, Eigen::MatrixXd& b, bool nonneg = false, int L0 = -1, double L1 = 0,
                                         double L2 = 0, double RL2 = 0, int maxit = 100, double tol = 1e-9, std::string L0_method = "ggperm",
                                         const int threads = 0) {

  Eigen::MatrixXd x(b.rows(), b.cols());
  RcppML::CoeffMatrix<double> model(a, nonneg, L0, L1, L2, RL2, maxit, tol, L0_method);

  #pragma omp parallel for num_threads(threads) schedule(dynamic)
  for (int i = 0; i < b.cols(); ++i) {
    Eigen::VectorXd b_vec = b.col(i);
    x.col(i) = model.solve(b_vec);
  }

  return x;
}

//[[Rcpp::export]]
Eigen::VectorXd Rcpp_solve_cd_vector_double(Eigen::MatrixXd& a, Eigen::VectorXd& b, Eigen::VectorXd x, bool nonneg = false, int L0 = -1, double L1 = 0,
                                            double L2 = 0, double RL2 = 0, int maxit = 100, double tol = 1e-9, std::string L0_method = "ggperm") {
  RcppML::CoeffMatrix<double> model(a, nonneg, L0, L1, L2, RL2, maxit, tol, L0_method);
  model.solve(b, x);
  return x;
}

//[[Rcpp::export]]
Eigen::MatrixXd Rcpp_solve_cd_matrix_double(Eigen::MatrixXd& a, Eigen::MatrixXd& b, Eigen::MatrixXd x, bool nonneg = false, int L0 = -1, double L1 = 0,
                                            double L2 = 0, double RL2 = 0, int maxit = 100, double tol = 1e-9, std::string L0_method = "ggperm",
                                            const int threads = 0) {

  RcppML::CoeffMatrix<double> model(a, nonneg, L0, L1, L2, RL2, maxit, tol, L0_method);

  #pragma omp parallel for num_threads(threads) schedule(dynamic)
  for (int i = 0; i < b.cols(); ++i) {
    Eigen::VectorXd b_vec = b.col(i);
    Eigen::VectorXd x_vec = x.col(i);
    model.solve(b_vec, x_vec);
  }

  return x;
}


//[[Rcpp::export]]
Eigen::MatrixXd Rcpp_projectW_double(Eigen::SparseMatrix<double>& A, Eigen::MatrixXd& W, bool nonneg = false, int L0 = -1, double L1 = 0,
                                     double L2 = 0, double RL2 = 0, int maxit = 100, double tol = 1e-9, std::string L0_method = "ggperm",
                                     const int threads = 0) {
  RcppML::MatrixFactorization<double> model(A, W.rows());
  model.MatrixW(W);
  model.L0H(L0);
  model.L1H(L1);
  model.L2H(L2);
  model.solveMaxit(maxit);
  model.solveTol(tol);
  model.L0method(L0_method);
  model.nonnegH(nonneg);
  Eigen::setNbThreads(threads);
  model.solveH();
  return model.MatrixH();
}

//[[Rcpp::export]]
Eigen::MatrixXd Rcpp_projectH_double(Eigen::SparseMatrix<double>& A, Eigen::MatrixXd& H, bool nonneg = false, int L0 = -1, double L1 = 0,
                                     double L2 = 0, double RL2 = 0, int maxit = 100, double tol = 1e-9, std::string L0_method = "ggperm",
                                     const int threads = 0) {
  RcppML::MatrixFactorization<double> model(A, H.rows());
  model.MatrixH(H);
  model.L0W(L0);
  model.L1W(L1);
  model.L2W(L2);
  model.solveMaxit(maxit);
  model.solveTol(tol);
  model.L0method(L0_method);
  model.nonnegW(nonneg);
  Eigen::setNbThreads(threads);
  model.solveWInPlace();
  return model.MatrixW();
}

//[[Rcpp::export]]
Eigen::VectorXf Rcpp_solve_vector_float(Eigen::MatrixXf& a, Eigen::VectorXf& b, bool nonneg = false, int L0 = -1, float L1 = 0,
                                        float L2 = 0, float RL2 = 0, int maxit = 100, float tol = 1e-9, std::string L0_method = "ggperm") {
  RcppML::CoeffMatrix<float> model(a, nonneg, L0, L1, L2, RL2, maxit, tol, L0_method);
  return model.solve(b);
}

//[[Rcpp::export]]
Eigen::VectorXf Rcpp_solve_cd_vector_float(Eigen::MatrixXf& a, Eigen::VectorXf& b, Eigen::VectorXf x, bool nonneg = false, int L0 = -1, float L1 = 0,
                                           float L2 = 0, float RL2 = 0, int maxit = 100, float tol = 1e-9, std::string L0_method = "ggperm") {
  RcppML::CoeffMatrix<float> model(a, nonneg, L0, L1, L2, RL2, maxit, tol, L0_method);
  model.solve(b, x);
  return x;
}

//[[Rcpp::export]]
Eigen::MatrixXf Rcpp_solve_matrix_float(Eigen::MatrixXf& a, Eigen::MatrixXf& b, bool nonneg = false, int L0 = -1, float L1 = 0,
                                        float L2 = 0, float RL2 = 0, int maxit = 100, float tol = 1e-9, std::string L0_method = "ggperm",
                                        const int threads = 0) {

  Eigen::MatrixXf x(b.rows(), b.cols());
  RcppML::CoeffMatrix<float> model(a, nonneg, L0, L1, L2, RL2, maxit, tol, L0_method);

  #pragma omp parallel for num_threads(threads) schedule(dynamic)
  for (int i = 0; i < b.cols(); ++i) {
    Eigen::VectorXf b_vec = b.col(i);
    x.col(i) = model.solve(b_vec);
  }

  return x;
}

//[[Rcpp::export]]
Eigen::MatrixXf Rcpp_solve_cd_matrix_float(Eigen::MatrixXf& a, Eigen::MatrixXf& b, Eigen::MatrixXf x, bool nonneg = false, int L0 = -1, float L1 = 0,
                                           float L2 = 0, float RL2 = 0, int maxit = 100, float tol = 1e-9, std::string L0_method = "ggperm",
                                           const int threads = 0) {

  RcppML::CoeffMatrix<float> model(a, nonneg, L0, L1, L2, RL2, maxit, tol, L0_method);

  #pragma omp parallel for num_threads(threads) schedule(dynamic)
  for (int i = 0; i < b.cols(); ++i) {
    Eigen::VectorXf b_vec = b.col(i);
    Eigen::VectorXf x_vec = x.col(i);
    model.solve(b_vec, x_vec);
  }

  return x;
}

//[[Rcpp::export]]
Eigen::MatrixXf Rcpp_projectW_float(Eigen::SparseMatrix<float>& A, Eigen::MatrixXf& W, bool nonneg = false, int L0 = -1, float L1 = 0,
                                    float L2 = 0, float RL2 = 0, int maxit = 100, float tol = 1e-9, std::string L0_method = "ggperm",
                                    const int threads = 0) {
  RcppML::MatrixFactorization<float> model(A, W.rows());
  model.MatrixW(W);
  model.L0H(L0);
  model.L1H(L1);
  model.L2H(L2);
  model.solveMaxit(maxit);
  model.solveTol(tol);
  model.L0method(L0_method);
  model.nonnegH(nonneg);
  Eigen::setNbThreads(threads);
  model.solveH();
  return model.MatrixH();
}

//[[Rcpp::export]]
Eigen::MatrixXf Rcpp_projectH_float(Eigen::SparseMatrix<float>& A, Eigen::MatrixXf& H, bool nonneg = false, int L0 = -1, float L1 = 0,
                                    float L2 = 0, float RL2 = 0, int maxit = 100, float tol = 1e-9, std::string L0_method = "ggperm",
                                    const int threads = 0) {
  RcppML::MatrixFactorization<float> model(A, H.rows());
  model.MatrixH(H);
  model.L0W(L0);
  model.L1W(L1);
  model.L2W(L2);
  model.solveMaxit(maxit);
  model.solveTol(tol);
  model.L0method(L0_method);
  model.nonnegW(nonneg);
  Eigen::setNbThreads(threads);
  model.solveWInPlace();
  return model.MatrixW();
}