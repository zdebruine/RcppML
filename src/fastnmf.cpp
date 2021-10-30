// This file is a standalone RcppEigen demo for basic minimal NMF

//[[Rcpp::depends(RcppEigen)]]
//[[Rcpp::plugins(openmp)]]
#include "RcppEigen.h"
#include "omp.h"

namespace Rcpp {
class SpMat {
public:
  Rcpp::IntegerVector i, p, Dim;
  Rcpp::NumericVector x;

  SpMat(Rcpp::S4 m) : i(m.slot("i")), p(m.slot("p")), Dim(m.slot("Dim")), x(m.slot("x")) {}
  int rows() { return Dim[0]; }
  int cols() { return Dim[1]; }

  class ColIterator {
  public:
    ColIterator(SpMat& ptr, int col) :
    ptr(ptr), col_(col), index(ptr.p[col]), max_index(ptr.p[col + 1]) {}
    operator bool() const { return (index < max_index); }
    ColIterator& operator++() { ++index; return *this; }
    const double& value() const { return ptr.x[index]; }
    int row() const { return ptr.i[index]; }

  private:
    SpMat& ptr;
    int col_, index, max_index;
  };

  // transpose
  SpMat transpose() {
    Rcpp::S4 s(std::string("dgCMatrix"));
    s.slot("i") = i;
    s.slot("p") = p;
    s.slot("x") = x;
    s.slot("Dim") = Dim;
    Rcpp::Environment base = Rcpp::Environment::namespace_env("Matrix");
    Rcpp::Function t_r = base["t"];
    Rcpp::S4 At = t_r(Rcpp::_["x"] = s);
    return SpMat(At);
  }

};
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

// Non-Negative Least Squares solver
// solve ax = b given "a", "b", and h.col(sample) giving "x", subject to non-negativity. Coordinate descent.
inline void solve_nnls(Eigen::MatrixXd& a, Eigen::VectorXd& b, Eigen::MatrixXd& h, const int sample) {
  double tol = 1;
  for (int it = 0; it < 100 && (tol / b.size()) > 1e-8; ++it) {
    tol = 0;
    for (int i = 0; i < h.rows(); ++i) {
      double diff = b(i) / a(i, i);
      if (-diff > h(i, sample)) {
        if (h(i, sample) != 0) {
          b -= a.col(i) * -h(i, sample);
          tol = 1;
          h(i, sample) = 0;
        }
      } else if (diff != 0) {
        h(i, sample) += diff;
        b -= a.col(i) * diff;
        tol += std::abs(diff / (h(i, sample) + 1e-15));
      }
    }
  }
}

// solve for 'h' given sparse 'A' in 'A = wh'
void update(Rcpp::SpMat& A, const Eigen::MatrixXd& w, Eigen::MatrixXd& h) {

  Eigen::MatrixXd a = w * w.transpose();
  a.diagonal().array() += 1e-15;
  Eigen::LLT<Eigen::MatrixXd, 1> a_llt = a.llt();
#pragma omp parallel for
  for (int i = 0; i < h.cols(); ++i) {
    Eigen::VectorXd b = Eigen::VectorXd::Zero(h.rows());
    for (Rcpp::SpMat::ColIterator it(A, i); it; ++it)
      b += it.value() * w.col(it.row());
    h.col(i) = a_llt.solve(b);
    if ((h.col(i).array() < 0).any()) {
      b -= a * h.col(i);
      solve_nnls(a, b, h, i);
    }
  }
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
Rcpp::List fastnmf(const Rcpp::S4& data, Eigen::MatrixXd w, const double tol = 1e-4, int maxit = 100) {

  Rcpp::SpMat A(data);
  Rcpp::SpMat At = A.transpose();

  if(w.cols() != A.rows()) Rcpp::stop("number of rows in 'A' and columns in 'w' are not equivalent");
  Eigen::MatrixXd h = Eigen::MatrixXd::Zero(w.rows(), A.cols());
  Eigen::VectorXd d = Eigen::VectorXd::Zero(w.rows());

  double tol_ = 1;

  while (maxit-- > 0 && tol_ > tol) {
    Eigen::MatrixXd w_it = w;
    update(A, w, h);

    // scale H
    d = h.rowwise().sum();
    d.array() += 1e-15;
    for (int i = 0; i < h.rows(); ++i)
      for (int j = 0; j < h.cols(); ++j)
        h(i, j) /= d(i);

    update(At, h, w);

    // scale W
    d = w.rowwise().sum();
    d.array() += 1e-15;
    for (int i = 0; i < w.rows(); ++i)
      for (int j = 0; j < w.cols(); ++j)
        w(i, j) /= d(i);

    tol_ = cor(w, w_it);
  }

  return Rcpp::List::create(Rcpp::Named("w") = w.transpose(), Rcpp::Named("d") = d, Rcpp::Named("h") = h);
}
