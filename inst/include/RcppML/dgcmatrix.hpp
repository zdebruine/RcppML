// zero-copy sparse matrix class for access by reference to R objects already in memory
// note that Eigen::SparseMatrix<double> requires a deep copy of R objects for use in C++

#ifndef RcppML_dgcmatrix
#define RcppML_dgcmatrix

#ifndef Rcpp_hpp
#include <Rcpp.h>
#endif

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

#endif