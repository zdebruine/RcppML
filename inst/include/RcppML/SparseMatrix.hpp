// zero-copy sparse matrix class for access by reference to R objects already in memory
// note that Eigen::SparseMatrix<double> requires a deep copy of R objects for use in C++

#ifndef RcppML_sparsematrix
#define RcppML_sparsematrix

#ifndef Rcpp_hpp
#include <Rcpp.h>
#endif

namespace RcppML {
  class SparseMatrix {
  public:
    Rcpp::IntegerVector i, p, Dim;
    Rcpp::NumericVector x;

    // Constructor from Rcpp::S4 object
    SparseMatrix(Rcpp::S4 m) : i(m.slot("i")), p(m.slot("p")), Dim(m.slot("Dim")), x(m.slot("x")) {}
    SparseMatrix() {}
    unsigned int rows() { return Dim[0]; }
    unsigned int cols() { return Dim[1]; }

    // const column iterator
    class InnerIterator {
    public:
      InnerIterator(SparseMatrix& ptr, int col) : ptr(ptr) { index = ptr.p[col]; max_index = ptr.p[col + 1]; }
      operator bool() const { return (index < max_index); }
      InnerIterator& operator++() { ++index; return *this; }
      const double& value() const { return ptr.x[index]; }
      int row() const { return ptr.i[index]; }

    private:
      SparseMatrix& ptr;
      int index, max_index;
    };

    // equivalent to the "Forward Range" concept in two boost::ForwardTraversalIterator
    class InnerRangedIterator {
    public:
      InnerRangedIterator(SparseMatrix& ptr, int col, const std::vector<unsigned int>& s) : ptr(ptr), s(s) {
        index = ptr.p[col];
        max_index = ptr.p[col + 1];
        s_size = s.size();
        // increment index to the first case where ptr.i intersects with s
        while (index < max_index && s_index < s_size && (unsigned int)ptr.i[index] != s[s_index])
          s[s_index] < (unsigned int)ptr.i[index] ? ++s_index : ++index;
      }
      operator bool() const { return (index < max_index&& s_index < s_size); }
      InnerRangedIterator& operator++() {
        ++index; ++s_index;
        while (index < max_index && s_index < s_size && (unsigned int)ptr.i[index] != s[s_index])
          s[s_index] < (unsigned int)ptr.i[index] ? ++s_index : ++index;
        return *this;
      }
      const double& value() const { return ptr.x[index]; }
      int row() const { return ptr.i[index]; }
      int rangedRow() const { return s_index; }

    private:
      SparseMatrix& ptr;
      const std::vector<unsigned int>& s;
      int index, max_index, s_index = 0, s_size;
    };

    // const row iterator
    class InnerRowIterator {
    public:
      InnerRowIterator(SparseMatrix& ptr, int j) : ptr(ptr) {
        index = 0, max_index = 0;
        for (; index < ptr.Dim[1]; ++index) {
          if (ptr.i[index] == j) break;
        }
        for (int r = 0; r < ptr.i.size(); ++r) {
          if (ptr.i[r] == j) max_index = r;
        }
      }
      operator bool() const { return index <= max_index; };
      InnerRowIterator& operator++() {
        ++index;
        for (; index <= max_index; ++index) {
          if (ptr.i[index] == row) break;
        }
        return *this;
      };
      int col() {
        int j = 0;
        for (; j < ptr.p.size(); ++j) {
          if (ptr.p[j] > index) break;
        }
        return j;
      };
      double& value() const { return ptr.x[index]; };
    private:
      SparseMatrix& ptr;
      int row = 0, index, max_index;
    };

    // return indices of rows with nonzero values for a given column
    Rcpp::IntegerVector nonzeroRows(int col) {
      return i[Rcpp::Range(p[col], p[col + 1] - 1)];
    }

    // transpose
    SparseMatrix t() {
      Rcpp::S4 s(std::string("dgCMatrix"));
      s.slot("i") = i;
      s.slot("p") = p;
      s.slot("x") = x;
      s.slot("Dim") = Dim;
      Rcpp::Environment base("package:Matrix");
      Rcpp::Function t_r = base["t"];
      Rcpp::S4 At = t_r(Rcpp::_["x"] = s);
      return SparseMatrix(At);
    }
  };
}
#endif