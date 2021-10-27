// zero-copy sparse matrix class for access by reference to R objects already in memory
// note that Eigen::SparseMatrix<double> requires a deep copy of R objects for use in C++

#ifndef RcppML_sparsematrix
#define RcppML_sparsematrix

#ifndef Rcpp_hpp
#include <Rcpp.h>
#endif

std::vector<unsigned int> get_range(Rcpp::IntegerVector& i, unsigned int begin, unsigned int end) {
  std::vector<unsigned int> v(end - begin);
  for (unsigned int j = 0, it = begin; it < end; ++j, ++it)
    v[j] = (unsigned int)i[it];
  return v;
}

// get values from 0 to max_size that are not in the "nonzeros" vector
std::vector<unsigned int> get_diff(std::vector<unsigned int>& nonzeros, const int max_size) {
  std::vector<unsigned int> all_vals(max_size);
  std::iota(all_vals.begin(), all_vals.end(), 0);
  std::vector<unsigned int> zeros;
  std::set_difference(all_vals.begin(), all_vals.end(), nonzeros.begin(), nonzeros.end(),
                      std::inserter(zeros, zeros.begin()));
  return zeros;
}

// remove intersecting values in "x" and "y" from "x"
// both "x" and "y" are sorted in ascending order
std::vector<unsigned int> remove_vals(std::vector<unsigned int>& x, const std::vector<unsigned int>& y) {
  unsigned int x_i = 0, y_i = 0, z_i = 0;
  std::vector<unsigned int> z = x;
  while (x_i < x.size()) {
    if (y_i > y.size() || y[y_i] > x[x_i]) {
      z[z_i] = x[x_i];
      ++x_i;
      ++z_i;
    } else if (y[y_i] == x[x_i]) {
      ++x_i;
      ++y_i;
    } else ++y_i;
  }
  z.resize(z_i);
  return z;
}

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
      InnerIterator(SparseMatrix& ptr, int col) :
        ptr(ptr), col_(col), index(ptr.p[col]), max_index(ptr.p[col + 1]) {}
      operator bool() const { return (index < max_index); }
      InnerIterator& operator++() { ++index; return *this; }
      const double& value() const { return ptr.x[index]; }
      int row() const { return ptr.i[index]; }
      int col() const { return col_; }

    private:
      SparseMatrix& ptr;
      int col_, index, max_index;
    };

    // equivalent to the "Forward Range" concept in two boost::ForwardTraversalIterator
    // iterates over non-zero values in `ptr.col(col)` at rows in `s`
    // `s` must be sorted in ascending order
    class InnerIteratorInRange {
    public:
      InnerIteratorInRange(SparseMatrix& ptr, int col, std::vector<unsigned int>& s) :
        ptr(ptr), s(s), col_(col), index(ptr.p[col]), max_index(ptr.p[col + 1] - 1), s_max_index(s.size() - 1) {
        // decrement max_index and s_max_index to last case where ptr.i intersects with s
        while ((unsigned int)ptr.i[max_index] != s[s_max_index] && max_index >= index && s_max_index >= 0)
          s[s_max_index] > (unsigned int)ptr.i[max_index] ? --s_max_index : --max_index;
        // increment index to the first case where ptr.i intersects with s
        while ((unsigned int)ptr.i[index] != s[s_index] && index <= max_index && s_index <= s_max_index)
          s[s_index] < (unsigned int)ptr.i[index] ? ++s_index : ++index;
      }
      operator bool() const { return (index <= max_index && s_index <= s_max_index); }
      InnerIteratorInRange& operator++() {
        ++index; ++s_index;
        while (index <= max_index && s_index <= s_max_index && (unsigned int)ptr.i[index] != s[s_index])
          s[s_index] < (unsigned int)ptr.i[index] ? ++s_index : ++index;
        return *this;
      }
      const double& value() const { return ptr.x[index]; }
      int row() const { return ptr.i[index]; }
      int col() const { return col_; }

    private:
      SparseMatrix& ptr;
      const std::vector<unsigned int>& s;
      int col_, index, max_index, s_max_index, s_index = 0, s_size;
    };

    // iterates over non-zero values in `ptr.col(col)` not at rows in `s_`
    // basically, turn this into InnerIteratorInRange by computing a vector `s` of non-intersecting
    //    non-zero rows in ptr at time of initialization 
    // `s` must be sorted in ascending order
    class InnerIteratorNotInRange {
    public:
      InnerIteratorNotInRange(SparseMatrix& ptr, int col, std::vector<unsigned int>& s_) :
        ptr(ptr), col_(col), index(ptr.p[col]), max_index(ptr.p[col + 1] - 1) {
        s = std::vector<unsigned int>(ptr.p[col_ + 1] - ptr.p[col]);
        if (s.size() > 0) {
          s = get_range(ptr.i, ptr.p[col_], ptr.p[col_ + 1]);
          if (s_.size() > 0)
            s = remove_vals(s, s_); // remove intersecting values in s_ from s
        }
        s_max_index = s.size() - 1;

        // decrement max_index and s_max_index to last case where ptr.i intersects with s
        while ((unsigned int)ptr.i[max_index] != s[s_max_index] && max_index >= index && s_max_index >= 0)
          s[s_max_index] > (unsigned int)ptr.i[max_index] ? --s_max_index : --max_index;
        // increment index to the first case where ptr.i intersects with s
        while ((unsigned int)ptr.i[index] != s[s_index] && index <= max_index && s_index <= s_max_index)
          s[s_index] < (unsigned int)ptr.i[index] ? ++s_index : ++index;
      }
      operator bool() const { return (index <= max_index && s_index <= s_max_index); }
      InnerIteratorNotInRange& operator++() {
        ++index; ++s_index;
        while (index <= max_index && s_index <= s_max_index && (unsigned int)ptr.i[index] != s[s_index])
          s[s_index] < (unsigned int)ptr.i[index] ? ++s_index : ++index;
        return *this;
      }
      const double& value() const { return ptr.x[index]; }
      int row() const { return ptr.i[index]; }
      int col() const { return col_; }

    private:
      SparseMatrix& ptr;
      std::vector<unsigned int> s;
      int col_, index, max_index, s_max_index, s_index = 0, s_size;
    };

    // const row iterator
    class InnerRowIterator {
    public:
      InnerRowIterator(SparseMatrix& ptr, int j) : ptr(ptr) {
        for (; index < ptr.Dim[1]; ++index)
          if (ptr.i[index] == j) break;
        for (int r = 0; r < ptr.i.size(); ++r)
          if (ptr.i[r] == j) max_index = r;
      }
      operator bool() const { return index <= max_index; };
      InnerRowIterator& operator++() {
        ++index;
        for (; index <= max_index; ++index)
          if (ptr.i[index] == row_) break;
        return *this;
      };
      int col() {
        int j = 0;
        for (; j < ptr.p.size(); ++j)
          if (ptr.p[j] > index) break;
        return j;
      };
      int row() { return row_; }
      double& value() const { return ptr.x[index]; };
    private:
      SparseMatrix& ptr;
      int row_ = 0, index = 0, max_index = 0;
    };

    // return indices of rows with nonzero values for a given column
    std::vector<unsigned int> nonzeroRowsInCol(int col) {
      std::vector<unsigned int> v = get_range(i, p[col], p[col + 1]);
      return v;
    }

    // return indices of rows with zeros values for a given column
    std::vector<unsigned int> zeroRowsInCol(int col) {
      // first get indices of non-zeros
      std::vector<unsigned int> nonzeros = get_range(i, p[col], p[col + 1]);
      std::vector<unsigned int> zeros = get_diff(nonzeros, Dim[0]);
      return zeros;
    }
    
    // number of nonzeros in a column
    unsigned int numNonzerosInCol(int col){
      return p[col + 1] - p[col];
    }

    // is approximately symmetric
    bool isAppxSymmetric() {
      if (Dim[0] == Dim[1]) {
        InnerIterator col_it(*this, 0);
        InnerRowIterator row_it(*this, 0);
        while (++col_it && ++row_it)
          if (col_it.value() != row_it.value())
            return false;
        return true;
      }
      return false;
    }

    // transpose
    SparseMatrix transpose() {
      Rcpp::S4 s(std::string("dgCMatrix"));
      s.slot("i") = i;
      s.slot("p") = p;
      s.slot("x") = x;
      s.slot("Dim") = Dim;
      Rcpp::Environment base = Rcpp::Environment::namespace_env("Matrix");
      Rcpp::Function t_r = base["t"];
      Rcpp::S4 At = t_r(Rcpp::_["x"] = s);
      return SparseMatrix(At);
    }

  };

  class SparsePatternMatrix {
  public:
    Rcpp::IntegerVector i, p, Dim;

    // Constructor from Rcpp::S4 object
    SparsePatternMatrix(Rcpp::S4 m) : i(m.slot("i")), p(m.slot("p")), Dim(m.slot("Dim")) {}
    SparsePatternMatrix() {}
    unsigned int rows() { return Dim[0]; }
    unsigned int cols() { return Dim[1]; }

    // return indices of rows with nonzero values for a given column
    std::vector<unsigned int> nonzerosInColumn(int col) {
      std::vector<unsigned int> v(p[col + 1] - p[col]);
      if (v.size() > 0) v = get_range(i, p[col], p[col + 1]);
      return v;
    }

    // const column iterator
    class InnerIterator {
    public:
      InnerIterator(SparsePatternMatrix& ptr, int col) :
        ptr(ptr), col_(col), index(ptr.p[col]), max_index(ptr.p[col + 1]) {}
      operator bool() const { return (index < max_index); }
      InnerIterator& operator++() { ++index; return *this; }
      int row() const { return ptr.i[index]; }
      int col() const { return col_; }

    private:
      SparsePatternMatrix& ptr;
      int col_, index, max_index;
    };

    // const row iterator
    class InnerRowIterator {
    public:
      InnerRowIterator(SparsePatternMatrix& ptr, int j) : ptr(ptr) {
        for (; index < ptr.Dim[1]; ++index)
          if (ptr.i[index] == j) break;
        for (int r = 0; r < ptr.i.size(); ++r)
          if (ptr.i[r] == j) max_index = r;
      }
      operator bool() const { return index <= max_index; };
      InnerRowIterator& operator++() {
        ++index;
        for (; index <= max_index; ++index)
          if (ptr.i[index] == row_) break;
        return *this;
      };
      int col() {
        int j = 0;
        for (; j < ptr.p.size(); ++j)
          if (ptr.p[j] > index) break;
        return j;
      };
      int row() { return row_; }
    private:
      SparsePatternMatrix& ptr;
      int row_ = 0, index = 0, max_index = 0;
    };

    // return indices of rows with nonzero values for a given column
    std::vector<unsigned int> nonzeroRowsInCol(int col) {
      std::vector<unsigned int> v = get_range(i, p[col], p[col + 1]);
      return v;
    }

    // return indices of rows with zeros values for a given column
    std::vector<unsigned int> zeroRowsInCol(int col) {
      std::vector<unsigned int> nonzeros = get_range(i, p[col], p[col + 1]);
      std::vector<unsigned int> zeros = get_diff(nonzeros, Dim[0]);
      return zeros;
    }

    // number of nonzeros in a column
    unsigned int numNonzerosInCol(int col){
      return p[col + 1] - p[col];
    }
    
    // is approximately symmetric
    bool isAppxSymmetric() {
      if (Dim[0] == Dim[1]) {
        InnerIterator col_it(*this, 0);
        InnerRowIterator row_it(*this, 0);
        while (++col_it && ++row_it)
          if (col_it.row() != row_it.col())
            return false;
        return true;
      }
      return false;
    }

    // transpose
    SparsePatternMatrix transpose() {
      Rcpp::S4 s(std::string("ngCMatrix"));
      s.slot("i") = i;
      s.slot("p") = p;
      s.slot("Dim") = Dim;
      Rcpp::Environment base = Rcpp::Environment::namespace_env("Matrix");
      Rcpp::Function t_r = base["t"];
      Rcpp::S4 At = t_r(Rcpp::_["x"] = s);
      return SparsePatternMatrix(At);
    }
  };
}

#endif