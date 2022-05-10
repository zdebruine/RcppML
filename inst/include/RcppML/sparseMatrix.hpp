// This file is part of RcppML, a Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU
// Public License

#ifndef RcppML_sparseMatrix
#define RcppML_sparseMatrix

#ifndef RcppCommon_h
#include <RcppCommon.h>
#endif

// forward declare classes
namespace RcppSparse {
class Matrix;
}  // namespace RcppSparse

// forward declare Rcpp::as<> Exporter
namespace Rcpp {

namespace traits {

template <>
class Exporter<RcppSparse::Matrix>;
}  // namespace traits
}  // namespace Rcpp

#include <Rcpp.h>

//[[Rcpp::plugins(openmp)]]
#ifdef _OPENMP
#include <omp.h>
#endif

namespace RcppSparse {
class Matrix {
public:
  // public member objects
  Rcpp::NumericVector x;
  Rcpp::IntegerVector i, p, Dim;
  
  // constructors
  Matrix(Rcpp::NumericVector x, Rcpp::IntegerVector i, Rcpp::IntegerVector p, Rcpp::IntegerVector Dim) : x(x), i(i), p(p), Dim(Dim) {}
  Matrix(const Rcpp::S4& s) {
    if (!s.hasSlot("x") || !s.hasSlot("p") || !s.hasSlot("i") || !s.hasSlot("Dim"))
      throw std::invalid_argument("Cannot construct RcppSparse::Matrix from this S4 object");
    x = s.slot("x");
    i = s.slot("i");
    p = s.slot("p");
    Dim = s.slot("Dim");
  }
  Matrix() {}
  
  unsigned int rows() { return Dim[0]; }
  unsigned int cols() { return Dim[1]; }
  unsigned int nrow() { return Dim[0]; }
  unsigned int ncol() { return Dim[1]; }
  unsigned int n_nonzero() { return x.size(); };
  Rcpp::NumericVector& nonzeros() { return x; };
  Rcpp::IntegerVector& innerIndexPtr() { return i; };
  Rcpp::IntegerVector& outerIndexPtr() { return p; };
  
  // create a deep copy of an R object
  Matrix clone() {
    Rcpp::NumericVector x_ = Rcpp::clone(x);
    Rcpp::IntegerVector i_ = Rcpp::clone(i);
    Rcpp::IntegerVector p_ = Rcpp::clone(p);
    Rcpp::IntegerVector Dim_ = Rcpp::clone(Dim);
    return Matrix(x_, i_, p_, Dim_);
  }
  
  // element lookup at specific index
  double at(int row, int col) const {
    for (int j = p[col]; j < p[col + 1]; ++j) {
      if (i[j] == row)
        return x[j];
      else if (i[j] > row)
        break;
    }
    return 0.0;
  }
  double operator()(int row, int col) const { return at(row, col); };
  double operator[](int index) const { return x[index]; };
  
  // subview clones
  Rcpp::NumericVector operator()(int row, Rcpp::IntegerVector& col) {
    Rcpp::NumericVector res(col.size());
    for (int j = 0; j < col.size(); ++j) res[j] = at(row, col[j]);
    return res;
  };
  Rcpp::NumericVector operator()(Rcpp::IntegerVector& row, int col) {
    Rcpp::NumericVector res(row.size());
    for (int j = 0; j < row.size(); ++j) res[j] = at(row[j], col);
    return res;
  };
  Rcpp::NumericMatrix operator()(Rcpp::IntegerVector& row, Rcpp::IntegerVector& col) {
    Rcpp::NumericMatrix res(row.size(), col.size());
    for (int j = 0; j < row.size(); ++j)
      for (int k = 0; k < col.size(); ++k)
        res(j, k) = at(row[j], col[k]);
    return res;
  };
  
  // column access (copy)
  Rcpp::NumericVector col(int col) {
    Rcpp::NumericVector c(Dim[0], 0.0);
    for (int j = p[col]; j < p[col + 1]; ++j)
      c[i[j]] = x[j];
    return c;
  }
  Rcpp::NumericMatrix col(Rcpp::IntegerVector& c) {
    Rcpp::NumericMatrix res(Dim[0], c.size());
    for (int j = 0; j < c.size(); ++j) {
      res.column(j) = col(c[j]);
    }
    return res;
  }
  
  // row access (copy)
  Rcpp::NumericVector row(int row) {
    Rcpp::NumericVector r(Dim[1], 0.0);
    for (int col = 0; col < Dim[1]; ++col) {
      for (int j = p[col]; j < p[col + 1]; ++j) {
        if (i[j] == row)
          r[col] = x[j];
        else if (i[j] > row)
          break;
      }
    }
    return r;
  }
  Rcpp::NumericMatrix row(Rcpp::IntegerVector& r) {
    Rcpp::NumericMatrix res(r.size(), Dim[1]);
    for (int j = 0; j < r.size(); ++j) {
      res.row(j) = row(r[j]);
    }
    return res;
  }
  
  // colSums and rowSums family
  Rcpp::NumericVector colSums() {
    Rcpp::NumericVector sums(Dim[1]);
    for (int col = 0; col < Dim[1]; ++col)
      for (int j = p[col]; j < p[col + 1]; ++j)
        sums(col) += x[j];
    return sums;
  }
  Rcpp::NumericVector rowSums() {
    Rcpp::NumericVector sums(Dim[0]);
    for (int col = 0; col < Dim[1]; ++col)
      for (int j = p[col]; j < p[col + 1]; ++j)
        sums(i[j]) += x[j];
    return sums;
  }
  Rcpp::NumericVector colMeans() {
    Rcpp::NumericVector sums = colSums();
    for (int i = 0; i < sums.size(); ++i)
      sums[i] = sums[i] / Dim[0];
    return sums;
  };
  Rcpp::NumericVector rowMeans() {
    Rcpp::NumericVector sums = rowSums();
    for (int i = 0; i < sums.size(); ++i)
      sums[i] = sums[i] / Dim[1];
    return sums;
  };
  
  // crossprod
  Rcpp::NumericMatrix crossprod() {
    Rcpp::NumericMatrix res(Dim[1], Dim[1]);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int col1 = 0; col1 < Dim[1]; ++col1) {
      for (int col2 = col1; col2 < Dim[1]; ++col2) {
        if (col1 == col2) {
          for (int j = p[col1]; j < p[col1 + 1]; ++j)
            res(col1, col1) += x[j] * x[j];
        } else {
          int col1_ind = p[col1], col1_max = p[col1 + 1];
          int col2_ind = p[col2], col2_max = p[col2 + 1];
          while (col1_ind < col1_max && col2_ind < col2_max) {
            int row1 = i[col1_ind];
            int row2 = i[col2_ind];
            if (row1 == row2) {
              res(col1, col2) += x[col1_ind] * x[col2_ind];
              ++col1_ind;
              ++col2_ind;
            } else if (row1 < row2) {
              do {
                ++col1_ind;
              } while (i[col1_ind] < row2 && col1_ind < col1_max);
            } else if (row2 < row1) {
              do {
                ++col2_ind;
              } while (i[col2_ind] < row1 && col2_ind < col2_max);
            }
          }
          res(col2, col1) = res(col1, col2);
        }
      }
    }
    return res;
  }
  
  // return indices of rows with nonzero values for a given column
  // this function is similar to Rcpp::Range, but unlike Rcpp::Range it is thread-safe
  std::vector<unsigned int> InnerIndices(int col) {
    std::vector<unsigned int> v(p[col + 1] - p[col]);
    for (int j = 0, it = p[col]; it < p[col + 1]; ++j, ++it)
      v[j] = (unsigned int)i[it];
    return v;
  }
  
  // return indices of rows with zeros values for a given column
  std::vector<unsigned int> emptyInnerIndices(int col) {
    // first get indices of non-zeros
    std::vector<unsigned int> nonzeros = InnerIndices(col);
    std::vector<unsigned int> all_vals(Dim[0]);
    std::iota(all_vals.begin(), all_vals.end(), 0);
    std::vector<unsigned int> zeros;
    std::set_difference(all_vals.begin(), all_vals.end(), nonzeros.begin(), nonzeros.end(),
                        std::inserter(zeros, zeros.begin()));
    return zeros;
  }
  
  // const column iterator
  class InnerIterator {
  public:
    InnerIterator(Matrix& ptr, int col) : ptr(ptr), col_(col), index(ptr.p[col]), max_index(ptr.p[col + 1]) {}
    operator bool() const { return (index < max_index); }
    InnerIterator& operator++() {
      ++index;
      return *this;
    }
    const double& value() const { return ptr.x[index]; }
    int row() const { return ptr.i[index]; }
    int col() const { return col_; }
    
  private:
    Matrix& ptr;
    int col_, index, max_index;
  };
  
  // equivalent to the "Forward Range" concept in two boost::ForwardTraversalIterator
  // iterates over non-zero values in `ptr.col(col)` at rows in `s`
  // `s` must be sorted in ascending order
  class InnerIteratorInRange {
  public:
    InnerIteratorInRange(Matrix& ptr, int col, std::vector<unsigned int>& s) : ptr(ptr), s(s), col_(col), index(ptr.p[col]), max_index(ptr.p[col + 1] - 1), s_max_index(s.size() - 1) {
      // decrement max_index and s_max_index to last case where ptr.i intersects with s
      while ((unsigned int)ptr.i[max_index] != s[s_max_index] && max_index >= index && s_max_index >= 0)
        s[s_max_index] > (unsigned int)ptr.i[max_index] ? --s_max_index : --max_index;
      // increment index to the first case where ptr.i intersects with s
      while ((unsigned int)ptr.i[index] != s[s_index] && index <= max_index && s_index <= s_max_index)
        s[s_index] < (unsigned int)ptr.i[index] ? ++s_index : ++index;
    }
    operator bool() const { return (index <= max_index && s_index <= s_max_index); }
    InnerIteratorInRange& operator++() {
      ++index;
      ++s_index;
      while (index <= max_index && s_index <= s_max_index && (unsigned int)ptr.i[index] != s[s_index])
        s[s_index] < (unsigned int)ptr.i[index] ? ++s_index : ++index;
      return *this;
    }
    const double& value() const { return ptr.x[index]; }
    int row() const { return ptr.i[index]; }
    int col() const { return col_; }
    
  private:
    Matrix& ptr;
    const std::vector<unsigned int>& s;
    int col_, index, max_index, s_max_index, s_index = 0, s_size;
  };
  
  // iterates over non-zero values in ptr.col(col) not at rows in s_
  // basically, turn this into InnerIteratorInRange by computing a vector `s` of non-intersecting
  //    non-zero rows in ptr at time of initialization
  // s must be sorted in ascending order
  class InnerIteratorNotInRange {
  public:
    InnerIteratorNotInRange(Matrix& ptr, int col, std::vector<unsigned int>& s_) : ptr(ptr), col_(col), index(ptr.p[col]), max_index(ptr.p[col + 1] - 1) {
      s = std::vector<unsigned int>(ptr.p[col_ + 1] - ptr.p[col]);
      if (s.size() > 0) {
        for (int j = 0, it = ptr.p[col_]; it < ptr.p[col_ + 1]; ++j, ++it)
          s[j] = (unsigned int)ptr.i[it];
        if (s_.size() > 0) {
          // remove intersecting values in s_ from s
          unsigned int si = 0, s_i = 0, z_i = 0;
          std::vector<unsigned int> z = s;
          while (si < s.size()) {
            if (s_i > s_.size() || s_[s_i] > s[si]) {
              z[z_i] = s[si];
              ++si;
              ++z_i;
            } else if (s_[s_i] == s[si]) {
              ++si;
              ++s_i;
            } else
              ++s_i;
          }
          z.resize(z_i);
          s = z;
        }
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
      ++index;
      ++s_index;
      while (index <= max_index && s_index <= s_max_index && (unsigned int)ptr.i[index] != s[s_index])
        s[s_index] < (unsigned int)ptr.i[index] ? ++s_index : ++index;
      return *this;
    }
    const double& value() const { return ptr.x[index]; }
    int row() const { return ptr.i[index]; }
    int col() const { return col_; }
    
  private:
    Matrix& ptr;
    std::vector<unsigned int> s;
    int col_, index, max_index, s_max_index, s_index = 0, s_size;
  };
  
  // const row iterator
  class InnerRowIterator {
  public:
    InnerRowIterator(Matrix& ptr, int j) : ptr(ptr) {
      for (; index < ptr.Dim[1]; ++index) {
        if (ptr.i[index] == j) break;
      }
      for (int r = 0; r < ptr.i.size(); ++r)
        if (ptr.i[r] == j) max_index = r;
    }
    operator bool() const { return index <= max_index; };
    InnerRowIterator& operator++() {
      ++index;
      for (; index <= max_index; ++index) {
        if (ptr.i[index] == row_) break;
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
    int row() { return row_; }
    double& value() const { return ptr.x[index]; };
    
  private:
    Matrix& ptr;
    int row_ = 0, index = 0, max_index = 0;
  };
  
  // number of nonzeros in a column
  unsigned int InnerNNZs(int col) {
    return p[col + 1] - p[col];
  }
  
  // is approximately symmetric
  bool isAppxSymmetric() {
    if (Dim[0] == Dim[1]) {
      InnerIterator col_it(*this, 0);
      InnerRowIterator row_it(*this, 0);
      while (++col_it && ++row_it) {
        if (col_it.value() != row_it.value())
          return false;
      }
      return true;
    }
    return false;
  }
  
  Matrix transpose() {
    Rcpp::S4 s(std::string("dgCMatrix"));
    s.slot("i") = i;
    s.slot("p") = p;
    s.slot("x") = x;
    s.slot("Dim") = Dim;
    Rcpp::Environment base = Rcpp::Environment::namespace_env("Matrix");
    Rcpp::Function t_r = base["t"];
    Rcpp::S4 At = t_r(Rcpp::_["x"] = s);
    return Matrix(At);
  };
  
  Rcpp::S4 wrap() {
    Rcpp::S4 s(std::string("dgCMatrix"));
    s.slot("x") = x;
    s.slot("i") = i;
    s.slot("p") = p;
    s.slot("Dim") = Dim;
    return s;
  }
};
}  // namespace RcppSparse

namespace Rcpp {
namespace traits {

template <>
class Exporter<RcppSparse::Matrix> {
  Rcpp::NumericVector x_;
  Rcpp::IntegerVector i, p, Dim;
  
public:
  Exporter(SEXP x) {
    Rcpp::S4 s(x);
    if (!s.hasSlot("x") || !s.hasSlot("p") || !s.hasSlot("i") || !s.hasSlot("Dim"))
      throw std::invalid_argument("Cannot construct RcppSparse::Matrix from this S4 object");
    x_ = s.slot("x");
    i = s.slot("i");
    p = s.slot("p");
    Dim = s.slot("Dim");
  }
  
  RcppSparse::Matrix get() {
    return RcppSparse::Matrix(x_, i, p, Dim);
  }
};

}  // namespace traits
}  // namespace Rcpp

#endif
