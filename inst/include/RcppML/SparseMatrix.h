#ifndef Rcpp_SparseMatrix
#define Rcpp_SparseMatrix
#include <RcppCommon.h>

namespace Rcpp {
class SparseMatrix;
}  // namespace Rcpp

// forward declare Rcpp::as<> Exporter
namespace Rcpp {
namespace traits {
template <>
class Exporter<Rcpp::SparseMatrix>;
}  // namespace traits
}  // namespace Rcpp

#include <Rcpp.h>

namespace Rcpp {

// this class is provided for consistency with Eigen::SparseMatrix, but using
// R objects (i.e. Rcpp::NumericVector, Rcpp::IntegerVector) that comprise Matrix::dgCMatrix in R.
// R objects are pointers to underlying memory-mapped SEXP vectors, and are usable in C++ without any
// affect on performance. Thus, this class achieves zero-copy access to R sparse matrix objects, with equal
// performance for read-only column iteration (`InnerIterator`) like `Eigen::SparseMatrix<double>`.
//
// The class is designed with an `InnerIterator` class that exactly mimics `Eigen::SparseMatrix<T>::InnerIterator`,
// and also contains `.rows()` and `.cols()` member functions. This allows it to substitute for `Eigen::SparseMatrix`
// in all SLAM routines.
class SparseMatrix {
   public:
    NumericVector x;
    IntegerVector i, p, Dim;

    // constructors
    SparseMatrix(NumericVector x, IntegerVector i, IntegerVector p, IntegerVector Dim) : x(x), i(i), p(p), Dim(Dim) {}
    SparseMatrix(const S4& s) {
        if (!s.hasSlot("x") || !s.hasSlot("p") || !s.hasSlot("i") || !s.hasSlot("Dim"))
            throw std::invalid_argument("Cannot construct SparseMatrix from this S4 object");
        x = s.slot("x");
        i = s.slot("i");
        p = s.slot("p");
        Dim = s.slot("Dim");
    }
    SparseMatrix() {}

    unsigned int rows() { return Dim[0]; }
    unsigned int cols() { return Dim[1]; }

    // const column iterator
    class InnerIterator {
       public:
        InnerIterator(SparseMatrix& ptr, int col) : ptr(ptr), col_(col), index(ptr.p[col]), max_index(ptr.p[col + 1]) {}
        operator bool() const { return (index < max_index); }
        InnerIterator& operator++() {
            ++index;
            return *this;
        }
        double& value() const { return ptr.x[index]; }
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
        InnerIteratorInRange(SparseMatrix& ptr, int col, std::vector<unsigned int>& s) : ptr(ptr), s(s), col_(col), index(ptr.p[col]), max_index(ptr.p[col + 1] - 1), s_max_index(s.size() - 1) {
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
        SparseMatrix& ptr;
        const std::vector<unsigned int>& s;
        int col_, index, max_index, s_max_index, s_index = 0, s_size;
    };

    // const row iterator
    class InnerRowIterator {
       public:
        InnerRowIterator(SparseMatrix& ptr, int j) : ptr(ptr) {
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
        SparseMatrix& ptr;
        int row_ = 0, index = 0, max_index = 0;
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

    SparseMatrix clone() {
        NumericVector x_ = Rcpp::clone(x);
        IntegerVector i_ = Rcpp::clone(i);
        IntegerVector p_ = Rcpp::clone(p);
        IntegerVector Dim_ = Rcpp::clone(Dim);
        return SparseMatrix(x_, i_, p_, Dim_);
    }

    SparseMatrix transpose() {
        S4 s(std::string("dgCMatrix"));
        s.slot("i") = i;
        s.slot("p") = p;
        s.slot("x") = x;
        s.slot("Dim") = Dim;
        Environment base = Environment::namespace_env("Matrix");
        Function t_r = base["t"];
        S4 At = t_r(_["x"] = s);
        return SparseMatrix(At);
    };

    S4 wrap() {
        S4 s(std::string("dgCMatrix"));
        s.slot("x") = x;
        s.slot("i") = i;
        s.slot("p") = p;
        s.slot("Dim") = Dim;
        return s;
    }
};

namespace traits {
/* support for Rcpp::as */

// export a sparse matrix
template <>
class Exporter<Rcpp::SparseMatrix> {
    Rcpp::NumericVector x_;
    Rcpp::IntegerVector i, p, Dim;

   public:
    Exporter(SEXP x) {
        Rcpp::S4 s(x);
        if (!s.hasSlot("x") || !s.hasSlot("p") || !s.hasSlot("i") || !s.hasSlot("Dim"))
            throw std::invalid_argument("Cannot construct Rcpp::SparseMatrix from this S4 object");
        x_ = s.slot("x");
        i = s.slot("i");
        p = s.slot("p");
        Dim = s.slot("Dim");
    }

    Rcpp::SparseMatrix get() {
        return Rcpp::SparseMatrix(x_, i, p, Dim);
    }
};

}  // namespace traits
}  // namespace Rcpp

#endif