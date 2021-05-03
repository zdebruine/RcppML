// 4/15/2020 Zach DeBruine (zach.debruine@vai.org)
// Please raise issues on github.com/zdebruine/RcppSparse/issues
//
// This header file extends the Rcpp namespace with a dgCMatrix sparse matrix class
// This class is documented at github.com/zdebruine/RcppSparse

//[[Rcpp::plugins(openmp)]]
#include <omp.h>
#include <rcpp.h>

namespace Rcpp {

    class mat {
    public:
        Rcpp::NumericVector x;
        int n_cols, n_rows;

        mat(Rcpp::NumericMatrix m) { x = Rcpp::NumericVector(m); n_cols = m.cols(); n_rows = m.rows(); }
        mat(int nr, int nc) { x = Rcpp::NumericVector(nr * nc, 0.0); n_cols = nr; n_rows = nc; }
        double& operator()(int r, int c) { return x[r * c + r]; }
        const double& operator()(int r, int c) const { return x[r * c + r]; }
        double& operator[](int i) { return x[i]; }
        const double& operator[](int i) const { return x[i]; }
        int cols() { return n_cols; }
        int rows() { return n_rows; }
        Rcpp::NumericVector column(int i) { Rcpp::NumericVector c(n_rows); for (int i = 0, ind = i * n_rows; i < n_rows; ++i) c[i] = x[ind + i]; return c; }
        Rcpp::NumericVector row(int i) { Rcpp::NumericVector c(n_cols); for (int i = 0; i < n_cols; ++i) c[i] = x[i * n_rows + i]; return c; }
    };

    template <> mat as(SEXP m) { return mat(m); }
    template <> SEXP wrap(const mat& m) { Rcpp::NumericVector v = m.x; v.attr("dim") = Dimension(m.n_rows, m.n_cols); return v; }
    
    class dgCMatrix;
    class dtpMatrix;
    
    typedef NumericVector vec;
    typedef IntegerVector ivec;
    typedef dgCMatrix SparseMatrix, spmat;
    typedef dtpMatrix TriangularMatrix, tmat;
    
    class dgCMatrix {
    public:
        ivec i, p, Dim;
        vec x;

        // constructors
        dgCMatrix(ivec& A_i, ivec& A_p, vec& A_x, int nrow) : i(A_i), p(A_p), x(A_x) { Dim = ivec::create(nrow, A_p.size() - 1); }
        dgCMatrix(ivec& A_i, ivec& A_p, vec& A_x, int nrow, int ncol) : i(A_i), p(A_p), x(A_x) { Dim = ivec::create(nrow, ncol); }
        dgCMatrix(S4 m) : i(m.slot("i")), p(m.slot("p")), x(m.slot("x")) { Dim = m.slot("Dim"); }

        int nrow() { return Dim[0]; } int rows() { return Dim[0]; }
        int ncol() { return Dim[1]; } int cols() { return Dim[1]; }
        int size() { return Dim[0] * Dim[1]; }
        int n_nonzero() { return x.size(); } int n_nonzeros() { return x.size(); }

        template<typename T> class itr {
        public:
            int row_;

            // constructors
            itr(dgCMatrix& ptr, int ind, int min_ind, int max_ind) : row_(-1), ptr(ptr), indx(ind), max_index(max_ind), min_index(min_ind) {}
            itr(dgCMatrix& ptr, int ind, int min_ind, int max_ind, int row_) : row_(row_), ptr(ptr), indx(ind), max_index(max_ind), min_index(min_ind) {}
            itr(const itr<T>& o) = default;
            itr<T>& operator=(const itr<T>& o) = default;

            // logical comparisons
            operator bool() const { return (indx <= max_index) && (indx >= min_index); }
            bool operator==(const itr<T>& o) const { return (indx == o.indx); }
            bool operator!=(const itr<T>& o) const { return (indx != o.indx); }
            bool operator<(const itr<T>& o) const { return (indx < o.indx); }
            bool operator>(const itr<T>& o) const { return (indx > o.indx); }
            bool operator<=(const itr<T>& o) const { return (indx <= o.indx); }
            bool operator>=(const itr<T>& o) const { return (indx >= o.indx); }
            bool operator==(int o) const { return (indx == o); }
            bool operator!=(int o) const { return (indx != o); }
            bool operator<(int o) const { return (indx < o); }
            bool operator>(int o) const { return (indx > o); }
            bool operator<=(int o) const { return (indx <= o); }
            bool operator>=(int o) const { return (indx >= o); }

            // increment/decrement
            itr<T>& operator++() { ++indx; if (row_ != -1) while (indx < max_index && ptr.i[indx] != row_) ++indx; return *this; }
            itr<T> operator++(int) { auto tmp(*this); ++indx; if (row_ != -1) while (indx < max_index && ptr.i[indx] != row_) ++indx; return tmp; }
            itr<T>& operator--() { --indx; if (row_ != -1) while (indx > min_index && ptr.i[indx] != row_) --indx; return *this; }
            itr<T> operator--(int) { auto tmp(*this); --indx; if (row_ != -1) while (indx > min_index && ptr.i[indx] != row_) --indx; return tmp; }

            // iterator member functions
            T& value() { return ptr.x[indx]; };
            const T& value() const { return ptr.x[indx]; };
            int index() { return indx; };
            int row() { return ptr.i[indx]; };
            int col() { int j = 0; for (; j < ptr.p.size(); ++j) if (ptr.p[j] > indx) break; return --j; };
            int max_index_in_range() { return max_index; }
            int min_index_in_range() { return min_index; }
            int n_nonzero() { return max_index - min_index; }
            int size() { if (row_ == -1) return ptr.rows(); return ptr.cols(); }

            // arithmetic operations across the iterator range
            T crossprod() { T sum = 0; for (auto tmp = *this; tmp; ++tmp) sum += tmp.value() * tmp.value(); return sum; }
            T crossprod(vec& o) {
                T sum = 0;
                if (row_ == -1) for (auto it = *this; it && it.col() < o.size(); ++it) sum += it.value() * o[it.col()];
                else for (auto it = *this; it && it.row() < o.size(); ++it) sum += it.value() * o[it.row()];
                return sum;
            }
            T crossprod(itr<T> o) {
                T sum = 0;
                if (row_ == -1) {
                    for (auto it1 = *this, it2 = o; it1 && it2; ++it1, ++it2) {
                        while (it1.row() != it2.row() && it1 && it2) { if (it1.row() < it2.row()) ++it1; else ++it2; if (!it1 || !it2) return sum; }
                        sum += (it1.value()) * (it2.value());
                    }
                } else {
                    for (auto it1 = *this, it2 = o; it1 && it2; ++it1, ++it2) {
                        while (it1.col() != it2.col() && it1 && it2) { if (it1.col() < it2.col()) ++it1; else ++it2; if (!it1 || !it2) return sum; }
                        sum += (it1.value()) * (it2.value());
                    }
                } return sum;
            }

            T sum() { T sum = 0; for (auto tmp = *this; tmp; ++tmp) sum += tmp.value(); return sum; }

            // convert to dense
            vec dense() {
                vec v(size());
                if (row_ == -1) for (auto it = *this; it; ++it) v[it.row()] = it.value();
                else for (auto it = *this; it; ++it) v[it.col()] = it.value();
                return v;
            }
        private:
            dgCMatrix& ptr;
            int indx, max_index, min_index;
        };
        typedef itr<double> iterator;
        typedef itr<const double> const_iterator;
        typedef itr<double> col_iterator;
        typedef itr<const double> const_col_iterator;
        typedef itr<double> row_iterator;
        typedef itr<const double> const_row_iterator;
        iterator begin() { return iterator(*this, 0, 0, x.size() - 1); }
        const_iterator const_begin() { return const_iterator(*this, 0, 0, x.size() - 1); }
        iterator end() { return iterator(*this, x.size() - 1, 0, x.size() - 1); }
        const_iterator const_end() { return const_iterator(*this, x.size() - 1, 0, x.size() - 1); }
        iterator index(int r, int c) { int j = p[c]; while (i[j] < r) ++j; return iterator(*this, j, 0, x.size() - 1); }
        const_iterator const_index(int r, int c) { int j = p[c]; while (i[j] < r) ++j; return const_iterator(*this, j, 0, x.size() - 1); }
        col_iterator begin_col(int c) { return col_iterator(*this, p[c], p[c], p[c + 1] - 1); }
        const_col_iterator const_begin_col(int c) { return const_col_iterator(*this, p[c], p[c], p[c + 1] - 1); }
        col_iterator end_col(int c) { return col_iterator(*this, p[c + 1] - 1, p[c], p[c + 1] - 1); }
        const_col_iterator const_end_col(int c) { return const_col_iterator(*this, p[c + 1] - 1, p[c], p[c + 1] - 1); }
        row_iterator begin_row(int r) {
            int j = 0; while (i[j] != r && j < i.size()) ++j;
            if (j == i.size()) return row_iterator(*this, 0, 1, 0, r);
            int k = i.size() - 1; while (i[k] != r && k > 0) --k;
            return row_iterator(*this, j, j, k, r);
        }
        row_iterator end_row(int r) {
            int j = 0; while (i[j] != r && j < i.size()) ++j;
            if (j == i.size()) return row_iterator(*this, 0, 1, 0, r);
            int k = i.size() - 1; while (i[k] != r && k > 0) --k;
            return row_iterator(*this, k, j, k, r);
        }
        col_iterator col(int c) { return col_iterator(*this, p[c], p[c], p[c + 1] - 1); }
        row_iterator row(int c) { return col_iterator(*this, p[c], p[c], p[c + 1] - 1); }

        // common element-wise mathematical functions
        void abs() { for (int j = 0; j < x.size(); ++j) x[j] = std::abs(x[j]); }
        void exp() { for (int j = 0; j < x.size(); ++j) x[j] = std::exp(x[j]); }
        void log() { for (int j = 0; j < x.size(); ++j) x[j] = std::log(x[j]); }
        void log10() { for (int j = 0; j < x.size(); ++j) x[j] = std::log10(x[j]); }
        void log2() { for (int j = 0; j < x.size(); ++j) x[j] = std::log2(x[j]); }
        void pow(double exp) { for (int j = 0; j < x.size(); ++j) x[j] = std::pow(x[j], exp); }
        void sqrt() { for (int j = 0; j < x.size(); ++j) x[j] = std::sqrt(x[j]); }
        void square() { for (int j = 0; j < x.size(); ++j) x[j] = x[j] * x[j]; }
        void cube() { for (int j = 0; j < x.size(); ++j) x[j] = x[j] * x[j] * x[j]; }
        void sin() { for (int j = 0; j < x.size(); ++j) x[j] = std::sin(x[j]); }
        void cos() { for (int j = 0; j < x.size(); ++j) x[j] = std::cos(x[j]); }
        void tan() { for (int j = 0; j < x.size(); ++j) x[j] = std::tan(x[j]); }
        void asin() { for (int j = 0; j < x.size(); ++j) x[j] = std::asin(x[j]); }
        void acos() { for (int j = 0; j < x.size(); ++j) x[j] = std::acos(x[j]); }
        void atan() { for (int j = 0; j < x.size(); ++j) x[j] = std::atan(x[j]); }
        void ceil() { for (int j = 0; j < x.size(); ++j) x[j] = std::ceil(x[j]); }
        void floor() { for (int j = 0; j < x.size(); ++j) x[j] = std::floor(x[j]); }
        void trunc() { for (int j = 0; j < x.size(); ++j) x[j] = std::trunc(x[j]); }
        void round() { for (int j = 0; j < x.size(); ++j) x[j] = std::round(x[j]); }

        double sum() { double s = 0; for (auto& i : x) s += i; return s; };
        double mean() { return sum() / x.size(); };

        dgCMatrix copy() {
            ivec i_copied(i.size());
            ivec p_copied(p.size());
            vec x_copied(x.size());
            std::copy(i.begin(), i.end(), i_copied.begin());
            std::copy(p.begin(), p.end(), p_copied.begin());
            std::copy(x.begin(), x.end(), x_copied.begin());
            return dgCMatrix(i_copied, p_copied, x_copied, Dim[0], Dim[1]);
        }

        // copy values from spmat into a dense double, vec, or mat
        double dense(int row, int col) { for (int j = p[col]; j < p[col + 1] && i[j] <= row; ++j) if (i[j] == row) return x[j]; return 0.0; }
        vec dense(int row, ivec& col) { vec res(col.size()); for (int j = 0; j < col.size(); ++j) res[j] = dense(row, col[j]); return res; };
        vec dense(ivec& row, int col) { vec res(row.size()); for (int j = 0; j < row.size(); ++j) res[j] = dense(row[j], col); return res; };
        mat dense(ivec& row, ivec& col) {
            mat res(row.size(), col.size());
            for (int j = 0; j < row.size(); ++j) for (int k = 0; k < col.size(); ++k) res(j, k) = dense(row[j], col[k]);
            return res;
        };
        vec dense_col(int col) { vec c(Dim[0], 0.0); for (int j = p[col]; j < p[col + 1]; ++j) c[i[j]] = x[j]; return c; }
        mat dense_cols(ivec& c) { mat res(Dim[0], c.size()); for (int j = 0; j < c.size(); ++j) res.column(j) = dense_col(c[j]); return res; }
        vec dense_row(int row) {
            vec r(Dim[1], 0.0);
            for (int col = 0; col < Dim[1]; ++col) { for (int j = p[col]; j < p[col + 1] && i[j] <= row; ++j) if (i[j] == row) r[col] = x[j]; } return r;
        }
        mat dense_rows(ivec& r) { mat res(r.size(), Dim[1]); for (int j = 0; j < r.size(); ++j) res.row(j) = dense_row(r[j]); return res; }

        vec colSums() { vec sums(Dim[1]); for (int col = 0; col < Dim[1]; ++col) { for (int j = p[col]; j < p[col + 1]; ++j) sums(col) += x[j]; } return sums; }
        vec rowSums() { vec sums(Dim[0]); for (int col = 0; col < Dim[1]; ++col) { for (int j = p[col]; j < p[col + 1]; ++j) sums(i[j]) += x[j]; } return sums; }
        vec colMeans() { return colSums() / Dim[0]; }
        vec rowMeans() { return rowSums() / Dim[1]; }

        tmat crossprod();
    };

    // iterator operators
    template <typename T> void operator/=(spmat::itr<T> lhs, vec& rhs) {
        if (lhs.row_ == -1) for (auto it = lhs; it && it.row() < lhs.size(); ++it) it.value() /= rhs[it.row()];
        else for (auto it = lhs; it && it.col() < lhs.size(); ++it) it.value() /= rhs[it.col()];
    }
    template <typename T> void operator/=(vec& lhs, spmat::itr<T> rhs) {
        if (rhs.row_ == -1) for (auto& it = rhs; it && it.row() < lhs.size(); ++it) lhs[it.row()] /= it.value();
        else for (auto& it = rhs; it && it.col() < lhs.size(); ++it) lhs[it.col()] /= it.value();
    }
    template <typename T> void operator*=(spmat::itr<T> lhs, vec& rhs) {
        if (lhs.row_ == -1) for (auto it = lhs; it && it.row() < lhs.size(); ++it) it.value() *= rhs[it.row()];
        else for (auto it = lhs; it && it.col() < lhs.size(); ++it) it.value() *= rhs[it.col()];
    }
    template <typename T> void operator*=(vec& lhs, spmat::itr<T> rhs) {
        if (rhs.row_ == -1) for (auto& it = rhs; it && it.row() < lhs.size(); ++it) lhs[it.row()] *= it.value();
        else for (auto& it = rhs; it && it.col() < lhs.size(); ++it) lhs[it.col()] *= it.value();
    }
    template <typename T> void operator+=(vec& lhs, spmat::itr<T> rhs) {
        if (rhs.row_ == -1) for (auto& it = rhs; it && it.row() < lhs.size(); ++it) lhs[it.row()] += it.value();
        else for (auto& it = rhs; it && it.col() < lhs.size(); ++it) lhs[it.col()] += it.value();
    }
    template <typename T> void operator-=(vec& lhs, spmat::itr<T> rhs) {
        if (rhs.row_ == -1) for (auto& it = rhs; it && it.row() < lhs.size(); ++it) lhs[it.row()] -= it.value();
        else for (auto& it = rhs; it && it.col() < lhs.size(); ++it) lhs[it.col()] -= it.value();
    }
    template <typename T> vec operator+(vec lhs, spmat::itr<T> rhs) { lhs += rhs; return lhs; }
    template <typename T> vec operator-(vec lhs, spmat::itr<T> rhs) { lhs -= rhs; return lhs; }
    template <typename T> vec operator/(vec lhs, spmat::itr<T> rhs) { lhs /= rhs; return lhs; }
    template <typename T> vec operator*(vec lhs, spmat::itr<T> rhs) { lhs *= rhs; return lhs; }

    // Pearson correlation, calculate R-squared value
/*
    double cor(vec& x, vec& y) {
        double sum_x = sum(x), sum_y = sum(y);
        return (x.size() * sum(x * y) - sum_x * sum_y) / std::sqrt((x.size() * sum(x * x) - sum_x * sum_x) * (x.size() * sum(y * y) - sum_y * sum_y));
    }
    template <typename T> double cor(spmat::itr<T>& x, vec& y) {
        double sum_x = x.sum(), sum_y = sum(y);
        return (x.size() * x.crossprod(y) - sum_x * sum_y) / std::sqrt((x.size() * sum(x.crossprod()) - sum_x * sum_x) * (x.size() * sum(y * y) - sum_y * sum_y));
    }
*/
    template <typename T> double cor(spmat::itr<T> x, spmat::itr<T> y) {
        double sum_x = x.sum(), sum_y = y.sum();
        return (x.size() * x.crossprod(y) - sum_x * sum_y) / std::sqrt((x.size() * x.crossprod() - sum_x * sum_x) * (x.size() * y.crossprod() - sum_y * sum_y));
    }
    /*
        template <typename T> double cor(vec& x, spmat::itr<T>& y) { return cor(y, x); }
        vec cor(mat& x, vec& y) { vec m(x.cols(), 0.0); for (int i = 0; i < x.cols(); ++i) m[i] = cor(x.column(i), y); return m; }
        vec cor(vec& x, mat& y) { return cor(y, x); }
        mat cor(mat& x, mat& y) { mat m(x.cols(), y.cols()); for (int i = 0; i < x.cols(); ++i) for (int j = 0; j < y.cols(); ++j) m(i, j) = cor(x.column(i), y.column(j)); return m; }
        mat cor(mat& x, spmat& y) { mat m(x.cols(), y.cols()); for (int i = 0; i < x.cols(); ++i) for (int j = 0; j < y.cols(); ++j) m(i, j) = cor(x.column(i), y.col(j)); return m; }
        mat cor(spmat& x, mat& y) { mat m(x.cols(), y.cols()); for (int i = 0; i < x.cols(); ++i) for (int j = 0; j < y.cols(); ++j) m(i, j) = cor(x.col(i), y.column(j)); return m; }
        */
    mat cor(spmat& x, spmat& y) { mat m(x.cols(), y.cols()); for (int i = 0; i < x.cols(); ++i) for (int j = 0; j < y.cols(); ++j) m(i, j) = cor(x.col(i), y.col(j)); return m; }
    mat cor(spmat& x) {
        mat m(x.cols(), x.cols());
        for (int i = 0; i < x.cols(); ++i) for (int j = 0; j <= i; ++j) {
            if (i == j) m(i, i) = 1;
            else { m(i, j) = cor(x.col(i), x.col(j)); m(j, i) = m(i, j); }
        } return m;
    }
    /*
        mat cor(mat& x) {
            mat m(x.cols(), x.cols());
            for (int i = 0; i < x.cols(); ++i) for (int j = 0; j <= i; ++j) {
                if (i == j) m(i, i) = 1;
                else { m(i, j) = cor(x.column(i), x.column(j)); m(j, i) = m(i, j); }
            } return m;
        }
    */

    // common element-wise mathematical functions
    spmat abs(spmat x) { for (int j = 0; j < x.x.size(); ++j) x.x[j] = std::abs(x.x[j]); return x; }
    spmat exp(spmat x) { for (int j = 0; j < x.x.size(); ++j) x.x[j] = std::exp(x.x[j]); return x; }
    spmat log(spmat x) { for (int j = 0; j < x.x.size(); ++j) x.x[j] = std::log(x.x[j]); return x; }
    spmat log10(spmat x) { for (int j = 0; j < x.x.size(); ++j) x.x[j] = std::log10(x.x[j]); return x; }
    spmat log2(spmat x) { for (int j = 0; j < x.x.size(); ++j) x.x[j] = std::log2(x.x[j]); return x; }
    spmat pow(spmat x, double exp) { for (int j = 0; j < x.x.size(); ++j) x.x[j] = std::pow(x.x[j], exp); return x; }
    spmat sqrt(spmat x) { for (int j = 0; j < x.x.size(); ++j) x.x[j] = std::sqrt(x.x[j]); return x; }
    spmat square(spmat x) { for (int j = 0; j < x.x.size(); ++j) x.x[j] = x.x[j] * x.x[j]; return x; }
    spmat cube(spmat x) { for (int j = 0; j < x.x.size(); ++j) x.x[j] = x.x[j] * x.x[j] * x.x[j]; return x; }
    spmat sin(spmat x) { for (int j = 0; j < x.x.size(); ++j) x.x[j] = std::sin(x.x[j]); return x; }
    spmat cos(spmat x) { for (int j = 0; j < x.x.size(); ++j) x.x[j] = std::cos(x.x[j]); return x; }
    spmat tan(spmat x) { for (int j = 0; j < x.x.size(); ++j) x.x[j] = std::tan(x.x[j]); return x; }
    spmat asin(spmat x) { for (int j = 0; j < x.x.size(); ++j) x.x[j] = std::asin(x.x[j]); return x; }
    spmat acos(spmat x) { for (int j = 0; j < x.x.size(); ++j) x.x[j] = std::acos(x.x[j]); return x; }
    spmat atan(spmat x) { for (int j = 0; j < x.x.size(); ++j) x.x[j] = std::atan(x.x[j]); return x; }
    spmat ceil(spmat x) { for (int j = 0; j < x.x.size(); ++j) x.x[j] = std::ceil(x.x[j]); return x; }
    spmat floor(spmat x) { for (int j = 0; j < x.x.size(); ++j) x.x[j] = std::floor(x.x[j]); return x; }
    spmat trunc(spmat x) { for (int j = 0; j < x.x.size(); ++j) x.x[j] = std::trunc(x.x[j]); return x; }
    spmat round(spmat x) { for (int j = 0; j < x.x.size(); ++j) x.x[j] = std::round(x.x[j]); return x; }

    template <> dgCMatrix as(SEXP m) { return dgCMatrix(m); }
    template <> SEXP wrap(const dgCMatrix& m) { S4 s(std::string("dgCMatrix")); s.slot("i") = m.i; s.slot("p") = m.p; s.slot("x") = m.x; s.slot("Dim") = m.Dim; return s; }

    // upper-triangular in column-major ordering
    class dtpMatrix {
    public:
        ivec Dim;
        vec x;

        dtpMatrix(int c) : Dim(ivec::create(c, c)), x(c* (c + 1) / 2) {}
        dtpMatrix(int c, double v) : Dim(ivec::create(c, c)), x(c* (c + 1) / 2, v) {}
        dtpMatrix(int c, const std::initializer_list<double>& v) : Dim(ivec::create(c, c)), x(v) {}
        dtpMatrix(mat& m) : Dim(ivec::create(m.cols(), m.cols())), x(m.cols()* (m.cols() + 1) / 2) {
            for (int i = 0, k = 0; i < m.cols(); ++i) for (int j = 0; j <= i; ++j, ++k) x[k] = m(i, j);
        }
        dtpMatrix(S4 m) : Dim(m.slot("Dim")), x(m.slot("x")) {}
        dtpMatrix(dgCMatrix& m) {
            Dim = ivec::create(m.cols(), m.cols());
            x = vec(m.cols() * (m.cols() + 1) / 2);
            for (int i = 0; i < m.cols(); ++i) {
                for (auto it = m.begin_col(i); it; ++it) {
                    if (it.row() > i) break;
                    x[i * (i + 1) + it.row()] = it.value();
                }
            }
        }
        dtpMatrix(vec& v, ivec& dim) : x(v) { Dim = dim; }

        class iterator {
        public:
            iterator(dtpMatrix& ptr, int ind) : ptr(ptr) { idx = ind; }
            bool operator!=(iterator& x) { return idx != x.idx; }
            iterator& operator++() { ++idx; return *this; }
            double& operator*() { return ptr.x[idx]; }
        private:
            dtpMatrix& ptr;
            int idx;
        };
        iterator begin() { return iterator(*this, 0); };
        iterator end() { return iterator(*this, Dim[1] * (Dim[1] + 1) / 2); };

        dtpMatrix clone() {
            ivec Dim_copied(2);
            vec x_copied(x.size());
            std::copy(Dim.begin(), Dim.end(), Dim_copied.begin());
            std::copy(x.begin(), x.end(), x_copied.begin());
            return dtpMatrix(x_copied, Dim_copied);
        }

        int nrow() { return Dim[0]; };
        int ncol() { return Dim[1]; };
        int rows() { return Dim[0]; };
        int cols() { return Dim[1]; };
        int size() { return Dim[0] * Dim[1]; };
        ivec i() {
            ivec r(Dim[1] * (Dim[1] + 1) / 2);
            for (int i = 0, k = 0; i < Dim[1]; ++i) { for (int j = 0; j <= i; ++j, ++k) r[k] = i; } return r;
        }
        ivec j() {
            ivec c(Dim[1] * (Dim[1] + 1) / 2);
            for (int i = 0, k = 0; i < Dim[1]; ++i) { for (int j = 0; j <= i; ++j, ++k) c[k] = j; } return c;
        }

        const double& operator[](int i) const { return x[i]; } // simple value in column-major storage
        double& operator[](int i) { return x[i]; }
        const double& operator()(int r, int c) const { return (c >= r) ? x[c * (c + 1) / 2 + r] : x[r * (r + 1) / 2 + c]; } // lower tri check
        double& operator()(int r, int c) { return (c >= r) ? x[c * (c + 1) / 2 + r] : x[r * (r + 1) / 2 + c]; }
        const double& at(int r, int c) const { return x[c * (c + 1) / 2 + r]; } // no lower tri check, must be index in upper tri
        double& at(int r, int c) { return x[c * (c + 1) / 2 + r]; }
        const double& diag(int i) const { return x[i * (i + 1) / 2]; } // diagonal value accessor
        double& diag(int i) { return x[i * (i + 3) / 2]; }
        vec operator*(vec& v) {
            vec res(Dim[0]);
            for (int i = 0; i < Dim[0]; ++i) for (int j = i; j < Dim[0]; ++j) res[i] += x[j * (j + 1) / 2 + i] * v[j];
            for (int i = 0; i < Dim[0]; ++i) for (int j = 0; j < i; ++j) res[i] += x[i * (i + 1) / 2 + j] * v[j];
            return res;
        }

        // scalar operators
        dtpMatrix operator+(double& val) { dtpMatrix n = clone(); for (double& i : n.x) i += val; return n; }
        dtpMatrix& operator+=(double& val) { for (double& i : x) i += val; return *this; }
        dtpMatrix operator-(double& val) { dtpMatrix n = clone(); for (double& i : n.x) i -= val; return n; }
        dtpMatrix& operator-=(double& val) { for (double& i : x) i -= val; return *this; }
        dtpMatrix operator*(double& val) { dtpMatrix n = clone(); for (double& i : n.x) i *= val; return n; }
        dtpMatrix& operator*=(double& val) { for (double& i : x) i *= val; return *this; }
        dtpMatrix operator/(double& val) { dtpMatrix n = clone(); for (double& i : n.x) i /= val; return n; }

        // increments
        dtpMatrix& operator++() { for (double& i : x) ++i; return *this; }
        dtpMatrix& operator--() { for (double& i : x) --i; return *this; }

        // diagonal
        vec diag() { vec d(Dim[1]); for (int i = 0; i < Dim[1]; ++i) d[i] = x[i * (i + 3) / 2]; return d; }

        // algebra methods
        dtpMatrix llt(); // Cholesky decomposition
        dtpMatrix llt(ivec& cols); // Cholesky decomposition on a subset of columns
        vec solve(vec& b); // solve a Cholesky decomposition by forward/backward substitution
        vec solve(vec& b, ivec& cols); // solve a Cholesky decomposition that is a subset of the original system
        vec nnsolve(dtpMatrix& a, vec& b, int maxit, double tol); // solve a Cholesky decomposition for a non-negative solution
    };

    template <> dtpMatrix as(SEXP m) { return dtpMatrix(m); }
    template <> SEXP wrap(const dtpMatrix& m) {
        S4 s(std::string("dtpMatrix"));
        s.slot("x") = m.x;
        s.slot("Dim") = m.Dim;
        s.slot("uplo") = "U";
        s.slot("diag") = "N";
        return s;
    }

    tmat crossprod(mat& x) {
        int n_cols = x.cols(), n_rows = x.rows();
        tmat res(n_cols);
        int ind = 0;
        for (int i = 0; i < n_cols; ++i) {
            for (int j = 0; j <= i; ++j, ++ind) {
                double sum = 0;
                int i_stop = (i + 1) * n_rows, j_ind = j * n_rows;
                for (int i_ind = i * n_rows; i_ind < i_stop; ++i_ind, ++j_ind) sum += x[i_ind] * x[j_ind];
                res[ind] = sum;
            }
        }
        return res;
    }

    // crossprod
    inline tmat spmat::crossprod() {
        tmat res(Dim[1], Dim[1]);
        #if defined(_OPENMP)
        #pragma omp parallel for
        #endif
        for (int c = 0; c < Dim[1]; ++c) {
            for (int c2 = 0; c2 < c; ++c2) res(c2, c) = col(c).crossprod(col(c2));
            res(c, c) = col(c).crossprod();
        }
        return res;
    }

    inline tmat tmat::llt() {
        tmat result(Dim[0]);
        for (int i = 0; i < Dim[0]; ++i) {
            for (int k = 0; k < i; ++k) {
                double val = x[i * (i + 1) / 2 + k];
                for (int j = 0; j < k; ++j) val -= result.x[i * (i + 1) / 2 + j] * result.x[k * (k + 1) / 2 + j];
                result.x[i * (i + 1) / 2 + k] = val / result.x[k * (k + 3) / 2];
            }
            double val = x[i * (i + 3) / 2];
            for (int j = 0; j < i; ++j) val -= result.x[i * (i + 1) / 2 + j] * result.x[i * (i + 1) / 2 + j];
            result.x[i * (i + 1) / 2 + i] = std::sqrt(val);
        }
        return result;
    }

    inline tmat tmat::llt(ivec& cols) {
        tmat result(cols.size());
        for (int i = 0; i < cols.size(); ++i) {
            int col_i = cols[i];
            for (int k = 0; k < i; ++k) {
                double val = x[col_i * (col_i + 1) / 2 + cols[k]];
                for (int j = 0; j < k; ++j) val -= result.x[i * (i + 1) / 2 + j] * result.x[k * (k + 1) / 2 + j];
                result.x[i * (i + 1) / 2 + k] = val / result.x[k * (k + 3) / 2];
            }
            double val = x[col_i * (col_i + 3) / 2];
            for (int j = 0; j < i; ++j) val -= result.x[i * (i + 1) / 2 + j] * result.x[i * (i + 1) / 2 + j];
            result.x[i * (i + 1) / 2 + i] = std::sqrt(val);
        }
        return result;
    }

    inline vec tmat::solve(vec& b) {
        int n = Dim[0];
        vec soln(n);
        for (int i = 0, pos = 0; i < n; i++, pos++) {
            double val = b[i];
            for (int j = 0; j < i; j++, pos++) val -= (x[pos] * soln[j]);
            soln[i] = val / x[pos];
        }
        for (int i = n - 1; i >= 0; i--) {
            double val = soln[i];
            for (int j = i + 1; j < n; j++) val -= (x[j * (j + 1) / 2 + i] * soln[j]);
            soln[i] = val / x[i * (i + 3) / 2];
        }
        return soln;
    }

    inline vec tmat::solve(vec& b, ivec& cols) {
        int n = cols.size();
        vec soln(b.size(), 0.0);
        for (int i = 0, pos = 0; i < n; i++, pos++) {
            double val = b[cols[i]];
            for (int j = 0; j < i; j++, pos++) val -= (x[pos] * soln[cols[j]]);
            soln[cols[i]] = val / x[pos];
        }
        for (int i = n - 1; i >= 0; i--) {
            double val = soln[cols[i]];
            for (int j = i + 1; j < n; j++) val -= (x[j * (j + 1) / 2 + i] * soln[cols[j]]);
            soln[cols[i]] = val / x[i * (i + 3) / 2];
        }
        return soln;
    }

    inline vec tmat::nnsolve(tmat& a, vec& b, int maxit = 0, double tol = 1e-8) {
        vec x = solve(b);
        while (min(x) < 0) {
            ivec gtz_ind;
            for (int i = 0; i < x.size(); ++i) if (x[i] > 0) gtz_ind.push_back(i);
            x = a.llt(gtz_ind).solve(b, gtz_ind);
        }
        if (maxit == 0) return x;
        vec b0 = a * x - b;
        double iter_tol = tol + 1;
        while (maxit-- > 0 && iter_tol > tol) {
            iter_tol = 0;
            for (int i = 0; i < Dim[0]; ++i) {
                double tmp = x[i] - b0[i] / a[i * (i + 3) / 2];
                if (tmp < 0) tmp = 0;
                double diff = tmp - x[i];
                if (diff != 0) {
                    for (int j = 0; j < Dim[0]; ++j)
                        b0[j] += (j > 1) ? a[j * (j + 1) / 2 + i] * diff : a[i * (i + 1) / 2 + j] * diff;
                    double tmp_tol = 2 * std::abs(diff) / (tmp + x[i] + 1e-16);
                    if (tmp_tol > iter_tol) iter_tol = tmp_tol;
                    x[i] = tmp;
                }
            }
        }
        return x;
    }
}

//[[Rcpp::export]]
Rcpp::vec solve(Rcpp::tmat& a, Rcpp::vec& b) {
    return a.llt().solve(b);
}

//[[Rcpp::export]]
Rcpp::vec nnls(Rcpp::tmat& a, Rcpp::vec& b, int maxit, double tol) {
    return a.llt().nnsolve(a, b, maxit, tol);
}

//[[Rcpp::export]]
void row_iterator(Rcpp::spmat& a, int row) {
    Rprintf("Column iterator: \n%3s %3s  %3s %3s %10s \n", "row", "col", "min", "max", "value");
    for (auto it = a.begin_row(row); it; ++it)
        Rprintf("%3d %3d %3d %3d %10.2e \n", it.row(), it.col(), it.min_index_in_range(), it.max_index_in_range(), it.value());
}

//[[Rcpp::export]]
Rcpp::spmat sp_sqrt(Rcpp::spmat& x) {
    x.sqrt();
    return x;
}

//[[Rcpp::export]]
Rcpp::vec sp_plus2(Rcpp::spmat& x, Rcpp::vec& v, int i) {
    return v + x.col(i);
}

//[[Rcpp::export]]
Rcpp::mat mat_test(Rcpp::mat& x) {
    return x;
}