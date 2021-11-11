// This file is a standalone RcppEigen demo for basic minimal NMF

//[[Rcpp::depends(RcppSparse)]]
#include <RcppSparse.h>
//[[Rcpp::depends(RcppEigen)]]
//[[Rcpp::plugins(openmp)]]
#include <RcppEigen.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// the goal is to simplify this code

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

// save sparse matrices in various formats, and benchmark methods to read it from disk and save to dgCMatrix

// solve for 'h' given sparse 'A' in 'A = wh'
Eigen::MatrixXd updateW(RcppSparse::Matrix A, const Eigen::MatrixXd& w, const double L1) {

    // need to specify size of rows and columns
    Eigen::MatrixXd h(w.rows(), A.cols());
    Eigen::MatrixXd a = w * w.transpose();
    a.diagonal().array() += 1e-15;
    for (int i = 0; i < h.cols(); ++i) {
        // need a column delimiter
        // read file.col(i) to two std::vectors (unsigned int) and (float)
        RcppML::SparseVector A_i = SpMatDiskView(filename, i);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(h.rows());
        for (RcppML::SparseVector::InnerIterator it(A_i); j < stream_value.size(); ++j)
            b += stream_value[j] * w.col(stream_row[j]);
        solve_nnls(a, b, h, i);
    }

    // scale h
    Eigen::VectorXd d = h.rowwise().sum();
    d.array() += 1e-15;
    for (int i = 0; i < h.rows(); ++i)
        for (int j = 0; j < h.cols(); ++j)
            h(i, j) /= d(i);

    // now update "w" in-place in "A"
    a = h * h.transpose();
    a.diagonal().array() += 1e-15;
    // calculate "b"
    Eigen::MatrixXd b_w = Eigen::MatrixXd::Zero(w.rows(), A.cols());
    for (int i = 0; i < h.cols(); ++i) {
        for (int j = 0; j < )
            b_w.col(it.row()) += it.value() * h.col(i);
    }
    if(L1 != 0) b_w.array() -= L1;
    Eigen::MatrixXd w_(w.rows(), w.cols());
    for (int i = 0; i < w.cols(); ++i) {
        solve_nnls(a, b.col(i), w_, i);
    }
    return w_;
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
Rcpp::List nmf_gold_standard(RcppSparse::Matrix A, Eigen::MatrixXd w, const double tol = 1e-4, int maxit = 100) {
    RcppSparse::Matrix At = A.transpose();

    if (w.cols() != A.rows()) Rcpp::stop("number of rows in 'A' and columns in 'w' are not equivalent");
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(w.rows(), A.cols());
    Eigen::VectorXd d = Eigen::VectorXd::Zero(w.rows());
    double tol_ = 1;
    for (int iter = 0; iter < maxit && tol_ > tol; ++iter) {
        Eigen::MatrixXd w_it = w;
        update_bench(A, w, h);

        d = h.rowwise().sum();
        d.array() += 1e-15;
        for (int i = 0; i < h.rows(); ++i)
            for (int j = 0; j < h.cols(); ++j)
                h(i, j) /= d(i);

        update_bench(At, h, w);

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