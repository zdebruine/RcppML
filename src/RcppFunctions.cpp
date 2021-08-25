// This file is part of RcppML, a Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU
// Public License v. 2.0.

#include <RcppML.hpp>

//[[Rcpp::export]]
Eigen::MatrixXd Rcpp_nnls(const Eigen::MatrixXd& a, Eigen::MatrixXd b, const unsigned int fast_maxit, const unsigned int cd_maxit,
                          const double cd_tol, const bool nonneg, double L1) {

  if (a.rows() != a.cols()) Rcpp::stop("'a' is not symmetric");
  if (a.rows() != b.rows()) Rcpp::stop("dimensions of 'b' and 'a' are not compatible!");
  if (L1 != 0) {
    L1 *= b.maxCoeff();
    for (unsigned int i = 0; i < b.size(); ++i) b(i) -= L1;
  }
  Eigen::LLT<Eigen::MatrixXd> a_llt = a.llt();
  for (unsigned int i = 0; i < b.cols(); ++i) {
    Eigen::VectorXd b_i = b.col(i);
    b.col(i) = c_nnls(a, b_i, a_llt, nonneg, cd_maxit, fast_maxit, cd_tol);
  }
  return b;
}

//[[Rcpp::export]]
Eigen::MatrixXd Rcpp_project_sparse(const Rcpp::S4& A_S4, const Eigen::MatrixXd& w, const bool nonneg, const double L1,
                                    const unsigned int cd_maxit, const unsigned int fast_maxit, const double cd_tol, const unsigned int threads) {

  RcppML::SparseMatrix A(A_S4);
  Eigen::MatrixXd h(w.rows(), A.cols());
  project(A, w, h, nonneg, L1, cd_maxit, fast_maxit, cd_tol, threads);
  return h;
}

//[[Rcpp::export]]
Eigen::MatrixXd Rcpp_project_dense(const Rcpp::NumericMatrix& A, const Eigen::MatrixXd& w, const bool nonneg, const double L1,
                                   const unsigned int cd_maxit, const unsigned int fast_maxit, const double cd_tol, const unsigned int threads) {

  Eigen::MatrixXd h(w.rows(), A.cols());
  project(A, w, h, nonneg, L1, cd_maxit, fast_maxit, cd_tol, threads);
  return h;
}

//[[Rcpp::export]]
Eigen::MatrixXd Rcpp_projectInPlace_sparse(const Rcpp::S4& A_S4, const Eigen::MatrixXd& h, const bool nonneg, const double L1,
                                           const unsigned int cd_maxit, const unsigned int fast_maxit, const double cd_tol, const unsigned int threads) {

  RcppML::SparseMatrix A(A_S4);
  Eigen::MatrixXd w(h.rows(), A.rows());
  projectInPlace(A, h, w, nonneg, L1, cd_maxit, fast_maxit, cd_tol, threads);
  return w;
}

//[[Rcpp::export]]
Eigen::MatrixXd Rcpp_projectInPlace_dense(const Rcpp::NumericMatrix& A, const Eigen::MatrixXd& h, const bool nonneg, const double L1,
                                          const unsigned int cd_maxit, const unsigned int fast_maxit, const double cd_tol, const unsigned int threads) {

  Eigen::MatrixXd w(h.rows(), A.rows());
  projectInPlace(A, h, w, nonneg, L1, cd_maxit, fast_maxit, cd_tol, threads);
  return w;
}

//[[Rcpp::export]]
Rcpp::List Rcpp_nmf_sparse(const Rcpp::S4& A_S4, const unsigned int k, const unsigned int seed, double tol, const bool nonneg,
                           const Rcpp::NumericVector L1, unsigned int maxit, const bool updateInPlace, const bool diag, const bool verbose,
                           const Rcpp::IntegerVector cd_maxit, const Rcpp::IntegerVector fast_maxit, const double cd_tol, const unsigned int threads) {

  RcppML::SparseMatrix A(A_S4);
  RcppML::MatrixFactorization m(k, A.rows(), A.cols(), seed);

  // set model parameters
  m.tol = tol; m.updateInPlace = updateInPlace; m.nonneg = nonneg; m.L1_w = L1(0); m.L1_h = L1(1);
  m.maxit = maxit; m.diag = diag; m.verbose = verbose; m.cd_maxit_w = cd_maxit(0); m.fast_maxit_w = fast_maxit(0); m.cd_maxit_h = cd_maxit(1); m.fast_maxit_h = fast_maxit(1);
  m.cd_tol = cd_tol; m.threads = threads;

  // fit the model by lternating least squares
  m.fit(A);

  return Rcpp::List::create(Rcpp::Named("w") = m.matrixW().transpose(), Rcpp::Named("d") = m.vectorD(),
                            Rcpp::Named("h") = m.matrixH(), Rcpp::Named("tol") = m.fit_tol(), Rcpp::Named("iter") = m.fit_iter());
}

//[[Rcpp::export]]
Rcpp::List Rcpp_nmf_dense(Rcpp::NumericMatrix& A, const unsigned int k, const unsigned int seed, double tol, const bool nonneg,
                           const Rcpp::NumericVector L1, unsigned int maxit, const bool updateInPlace, const bool diag, const bool verbose,
                           const Rcpp::IntegerVector cd_maxit, const Rcpp::IntegerVector fast_maxit, const double cd_tol, const unsigned int threads) {

  RcppML::MatrixFactorization m(k, A.rows(), A.cols(), seed);
  m.tol = tol; m.updateInPlace = updateInPlace; m.nonneg = nonneg; m.L1_w = L1(0); m.L1_h = L1(1);
  m.maxit = maxit; m.diag = diag; m.verbose = verbose; m.cd_maxit_w = cd_maxit(0); m.fast_maxit_w = fast_maxit(0); m.cd_maxit_h = cd_maxit(1); m.fast_maxit_h = fast_maxit(1);
  m.cd_tol = cd_tol; m.threads = threads;
  m.fit(A);
  return Rcpp::List::create(Rcpp::Named("w") = m.matrixW().transpose(), Rcpp::Named("d") = m.vectorD(),
                            Rcpp::Named("h") = m.matrixH(), Rcpp::Named("tol") = m.fit_tol(), Rcpp::Named("iter") = m.fit_iter());
}

//[[Rcpp::export]]
double Rcpp_mse_sparse(const Rcpp::S4& A_S4, Eigen::MatrixXd& w, Eigen::VectorXd& d, Eigen::MatrixXd& h, const unsigned int threads) {

  RcppML::SparseMatrix A(A_S4);
  if (w.rows() == A.rows()) w.transposeInPlace();
  if (h.rows() == A.cols()) h.transposeInPlace();
  if (w.rows() != h.rows()) Rcpp::stop("'w' and 'h' are not of equal rank");
  if (w.cols() != A.rows()) Rcpp::stop("dimensions of 'w' and 'A' are incompatible");
  if (h.cols() != A.cols()) Rcpp::stop("dimensions of 'h' and 'A' are incompatible");
  if (d.size() != w.rows()) Rcpp::stop("length of 'd' is not equal to the rank of 'w' and 'h'");

  RcppML::MatrixFactorization m(w, d, h);
  m.threads = threads;
  return m.mse(A);
}

//[[Rcpp::export]]
double Rcpp_mse_dense(Rcpp::NumericMatrix& A, Eigen::MatrixXd& w, Eigen::VectorXd& d, Eigen::MatrixXd& h, const unsigned int threads) {

  if (w.rows() == A.rows()) w.transposeInPlace();
  if (h.rows() == A.cols()) h.transposeInPlace();
  if (w.rows() != h.rows()) Rcpp::stop("'w' and 'h' are not of equal rank");
  if (w.cols() != A.rows()) Rcpp::stop("dimensions of 'w' and 'A' are incompatible");
  if (h.cols() != A.cols()) Rcpp::stop("dimensions of 'h' and 'A' are incompatible");
  if (d.size() != w.rows()) Rcpp::stop("length of 'd' is not equal to the rank of 'w' and 'h'");
  
  RcppML::MatrixFactorization m(w, d, h);
  m.threads = threads;
  return m.mse(A);
}

//[[Rcpp::export]]
Rcpp::List Rcpp_bipartition_sparse(const Rcpp::S4& A_S4, const std::vector<unsigned int>& samples, const double tol, const bool nonneg,
                                   bool calc_dist, const unsigned int maxit, const bool verbose, const unsigned int seed) {

  RcppML::SparseMatrix A(A_S4);
  Eigen::MatrixXd w = randomMatrix(2, A.rows(), seed);
  bipartitionModel m = c_bipartition_sparse(A, w, samples, tol, nonneg, calc_dist, maxit, verbose);

  return Rcpp::List::create(Rcpp::Named("v") = m.v, Rcpp::Named("dist") = m.dist, Rcpp::Named("size1") = m.size1,
                            Rcpp::Named("size2") = m.size2, Rcpp::Named("samples1") = m.samples1, Rcpp::Named("samples2") = m.samples2,
                            Rcpp::Named("center1") = m.center1, Rcpp::Named("center2") = m.center2);
}

//[[Rcpp::export]]
Rcpp::List Rcpp_bipartition_dense(const Rcpp::NumericMatrix& A, const std::vector<unsigned int>& samples, const double tol, const bool nonneg,
                                   bool calc_dist, const unsigned int maxit, const bool verbose, const unsigned int seed) {

  Eigen::MatrixXd w = randomMatrix(2, A.rows(), seed);
  bipartitionModel m = c_bipartition_dense(A, w, samples, tol, nonneg, calc_dist, maxit, verbose);

  return Rcpp::List::create(Rcpp::Named("v") = m.v, Rcpp::Named("dist") = m.dist, Rcpp::Named("size1") = m.size1,
                            Rcpp::Named("size2") = m.size2, Rcpp::Named("samples1") = m.samples1, Rcpp::Named("samples2") = m.samples2,
                            Rcpp::Named("center1") = m.center1, Rcpp::Named("center2") = m.center2);
}

//[[Rcpp::export]]
Rcpp::List Rcpp_dclust_sparse(const Rcpp::S4& A_S4, const double min_dist, const unsigned int min_samples, const bool verbose,
                              const unsigned int threads, const double tol, const bool nonneg, const unsigned int maxit, const unsigned int seed) {

  RcppML::SparseMatrix A(A_S4);

  RcppML::clusterModel m = RcppML::clusterModel(A, min_samples, min_dist);
  m.nonneg = nonneg; m.verbose = verbose; m.tol = tol; m.min_dist = min_dist; m.seed = seed; m.maxit = maxit; m.threads = threads;
  m.min_samples = min_samples;

  m.dclust();

  std::vector<cluster> clusters = m.getClusters();

  Rcpp::List result(clusters.size());
  for (unsigned int i = 0; i < clusters.size(); ++i) {
    result[i] = Rcpp::List::create(Rcpp::Named("id") = clusters[i].id, Rcpp::Named("samples") = clusters[i].samples,
                                   Rcpp::Named("center") = clusters[i].center, Rcpp::Named("dist") = clusters[i].dist,
                                   Rcpp::Named("leaf") = clusters[i].leaf);
  }
  return result;
}