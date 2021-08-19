// This file is part of RcppML, a Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU
// Public License v. 2.0.

#include <RcppML.hpp>

//[[Rcpp::export]]
Eigen::MatrixXd Rcpp_nnls(
  const Eigen::MatrixXd& a,
  Eigen::MatrixXd b,
  const unsigned int fast_maxit = 10,
  const unsigned int cd_maxit = 100,
  const double cd_tol = 1e-8,
  const bool nonneg = true) {

  Eigen::LLT<Eigen::MatrixXd> a_llt = a.llt();
  for (unsigned int i = 0; i < b.cols(); ++i){
    Eigen::VectorXd b_i = b.col(i);
    b.col(i) = c_nnls(a, b_i, a.llt(), fast_maxit, cd_maxit, cd_tol, nonneg);
  }
  return b;
}

//[[Rcpp::export]]
Eigen::MatrixXd Rcpp_cdnnls(
  const Eigen::MatrixXd& a,
  Eigen::MatrixXd& b,
  Eigen::MatrixXd x,
  const unsigned int cd_maxit = 100,
  const double cd_tol = 1e-8,
  const bool nonneg = true) {

  for (unsigned int i = 0; i < b.cols(); ++i) {
    Eigen::VectorXd x_i = x.col(i);
    Eigen::VectorXd b_i = b.col(i);
    x.col(i) = c_cdnnls(a, b_i, x_i, cd_maxit, cd_tol, nonneg);
  }
  return x;
}

//[[Rcpp::export]]
double Rcpp_mse_sparse(
  const Rcpp::S4& A_S4,
  Eigen::MatrixXd w,
  const Eigen::VectorXd& d,
  const Eigen::MatrixXd& h,
  const unsigned int threads) {

  Rcpp::dgCMatrix A(A_S4);
  double err = mse(A, w, d, h, threads);
  
  return err;
}

//[[Rcpp::export]]
double Rcpp_mse_dense(
  const Rcpp::NumericMatrix& A,
  Eigen::MatrixXd w,
  const Eigen::VectorXd& d,
  const Eigen::MatrixXd& h,
  const unsigned int threads) {

  double err = mse(A, w, d, h, threads);
  
  return err;
}

//[[Rcpp::export]]
Eigen::MatrixXd Rcpp_project_sparse(
  const Rcpp::S4& A_S4,
  Eigen::MatrixXd& w,
  const bool nonneg,
  const unsigned int fast_maxit,
  const unsigned int cd_maxit,
  const double cd_tol,
  const double L1,
  const unsigned int threads) {

  Rcpp::dgCMatrix A(A_S4);
  return project(A, w, nonneg, fast_maxit, cd_maxit, cd_tol, L1, threads);
}

//[[Rcpp::export]]
Eigen::MatrixXd Rcpp_project_dense(
    const Rcpp::NumericMatrix& A,
    Eigen::MatrixXd& w,
    const bool nonneg,
    const unsigned int fast_maxit,
    const unsigned int cd_maxit,
    const double cd_tol,
    const double L1,
    const unsigned int threads) {
  
    return project(A, w, nonneg, fast_maxit, cd_maxit, cd_tol, L1, threads);
}

//[[Rcpp::export]]
Rcpp::List Rcpp_nmf_sparse(
  const Rcpp::S4& A_S4,
  const Rcpp::S4& At_S4,
  const bool symmetric,
  Eigen::MatrixXd& w_init,
  const double tol = 1e-3,
  const bool nonneg = true,
  const double L1_w = 0,
  const double L1_h = 0,
  const unsigned int maxit = 100,
  const bool diag = true,
  const unsigned int fast_maxit = 10,
  const unsigned int cd_maxit = 100,
  const double cd_tol = 1e-8,
  const bool verbose = false,
  const unsigned int threads = 0) {

  Rcpp::dgCMatrix A(A_S4), At(At_S4);

  wdhmodel m = c_nmf_sparse(A, At, symmetric, w_init, tol, nonneg, L1_w, L1_h, maxit, diag, fast_maxit, cd_maxit, cd_tol, verbose, threads);
  unsigned int it = m.it + 1;
  return Rcpp::List::create(Rcpp::Named("w") = m.w, Rcpp::Named("d") = m.d, Rcpp::Named("h") = m.h, Rcpp::Named("tol") = m.tol, Rcpp::Named("iter") = it);
}

//[[Rcpp::export]]
Rcpp::List Rcpp_nmf_dense(
  const Rcpp::NumericMatrix& A,
  const bool symmetric,
  Eigen::MatrixXd& w_init,
  const double tol = 1e-3,
  const bool nonneg = true,
  const double L1_w = 0,
  const double L1_h = 0,
  const unsigned int maxit = 100,
  const bool diag = true,
  const unsigned int fast_maxit = 10,
  const unsigned int cd_maxit = 100,
  const double cd_tol = 1e-8,
  const bool verbose = false,
  const unsigned int threads = 0) {

  Rcpp::NumericMatrix At;
  if(!symmetric) At = Rcpp::transpose(A);

  wdhmodel m = c_nmf_dense(A, At, symmetric, w_init, tol, nonneg, L1_w, L1_h, maxit, diag, fast_maxit, cd_maxit, cd_tol, verbose, threads);
  unsigned int it = m.it + 1;
  return Rcpp::List::create(Rcpp::Named("w") = m.w, Rcpp::Named("d") = m.d, Rcpp::Named("h") = m.h, Rcpp::Named("tol") = m.tol, Rcpp::Named("iter") = it);
}

//[[Rcpp::export]]
Rcpp::List Rcpp_nmf1_sparse(
  const Rcpp::S4& A_S4,
  Eigen::MatrixXd& w_init,
  const double tol,
  const bool nonneg,
  const unsigned int maxit,
  const bool verbose,
  const bool diag){

  Rcpp::dgCMatrix A(A_S4);
  wdhmodel m = c_nmf1_sparse(A, w_init, tol, nonneg, maxit, verbose, diag);
  unsigned int it = m.it + 1;
  return Rcpp::List::create(Rcpp::Named("w") = m.w, Rcpp::Named("d") = m.d, Rcpp::Named("h") = m.h, Rcpp::Named("tol") = m.tol, Rcpp::Named("iter") = it);
}

//[[Rcpp::export]]
Rcpp::List Rcpp_nmf1_dense(
  const Rcpp::NumericMatrix& A,
  Eigen::MatrixXd& w_init,
  const double tol,
  const bool nonneg,
  const unsigned int maxit,
  const bool verbose,
  const bool diag){

  wdhmodel m = c_nmf1_dense(A, w_init, tol, nonneg, maxit, verbose, diag);
  unsigned int it = m.it + 1;
  return Rcpp::List::create(Rcpp::Named("w") = m.w, Rcpp::Named("d") = m.d, Rcpp::Named("h") = m.h, Rcpp::Named("tol") = m.tol, Rcpp::Named("iter") = it);
}

//[[Rcpp::export]]
Rcpp::List Rcpp_nmf2_sparse(
  const Rcpp::S4& A_S4,
  Eigen::MatrixXd& w_init,
  const double tol,
  const bool nonneg,
  const unsigned int maxit,
  const bool verbose,
  const bool diag,
  const std::vector<unsigned int> samples){

  Rcpp::dgCMatrix A(A_S4);
  wdhmodel m = c_nmf2_sparse(A, w_init, tol, nonneg, maxit, verbose, diag, samples);
  unsigned int it = m.it + 1;
  return Rcpp::List::create(Rcpp::Named("w") = m.w, Rcpp::Named("d") = m.d, Rcpp::Named("h") = m.h, Rcpp::Named("tol") = m.tol, Rcpp::Named("iter") = it);
}

//[[Rcpp::export]]
Rcpp::List Rcpp_nmf2_dense(
  const Rcpp::NumericMatrix& A,
  Eigen::MatrixXd& w_init,
  const double tol,
  const bool nonneg,
  const unsigned int maxit,
  const bool verbose,
  const bool diag,
  const std::vector<unsigned int> samples){

  wdhmodel m = c_nmf2_dense(A, w_init, tol, nonneg, maxit, verbose, diag, samples);
  unsigned int it = m.it + 1;
  return Rcpp::List::create(Rcpp::Named("w") = m.w, Rcpp::Named("d") = m.d, Rcpp::Named("h") = m.h, Rcpp::Named("tol") = m.tol, Rcpp::Named("iter") = it);
}

//[[Rcpp::export]]
Rcpp::List Rcpp_bipartition_sparse(
  const Rcpp::S4& A_S4,
  Eigen::MatrixXd& w,
  const std::vector<unsigned int>& samples,
  const double tol = 1e-4,
  const bool nonneg = true,
  bool calc_centers = true,
  bool calc_dist = true,
  const unsigned int maxit = 100,
  const bool verbose = false,
  const bool diag = true, 
  const unsigned int seed = 0){

  Rcpp::dgCMatrix A(A_S4);
  bipartitionModel m = c_bipartition_sparse(A, w, tol, nonneg, samples, calc_centers, calc_dist, maxit, verbose, diag);

  for(unsigned int i = 0; i < m.samples1.size(); ++i) ++m.samples1[i];
  for(unsigned int i = 0; i < m.samples2.size(); ++i) ++m.samples2[i];

    return Rcpp::List::create(
    Rcpp::Named("v") = m.v, 
    Rcpp::Named("dist") = m.dist,
    Rcpp::Named("size1") = m.size1,
    Rcpp::Named("size2") = m.size2,
    Rcpp::Named("samples1") = m.samples1,
    Rcpp::Named("samples2") = m.samples2,
    Rcpp::Named("center1") = m.center1,
    Rcpp::Named("center2") = m.center2);
}

//[[Rcpp::export]]
Rcpp::List Rcpp_bipartition_dense(
  const Rcpp::NumericMatrix& A,
  Eigen::MatrixXd& w,
  const std::vector<unsigned int>& samples,
  const double tol = 1e-4,
  const bool nonneg = true,
  bool calc_centers = true,
  bool calc_dist = true,
  const unsigned int maxit = 100,
  const bool verbose = false,
  const bool diag = true){
  
  bipartitionModel m = c_bipartition_dense(A, w, tol, nonneg, samples, calc_centers, calc_dist, maxit, verbose, diag);

  for(unsigned int i = 0; i < m.samples1.size(); ++i) ++m.samples1[i];
  for(unsigned int i = 0; i < m.samples2.size(); ++i) ++m.samples2[i];
  
  return Rcpp::List::create(
    Rcpp::Named("v") = m.v, 
    Rcpp::Named("dist") = m.dist,
    Rcpp::Named("size1") = m.size1,
    Rcpp::Named("size2") = m.size2,
    Rcpp::Named("samples1") = m.samples1,
    Rcpp::Named("samples2") = m.samples2,
    Rcpp::Named("center1") = m.center1,
    Rcpp::Named("center2") = m.center2);
}

//[[Rcpp::export]]
Rcpp::List Rcpp_dclust_sparse(
  const Rcpp::S4& A_S4, 
  Eigen::MatrixXd& w,
  const double min_dist, 
  const unsigned int min_samples, 
  const bool verbose,
  const unsigned int threads,
  const double bipartition_tol,
  const bool bipartition_nonneg,
  const unsigned int bipartition_maxit,
  const bool calc_centers,
  const bool diag) {

  Rcpp::dgCMatrix A(A_S4);

  std::vector<clusterModel> clusters = dclust(A, w, min_dist, min_samples, true, threads, bipartition_tol, bipartition_nonneg, bipartition_maxit, calc_centers, diag);

  Rcpp::List result(clusters.size());
  for (unsigned int i = 0; i < clusters.size(); ++i) {
    result[i] = Rcpp::List::create(
      Rcpp::Named("id") = clusters[i].id,
      Rcpp::Named("samples") = clusters[i].samples,
      Rcpp::Named("center") = clusters[i].center,
      Rcpp::Named("dist") = clusters[i].dist,
      Rcpp::Named("leaf") = clusters[i].leaf);
  }
  return result;
}

//[[Rcpp::export]]
Rcpp::List Rcpp_dclust_dense(
  const Rcpp::NumericMatrix& A, 
  Eigen::MatrixXd& w,
  const double min_dist, 
  const unsigned int min_samples, 
  const bool verbose,
  const unsigned int threads,
  const double bipartition_tol,
  const bool bipartition_nonneg,
  const unsigned int bipartition_maxit,
  const bool calc_centers,
  const bool diag,
  const unsigned int seed) {

  std::vector<clusterModel> clusters = dclust(A, w, min_dist, min_samples, true, threads, bipartition_tol, bipartition_nonneg, bipartition_maxit, calc_centers, diag);

  Rcpp::List result(clusters.size());
  for (unsigned int i = 0; i < clusters.size(); ++i) {
    result[i] = Rcpp::List::create(
      Rcpp::Named("id") = clusters[i].id,
      Rcpp::Named("samples") = clusters[i].samples,
      Rcpp::Named("center") = clusters[i].center,
      Rcpp::Named("dist") = clusters[i].dist,
      Rcpp::Named("leaf") = clusters[i].leaf);
  }
  return result;
}