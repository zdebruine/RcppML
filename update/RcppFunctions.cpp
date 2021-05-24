// This file is part of RcppML, an RcppEigen template library
//  for machine learning.
//
// Copyright (C) 2021 Zach DeBruine <zach.debruine@gmail.com>
// github.com/zdebruine/RcppML

#ifndef RCPPML_MATRIXFACTORIZATION_H
#include "RcppML_MatrixFactorization.h"
#endif
#ifndef RCPPML_MATRIXFACTORIZATION2_H
#include "RcppML_MatrixFactorization2.h"
#endif

//[[Rcpp::export]]
Eigen::VectorXd solve_Rcpp_double(Eigen::MatrixXd& a, Eigen::VectorXd& b, bool nonneg = false, int L0 = -1, double L1 = 0,
                      double L2 = 0, double RL2 = 0, int maxit = 100, double tol = 1e-9, std::string L0_method = "ggperm") {
    RcppML::CoeffMatrix<double> model(a, nonneg, L0, L1, L2, RL2, maxit, tol, L0_method);
    return model.solve(b);
}

//[[Rcpp::export]]
Eigen::VectorXf solve_Rcpp_float(Eigen::MatrixXf& a, Eigen::VectorXf& b, bool nonneg = false, int L0 = -1, float L1 = 0,
                      float L2 = 0, float RL2 = 0, int maxit = 100, float tol = 1e-9, std::string L0_method = "ggperm") {
    RcppML::CoeffMatrix<float> model(a, nonneg, L0, L1, L2, RL2, maxit, tol, L0_method);
    return model.solve(b);
}

//[[Rcpp::export]]
Eigen::VectorXd solve_cd_Rcpp_double(Eigen::MatrixXd& a, Eigen::VectorXd& b, Eigen::VectorXd x, bool nonneg = false, int L0 = -1, double L1 = 0,
                      double L2 = 0, double RL2 = 0, int maxit = 100, double tol = 1e-9, std::string L0_method = "ggperm") {
    RcppML::CoeffMatrix<double> model(a, nonneg, L0, L1, L2, RL2, maxit, tol, L0_method);
    model.solve(b, x);
    return x;
}

//[[Rcpp::export]]
Eigen::VectorXf solve_cd_Rcpp_float(Eigen::MatrixXf& a, Eigen::VectorXf& b, Eigen::VectorXf x, bool nonneg = false, int L0 = -1, float L1 = 0,
                      float L2 = 0, float RL2 = 0, int maxit = 100, float tol = 1e-9, std::string L0_method = "ggperm") {
    RcppML::CoeffMatrix<float> model(a, nonneg, L0, L1, L2, RL2, maxit, tol, L0_method);
    model.solve(b, x);
    return x;
}

//[[Rcpp::export]]
Eigen::MatrixXd projectW(RcppML::SparseMatrix& A, Eigen::MatrixXd& W, bool nonneg = false, int L0 = -1, double L1 = 0,
                         double L2 = 0, double RL2 = 0, int maxit = 100, double tol = 1e-9, std::string L0_method = "ggperm") {
    RcppML::MatrixFactorization<double> model(A, W.cols());
    model.matrixW(W.transpose());
    L0H(L0); L1H(L1); L2H(L2); solveMaxit(maxit); solveTol(tol); L0method(L0_method); nonnegH(nonneg);
    model.projectW();
    return model.matrixH();
}

//[[Rcpp::export]]
Eigen::MatrixXd projectH(RcppML::SparseMatrix& A, Eigen::MatrixXd& H, bool nonneg = false, int L0 = -1, double L1 = 0,
                         double L2 = 0, double RL2 = 0, int maxit = 100, double tol = 1e-9, std::string L0_method = "ggperm") {
    RcppML::MatrixFactorization<double> model(A, H.rows());
    model.matrixH(H);
    L0W(L0); L1W(L1); L2W(L2); solveMaxit(maxit); solveTol(tol); L0method(L0_method); nonnegW(nonneg);
    model.projectH();
    return model.matrixW().transpose();
}

//[[Rcpp::export]]
Rcpp::List Rcpp_amf_double(const Eigen::SparseMatrix<double>& A, const unsigned int k, const unsigned int seed = 0,
                           const double tol = 1e-3, const bool nonneg_w = true, const bool nonneg_h = true, const double L1_w = 0,
                           const double L1_h = 0, const unsigned int maxit = 25, const unsigned int threads = 0,
                           const bool verbose = true, const bool calc_mse = false, const bool lowmem = false, const bool diag = true,
                           const unsigned int cd_maxit = 0, const double cd_tol = 1e-8) {

  if (seed == 0) srand((unsigned int)time(0));
  else srand(seed);

  Eigen::setNbThreads(threads);

  WDH<double> model;
  if (k == 1) model = amf1(A, tol, maxit, diag, verbose);
  else if (k == 2) model = amf2(A, tol, nonneg_w, nonneg_h, maxit, calc_mse, diag, verbose);
  else model = amf(A, k, tol, nonneg_w, nonneg_h, L1_w, L1_h, maxit, calc_mse, lowmem, diag, cd_maxit, cd_tol, verbose);
  return(Rcpp::List::create(
    Rcpp::Named("w") = model.W,
    Rcpp::Named("d") = model.D,
    Rcpp::Named("h") = model.H,
    Rcpp::Named("tol") = model.tol,
    Rcpp::Named("mse") = model.mse));
}

//[[Rcpp::export]]
Rcpp::List Rcpp_amf_float(const Eigen::SparseMatrix<float>& A, const unsigned int k, const unsigned int seed = 0,
                          const float tol = 1e-3, const bool nonneg_w = true, const bool nonneg_h = true,
                          const float L1_w = 0, const float L1_h = 0, const unsigned int maxit = 100, const unsigned int threads = 0,
                          const bool verbose = false, const bool calc_mse = false, const bool lowmem = false, const bool diag = true,
                          const unsigned int cd_maxit = 0, const float cd_tol = 1e-8) {

  if (seed == 0) srand((unsigned int)time(0));
  else srand(seed);

  Eigen::setNbThreads(threads);

  WDH<float> model;
  if (k == 1) model = amf1(A, tol, maxit, diag, verbose);
  else if (k == 2) model = amf2(A, tol, nonneg_w, nonneg_h, maxit, calc_mse, diag, verbose);
  else model = amf(A, k, tol, nonneg_w, nonneg_h, L1_w, L1_h, maxit, calc_mse, lowmem, diag, cd_maxit, cd_tol, verbose);

  return(Rcpp::List::create(
    Rcpp::Named("w") = model.W,
    Rcpp::Named("d") = model.D,
    Rcpp::Named("h") = model.H,
    Rcpp::Named("tol") = model.tol,
    Rcpp::Named("mse") = model.mse));
}

//[[Rcpp::export]]
Eigen::VectorXd Rcpp_svd1(const Eigen::SparseMatrix<double>& A, const double tol = 1e-3, const unsigned int maxit = 100,
                          const bool return_v = true, const unsigned int threads = 0) {

  Eigen::setNbThreads(threads);
  return svd1(A, tol, maxit, return_v);
}

//[[Rcpp::export]]
Eigen::VectorXd Rcpp_nnls_double(const Eigen::MatrixXd& a, Eigen::MatrixXd b, const unsigned int cd_maxit = 0, const double cd_tol = 1e-8) {

  Eigen::LLT<Eigen::MatrixXd> a_llt = a.llt();
  for (unsigned int i = 0; i < b.cols(); ++i)
    b.col(i) = nnls(a, b.col(i), a.llt(), cd_maxit, cd_tol);
  return b;
}

//[[Rcpp::export]]
Eigen::VectorXf Rcpp_nnls_float(const Eigen::MatrixXf& a, Eigen::MatrixXf b, const unsigned int cd_maxit = 0, const float cd_tol = 1e-8) {
  Eigen::LLT<Eigen::MatrixXf> a_llt = a.llt();
  for (unsigned int i = 0; i < b.cols(); ++i)
    b.col(i) = nnls(a, b.col(i), a.llt(), cd_maxit, cd_tol);
  return b;
}

//[[Rcpp::export]]
double Rcpp_mse_double(const Eigen::SparseMatrix<double>& A, const Eigen::MatrixXd& w, const Eigen::VectorXd& d, const Eigen::MatrixXd& h) {
  return mse(A, w, d, h);
}

//[[Rcpp::export]]
float Rcpp_mse_float(const Eigen::SparseMatrix<float>& A, const Eigen::MatrixXf& w, const Eigen::VectorXf& d, const Eigen::MatrixXf& h) {
  return mse(A, w, d, h);
}

//[[Rcpp::export]]
Eigen::VectorXd Rcpp_svd1(const Eigen::SparseMatrix<double>& A, const double tol = 1e-3, const unsigned int maxit = 100,
                          const bool return_v = true, const unsigned int threads = 0) {

  Eigen::setNbThreads(threads);
  return svd1(A, tol, maxit, return_v);
}
