// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// Rcpp_predict_sparse
Eigen::MatrixXd Rcpp_predict_sparse(const Rcpp::S4& A, const Rcpp::S4& mask, Eigen::MatrixXd w, const bool nonneg, const double L1, const double L2, const unsigned int threads, const bool mask_zeros);
RcppExport SEXP _RcppML_Rcpp_predict_sparse(SEXP ASEXP, SEXP maskSEXP, SEXP wSEXP, SEXP nonnegSEXP, SEXP L1SEXP, SEXP L2SEXP, SEXP threadsSEXP, SEXP mask_zerosSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::S4& >::type A(ASEXP);
    Rcpp::traits::input_parameter< const Rcpp::S4& >::type mask(maskSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type w(wSEXP);
    Rcpp::traits::input_parameter< const bool >::type nonneg(nonnegSEXP);
    Rcpp::traits::input_parameter< const double >::type L1(L1SEXP);
    Rcpp::traits::input_parameter< const double >::type L2(L2SEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type threads(threadsSEXP);
    Rcpp::traits::input_parameter< const bool >::type mask_zeros(mask_zerosSEXP);
    rcpp_result_gen = Rcpp::wrap(Rcpp_predict_sparse(A, mask, w, nonneg, L1, L2, threads, mask_zeros));
    return rcpp_result_gen;
END_RCPP
}
// Rcpp_predict_dense
Eigen::MatrixXd Rcpp_predict_dense(Eigen::MatrixXd& A_, const Rcpp::S4& mask, Eigen::MatrixXd w, const bool nonneg, const double L1, const double L2, const unsigned int threads, const bool mask_zeros);
RcppExport SEXP _RcppML_Rcpp_predict_dense(SEXP A_SEXP, SEXP maskSEXP, SEXP wSEXP, SEXP nonnegSEXP, SEXP L1SEXP, SEXP L2SEXP, SEXP threadsSEXP, SEXP mask_zerosSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type A_(A_SEXP);
    Rcpp::traits::input_parameter< const Rcpp::S4& >::type mask(maskSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type w(wSEXP);
    Rcpp::traits::input_parameter< const bool >::type nonneg(nonnegSEXP);
    Rcpp::traits::input_parameter< const double >::type L1(L1SEXP);
    Rcpp::traits::input_parameter< const double >::type L2(L2SEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type threads(threadsSEXP);
    Rcpp::traits::input_parameter< const bool >::type mask_zeros(mask_zerosSEXP);
    rcpp_result_gen = Rcpp::wrap(Rcpp_predict_dense(A_, mask, w, nonneg, L1, L2, threads, mask_zeros));
    return rcpp_result_gen;
END_RCPP
}
// Rcpp_mse_sparse
double Rcpp_mse_sparse(const Rcpp::S4& A, const Rcpp::S4& mask, Eigen::MatrixXd w, Eigen::VectorXd d, Eigen::MatrixXd h, const unsigned int threads, const bool mask_zeros);
RcppExport SEXP _RcppML_Rcpp_mse_sparse(SEXP ASEXP, SEXP maskSEXP, SEXP wSEXP, SEXP dSEXP, SEXP hSEXP, SEXP threadsSEXP, SEXP mask_zerosSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::S4& >::type A(ASEXP);
    Rcpp::traits::input_parameter< const Rcpp::S4& >::type mask(maskSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type w(wSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type d(dSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type h(hSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type threads(threadsSEXP);
    Rcpp::traits::input_parameter< const bool >::type mask_zeros(mask_zerosSEXP);
    rcpp_result_gen = Rcpp::wrap(Rcpp_mse_sparse(A, mask, w, d, h, threads, mask_zeros));
    return rcpp_result_gen;
END_RCPP
}
// Rcpp_mse_dense
double Rcpp_mse_dense(Eigen::MatrixXd& A_, const Rcpp::S4& mask, Eigen::MatrixXd w, Eigen::VectorXd d, Eigen::MatrixXd h, const unsigned int threads, const bool mask_zeros);
RcppExport SEXP _RcppML_Rcpp_mse_dense(SEXP A_SEXP, SEXP maskSEXP, SEXP wSEXP, SEXP dSEXP, SEXP hSEXP, SEXP threadsSEXP, SEXP mask_zerosSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type A_(A_SEXP);
    Rcpp::traits::input_parameter< const Rcpp::S4& >::type mask(maskSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type w(wSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type d(dSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type h(hSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type threads(threadsSEXP);
    Rcpp::traits::input_parameter< const bool >::type mask_zeros(mask_zerosSEXP);
    rcpp_result_gen = Rcpp::wrap(Rcpp_mse_dense(A_, mask, w, d, h, threads, mask_zeros));
    return rcpp_result_gen;
END_RCPP
}
// Rcpp_mse_missing_sparse
double Rcpp_mse_missing_sparse(const Rcpp::S4& A, const Rcpp::S4& mask, Eigen::MatrixXd w, Eigen::VectorXd d, Eigen::MatrixXd h, const unsigned int threads);
RcppExport SEXP _RcppML_Rcpp_mse_missing_sparse(SEXP ASEXP, SEXP maskSEXP, SEXP wSEXP, SEXP dSEXP, SEXP hSEXP, SEXP threadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::S4& >::type A(ASEXP);
    Rcpp::traits::input_parameter< const Rcpp::S4& >::type mask(maskSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type w(wSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type d(dSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type h(hSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type threads(threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(Rcpp_mse_missing_sparse(A, mask, w, d, h, threads));
    return rcpp_result_gen;
END_RCPP
}
// Rcpp_mse_missing_dense
double Rcpp_mse_missing_dense(Eigen::MatrixXd& A_, const Rcpp::S4& mask, Eigen::MatrixXd w, Eigen::VectorXd d, Eigen::MatrixXd h, const unsigned int threads);
RcppExport SEXP _RcppML_Rcpp_mse_missing_dense(SEXP A_SEXP, SEXP maskSEXP, SEXP wSEXP, SEXP dSEXP, SEXP hSEXP, SEXP threadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type A_(A_SEXP);
    Rcpp::traits::input_parameter< const Rcpp::S4& >::type mask(maskSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type w(wSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type d(dSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type h(hSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type threads(threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(Rcpp_mse_missing_dense(A_, mask, w, d, h, threads));
    return rcpp_result_gen;
END_RCPP
}
// Rcpp_nmf_sparse
Rcpp::List Rcpp_nmf_sparse(const Rcpp::S4& A, const Rcpp::S4& mask, const double tol, const unsigned int maxit, const bool verbose, const bool nonneg, const std::vector<double> L1, const std::vector<double> L2, const bool diag, const unsigned int threads, Rcpp::List w_init, const Rcpp::S4& link_matrix_h, const bool mask_zeros, const bool link_h, const bool sort_model);
RcppExport SEXP _RcppML_Rcpp_nmf_sparse(SEXP ASEXP, SEXP maskSEXP, SEXP tolSEXP, SEXP maxitSEXP, SEXP verboseSEXP, SEXP nonnegSEXP, SEXP L1SEXP, SEXP L2SEXP, SEXP diagSEXP, SEXP threadsSEXP, SEXP w_initSEXP, SEXP link_matrix_hSEXP, SEXP mask_zerosSEXP, SEXP link_hSEXP, SEXP sort_modelSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::S4& >::type A(ASEXP);
    Rcpp::traits::input_parameter< const Rcpp::S4& >::type mask(maskSEXP);
    Rcpp::traits::input_parameter< const double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type maxit(maxitSEXP);
    Rcpp::traits::input_parameter< const bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< const bool >::type nonneg(nonnegSEXP);
    Rcpp::traits::input_parameter< const std::vector<double> >::type L1(L1SEXP);
    Rcpp::traits::input_parameter< const std::vector<double> >::type L2(L2SEXP);
    Rcpp::traits::input_parameter< const bool >::type diag(diagSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type threads(threadsSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type w_init(w_initSEXP);
    Rcpp::traits::input_parameter< const Rcpp::S4& >::type link_matrix_h(link_matrix_hSEXP);
    Rcpp::traits::input_parameter< const bool >::type mask_zeros(mask_zerosSEXP);
    Rcpp::traits::input_parameter< const bool >::type link_h(link_hSEXP);
    Rcpp::traits::input_parameter< const bool >::type sort_model(sort_modelSEXP);
    rcpp_result_gen = Rcpp::wrap(Rcpp_nmf_sparse(A, mask, tol, maxit, verbose, nonneg, L1, L2, diag, threads, w_init, link_matrix_h, mask_zeros, link_h, sort_model));
    return rcpp_result_gen;
END_RCPP
}
// Rcpp_nmf_dense
Rcpp::List Rcpp_nmf_dense(Eigen::MatrixXd& A_, const Rcpp::S4& mask, const double tol, const unsigned int maxit, const bool verbose, const bool nonneg, const std::vector<double> L1, const std::vector<double> L2, const bool diag, const unsigned int threads, Rcpp::List w_init, const Rcpp::S4& link_matrix_h, const bool mask_zeros, const bool link_h, const bool sort_model);
RcppExport SEXP _RcppML_Rcpp_nmf_dense(SEXP A_SEXP, SEXP maskSEXP, SEXP tolSEXP, SEXP maxitSEXP, SEXP verboseSEXP, SEXP nonnegSEXP, SEXP L1SEXP, SEXP L2SEXP, SEXP diagSEXP, SEXP threadsSEXP, SEXP w_initSEXP, SEXP link_matrix_hSEXP, SEXP mask_zerosSEXP, SEXP link_hSEXP, SEXP sort_modelSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type A_(A_SEXP);
    Rcpp::traits::input_parameter< const Rcpp::S4& >::type mask(maskSEXP);
    Rcpp::traits::input_parameter< const double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type maxit(maxitSEXP);
    Rcpp::traits::input_parameter< const bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< const bool >::type nonneg(nonnegSEXP);
    Rcpp::traits::input_parameter< const std::vector<double> >::type L1(L1SEXP);
    Rcpp::traits::input_parameter< const std::vector<double> >::type L2(L2SEXP);
    Rcpp::traits::input_parameter< const bool >::type diag(diagSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type threads(threadsSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type w_init(w_initSEXP);
    Rcpp::traits::input_parameter< const Rcpp::S4& >::type link_matrix_h(link_matrix_hSEXP);
    Rcpp::traits::input_parameter< const bool >::type mask_zeros(mask_zerosSEXP);
    Rcpp::traits::input_parameter< const bool >::type link_h(link_hSEXP);
    Rcpp::traits::input_parameter< const bool >::type sort_model(sort_modelSEXP);
    rcpp_result_gen = Rcpp::wrap(Rcpp_nmf_dense(A_, mask, tol, maxit, verbose, nonneg, L1, L2, diag, threads, w_init, link_matrix_h, mask_zeros, link_h, sort_model));
    return rcpp_result_gen;
END_RCPP
}
// Rcpp_bipartition_sparse
Rcpp::List Rcpp_bipartition_sparse(const Rcpp::S4& A, const double tol, const unsigned int maxit, const bool nonneg, const std::vector<unsigned int>& samples, const unsigned int seed, const bool verbose, const bool calc_dist, const bool diag);
RcppExport SEXP _RcppML_Rcpp_bipartition_sparse(SEXP ASEXP, SEXP tolSEXP, SEXP maxitSEXP, SEXP nonnegSEXP, SEXP samplesSEXP, SEXP seedSEXP, SEXP verboseSEXP, SEXP calc_distSEXP, SEXP diagSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::S4& >::type A(ASEXP);
    Rcpp::traits::input_parameter< const double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type maxit(maxitSEXP);
    Rcpp::traits::input_parameter< const bool >::type nonneg(nonnegSEXP);
    Rcpp::traits::input_parameter< const std::vector<unsigned int>& >::type samples(samplesSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< const bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< const bool >::type calc_dist(calc_distSEXP);
    Rcpp::traits::input_parameter< const bool >::type diag(diagSEXP);
    rcpp_result_gen = Rcpp::wrap(Rcpp_bipartition_sparse(A, tol, maxit, nonneg, samples, seed, verbose, calc_dist, diag));
    return rcpp_result_gen;
END_RCPP
}
// Rcpp_bipartition_dense
Rcpp::List Rcpp_bipartition_dense(const Eigen::MatrixXd& A, const double tol, const unsigned int maxit, const bool nonneg, const std::vector<unsigned int>& samples, const unsigned int seed, const bool verbose, const bool calc_dist, const bool diag);
RcppExport SEXP _RcppML_Rcpp_bipartition_dense(SEXP ASEXP, SEXP tolSEXP, SEXP maxitSEXP, SEXP nonnegSEXP, SEXP samplesSEXP, SEXP seedSEXP, SEXP verboseSEXP, SEXP calc_distSEXP, SEXP diagSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type A(ASEXP);
    Rcpp::traits::input_parameter< const double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type maxit(maxitSEXP);
    Rcpp::traits::input_parameter< const bool >::type nonneg(nonnegSEXP);
    Rcpp::traits::input_parameter< const std::vector<unsigned int>& >::type samples(samplesSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< const bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< const bool >::type calc_dist(calc_distSEXP);
    Rcpp::traits::input_parameter< const bool >::type diag(diagSEXP);
    rcpp_result_gen = Rcpp::wrap(Rcpp_bipartition_dense(A, tol, maxit, nonneg, samples, seed, verbose, calc_dist, diag));
    return rcpp_result_gen;
END_RCPP
}
// Rcpp_dclust_sparse
Rcpp::List Rcpp_dclust_sparse(const Rcpp::S4& A, const unsigned int min_samples, const double min_dist, const bool verbose, const double tol, const unsigned int maxit, const bool nonneg, const unsigned int seed, const unsigned int threads);
RcppExport SEXP _RcppML_Rcpp_dclust_sparse(SEXP ASEXP, SEXP min_samplesSEXP, SEXP min_distSEXP, SEXP verboseSEXP, SEXP tolSEXP, SEXP maxitSEXP, SEXP nonnegSEXP, SEXP seedSEXP, SEXP threadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Rcpp::S4& >::type A(ASEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type min_samples(min_samplesSEXP);
    Rcpp::traits::input_parameter< const double >::type min_dist(min_distSEXP);
    Rcpp::traits::input_parameter< const bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< const double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type maxit(maxitSEXP);
    Rcpp::traits::input_parameter< const bool >::type nonneg(nonnegSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< const unsigned int >::type threads(threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(Rcpp_dclust_sparse(A, min_samples, min_dist, verbose, tol, maxit, nonneg, seed, threads));
    return rcpp_result_gen;
END_RCPP
}
// nnls
Eigen::MatrixXd nnls(Eigen::MatrixXd a, Eigen::MatrixXd b, unsigned int cd_maxit, const double cd_tol, const bool fast_nnls, const double L1, const double L2, const double PE);
RcppExport SEXP _RcppML_nnls(SEXP aSEXP, SEXP bSEXP, SEXP cd_maxitSEXP, SEXP cd_tolSEXP, SEXP fast_nnlsSEXP, SEXP L1SEXP, SEXP L2SEXP, SEXP PESEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type a(aSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type b(bSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type cd_maxit(cd_maxitSEXP);
    Rcpp::traits::input_parameter< const double >::type cd_tol(cd_tolSEXP);
    Rcpp::traits::input_parameter< const bool >::type fast_nnls(fast_nnlsSEXP);
    Rcpp::traits::input_parameter< const double >::type L1(L1SEXP);
    Rcpp::traits::input_parameter< const double >::type L2(L2SEXP);
    Rcpp::traits::input_parameter< const double >::type PE(PESEXP);
    rcpp_result_gen = Rcpp::wrap(nnls(a, b, cd_maxit, cd_tol, fast_nnls, L1, L2, PE));
    return rcpp_result_gen;
END_RCPP
}
// Rcpp_bipartite_match
Rcpp::List Rcpp_bipartite_match(Rcpp::NumericMatrix x);
RcppExport SEXP _RcppML_Rcpp_bipartite_match(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(Rcpp_bipartite_match(x));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_RcppML_Rcpp_predict_sparse", (DL_FUNC) &_RcppML_Rcpp_predict_sparse, 8},
    {"_RcppML_Rcpp_predict_dense", (DL_FUNC) &_RcppML_Rcpp_predict_dense, 8},
    {"_RcppML_Rcpp_mse_sparse", (DL_FUNC) &_RcppML_Rcpp_mse_sparse, 7},
    {"_RcppML_Rcpp_mse_dense", (DL_FUNC) &_RcppML_Rcpp_mse_dense, 7},
    {"_RcppML_Rcpp_mse_missing_sparse", (DL_FUNC) &_RcppML_Rcpp_mse_missing_sparse, 6},
    {"_RcppML_Rcpp_mse_missing_dense", (DL_FUNC) &_RcppML_Rcpp_mse_missing_dense, 6},
    {"_RcppML_Rcpp_nmf_sparse", (DL_FUNC) &_RcppML_Rcpp_nmf_sparse, 15},
    {"_RcppML_Rcpp_nmf_dense", (DL_FUNC) &_RcppML_Rcpp_nmf_dense, 15},
    {"_RcppML_Rcpp_bipartition_sparse", (DL_FUNC) &_RcppML_Rcpp_bipartition_sparse, 9},
    {"_RcppML_Rcpp_bipartition_dense", (DL_FUNC) &_RcppML_Rcpp_bipartition_dense, 9},
    {"_RcppML_Rcpp_dclust_sparse", (DL_FUNC) &_RcppML_Rcpp_dclust_sparse, 9},
    {"_RcppML_nnls", (DL_FUNC) &_RcppML_nnls, 8},
    {"_RcppML_Rcpp_bipartite_match", (DL_FUNC) &_RcppML_Rcpp_bipartite_match, 1},
    {NULL, NULL, 0}
};

RcppExport void R_init_RcppML(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
