#pragma once
// Shared includes and helper types for all RcppFunctions_*.cpp TUs.
// This header is included by each TU; it must not contain any
// [[Rcpp::export]] annotations or function definitions that would be
// compiled into multiple TUs without inline/template qualification.

#include <RcppEigen.h>
#include "../inst/include/FactorNet.h"
#include <RcppHungarian.h>
#include <numeric>  // for std::iota

// ============================================================================
// Utility: Create Eigen::Map from R dgCMatrix (zero-copy)
// ============================================================================

using MappedSpMat = Eigen::Map<const Eigen::SparseMatrix<double>>;

inline MappedSpMat mapSparseMatrix(const Rcpp::S4& s4) {
    Rcpp::IntegerVector dims = s4.slot("Dim");
    Rcpp::IntegerVector i = s4.slot("i");
    Rcpp::IntegerVector p = s4.slot("p");
    Rcpp::NumericVector x = s4.slot("x");
    return MappedSpMat(dims[0], dims[1], x.size(), p.begin(), i.begin(), x.begin());
}

// Print and interrupt functions for Rcpp integration
inline void rcpp_print(const std::string& s) { Rcpp::Rcout << s; }
inline void rcpp_check_interrupt() { Rcpp::checkUserInterrupt(); }

// ============================================================================
// Shared helpers to reduce duplication
// ============================================================================

// Helper: extract graph matrix from R S4 and compute degree vector
struct GraphData {
    Eigen::SparseMatrix<double> matrix;
    Eigen::VectorXd degrees;
    bool valid = false;
};

inline GraphData extract_graph(const Rcpp::Nullable<Rcpp::S4>& graph, double lambda) {
    GraphData gd;
    if (graph.isNotNull() && lambda > 0) {
        gd.matrix = Rcpp::as<Eigen::SparseMatrix<double>>(Rcpp::S4(graph));
        gd.degrees = Eigen::VectorXd::Zero(gd.matrix.cols());
        for (int j = 0; j < gd.matrix.outerSize(); ++j) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(gd.matrix, j); it; ++it) {
                if (it.row() != j) gd.degrees(j) += 1;
            }
        }
        gd.valid = true;
    }
    return gd;
}
