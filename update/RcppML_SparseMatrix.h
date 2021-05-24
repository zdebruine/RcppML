// This file is part of RcppML, an RcppEigen template library
//  for machine learning.
//
// Copyright (C) 2021 Zach DeBruine <zach.debruine@gmail.com>
// github.com/zdebruine/RcppML

#ifndef RCPPML_SPARSEMATRIX_H
#define RCPPML_SPARSEMATRIX_H

#ifndef RCPPML_H
#include "RcppML.h"
#endif

namespace RcppML {

    class SparseMatrix {
    public:
        Rcpp::IntegerVector i, p, Dim;
        Rcpp::NumericVector x;

        // Constructor from Rcpp::S4 object
        SparseMatrix(Rcpp::S4 m) : i(m.slot("i")), p(m.slot("p")), Dim(m.slot("Dim")), x(m.slot("x")) {}
        SparseMatrix(Rcpp::IntegerVector i, Rcpp::IntegerVector p, Rcpp::IntegerVector Dim, Rcpp::NumericVector x) : i(i), p(p), Dim(Dim), x(x) {}

        // Convert to Rcpp::S4 object
        Rcpp::S4 as_S4(SparseMatrix& m) {
            Rcpp::S4 s(std::string("dgCMatrix"));
            s.slot("i") = m.i;
            s.slot("p") = m.p;
            s.slot("x") = m.x;
            s.slot("Dim") = Rcpp::IntegerVector::create(m.rows(), m.cols());
            return s;
        }

        int rows() { return Dim[0]; }
        int cols() { return Dim[1]; }

        class InnerIterator {
        public:
            InnerIterator(SparseMatrix& ptr, int col) : ptr(ptr) { index = ptr.p[col]; max_index = ptr.p[col + 1]; }
            operator bool() const { return (index < max_index); }
            InnerIterator& operator++() { ++index; return *this; }
            const double& operator*() const { return ptr.x[index]; }
            int row() const { return ptr.i[index]; }
        private:
            SparseMatrix& ptr;
            int index, max_index;
        };

    };

} // namespace RcppML

#endif // RCPPML_SPARSEMATRIX_H