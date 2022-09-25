// This file is part of VSE, Vectorized Sparse Encodings,
//  a C++ library implementing scalable and compressed
//  structures for discrete sparse data
//
// Copyright (C) 2022 Zach DeBruine <debruinz@gvsu.edu>
//
// This source code is subject to the terms of the GPL
// Public License v. 3.0.

#ifndef VSE_CSC_P_H
#define VSE_CSC_P_H

//[[Rcpp::depends(RcppEigen)]]
#ifndef RcppEigen__RcppEigen__h
#include <RcppEigen.h>
#endif

// CSC_P ENCODING
// This is a vectorized CSC encoding, where value/index pairs are stored in a single vector using std::pair
// Note that std::pair supports type polymorphism (idx_t and val_t)
template <typename idx_t, typename val_t>
class CSC_P {
   private:
    std::vector<std::pair<idx_t, val_t>> row_data;
    size_t nrow;

   public:
    CSC_P(Eigen::SparseMatrix<val_t> ptr, const size_t col, const size_t num_row) : nrow(num_row) {
        idx_t num_nonzero = ptr.outerIndexPtr()[col + 1] - ptr.outerIndexPtr()[col];
        row_data.reserve(num_nonzero);
        for (typename Eigen::SparseMatrix<val_t>::InnerIterator it(ptr, col); it; ++it) {
            row_data.push_back(std::pair<idx_t, val_t>(it.row(), it.value()));
        }
        row_data.shrink_to_fit();
    }

    size_t mem_usage() {
        size_t mem_used = (sizeof(val_t) + sizeof(idx_t) + sizeof(row_data[0]));
        mem_used *= row_data.capacity();
        return mem_used;
    }

    // const sparse random access iterator
    class iterator {
       public:
        iterator(CSC_P& ptr, size_t pos) : ptr(ptr), pos(pos) {}
        iterator operator++() {
            ++pos;
            return *this;
        }
        const bool operator!=(const iterator& other) const { return pos != other.index(); }
        const val_t& operator*() const { return ptr.row_data[pos].second; }
        const idx_t row() const { return ptr.row_data[pos].first; }
        const idx_t index() const { return pos; }

       private:
        CSC_P& ptr;
        idx_t pos = 0;
    };

    iterator begin() { return iterator(*this, (idx_t)0); }
    iterator end() { return iterator(*this, row_data.size()); }

    Rcpp::List view() {
        Rcpp::List l = Rcpp::List::create(0);
        for (size_t i = 0; i < row_data.size(); ++i) {
            l.push_back(
                Rcpp::List::create(
                    Rcpp::Named("index") = row_data[i].first,
                    Rcpp::Named("value") = row_data[i].second));
        }
        return l;
    }
};

#endif