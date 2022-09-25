// This file is part of VSE, Vectorized Sparse Encodings,
//  a C++ library implementing scalable and compressed
//  structures for discrete sparse data
//
// Copyright (C) 2022 Zach DeBruine <debruinz@gvsu.edu>
//
// This source code is subject to the terms of the GPL
// Public License v. 3.0.

#ifndef VSE_CSC_H
#define VSE_CSC_H

//[[Rcpp::depends(RcppEigen)]]
#ifndef RcppEigen__RcppEigen__h
#include <RcppEigen.h>
#endif

// CSC ENCODING
// this is typical CSC storage, using two vectors of equal length, one for values and another for corresponding indices
template <typename idx_t, typename val_t>
class CSC {
   private:
    std::vector<idx_t> idx;
    std::vector<val_t> val;
    size_t nrow;

   public:
    CSC(Eigen::SparseMatrix<val_t> ptr, const size_t col, const size_t nrow) : nrow(nrow) {
        idx_t num_nonzero = ptr.outerIndexPtr()[col + 1] - ptr.outerIndexPtr()[col];
        idx.reserve(num_nonzero);
        val.reserve(num_nonzero);
        for (typename Eigen::SparseMatrix<val_t>::InnerIterator it(ptr, col); it; ++it) {
            val.push_back(it.value());
            idx.push_back(it.row());
        }
        idx.shrink_to_fit();
        val.shrink_to_fit();
    }

    size_t mem_usage() {
        size_t mem_used = 0;
        mem_used += idx.capacity() * sizeof(idx_t) + sizeof(idx);
        mem_used += val.capacity() * sizeof(val_t) + sizeof(val);
        return mem_used;
    }

    // const sparse random access iterator
    class iterator {
       public:
        iterator(CSC& ptr, size_t pos) : ptr(ptr), pos(pos) {}
        iterator operator++() {
            ++pos;
            return *this;
        }
        const bool operator!=(const iterator& other) const { return pos != other.index(); }
        const val_t& operator*() const { return ptr.val[pos]; }
        const idx_t row() const { return ptr.idx[pos]; }
        const idx_t index() const { return pos; }

       private:
        CSC& ptr;
        idx_t pos = 0;
    };

    iterator begin() { return iterator(*this, (idx_t)0); }
    iterator end() { return iterator(*this, idx.size()); }

    Rcpp::List view() {
        return Rcpp::List::create(Rcpp::Named("i") = idx, Rcpp::Named("x") = val);
    }
};

#endif