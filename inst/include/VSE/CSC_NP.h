// This file is part of VSE, Vectorized Sparse Encodings,
//  a C++ library implementing scalable and compressed
//  structures for discrete sparse data
//
// Copyright (C) 2022 Zach DeBruine <debruinz@gvsu.edu>
//
// This source code is subject to the terms of the GPL
// Public License v. 3.0.

#ifndef VSE_CSC_NP_H
#define VSE_CSC_NP_H

//[[Rcpp::depends(RcppEigen)]]
#ifndef RcppEigen__RcppEigen__h
#include <RcppEigen.h>
#endif

// CSC_NP ENCODING
// This is a vectorized CSC encoding, where value/index pairs are stored contiguously in a std::vector
// Note that the idx_t must also be the val_t, and values cannot be less than 0.
template <typename idx_t, typename val_t>
class CSC_NP {
   private:
    std::vector<idx_t> row_data;
    size_t nrow;

   public:
    CSC_NP(Eigen::SparseMatrix<val_t> ptr, const size_t col, const size_t num_row) : nrow(num_row) {
        // idx_t must be the same as val_t, because they are stored in the same vector
        static_assert(std::is_same<idx_t, val_t>::value, "idx_t and val_t must be the same for CSC_NP encoding");

        idx_t num_nonzero = ptr.outerIndexPtr()[col + 1] - ptr.outerIndexPtr()[col];
        row_data.reserve(num_nonzero * 2);

        for (typename Eigen::SparseMatrix<val_t>::InnerIterator it(ptr, col); it; ++it) {
            row_data.push_back(it.row());
            row_data.push_back(it.value());
        }
        row_data.shrink_to_fit();
    }

    size_t mem_usage() {
        return row_data.capacity() * sizeof(idx_t) + sizeof(row_data);
    }

    // const sparse random access iterator
    class iterator {
       public:
        iterator(CSC_NP& ptr, size_t pos) : ptr(ptr), pos(pos) {}
        iterator operator++() {
            pos += 2;
            return *this;
        }
        const bool operator!=(const iterator& other) const { return pos != other.index(); }
        const val_t& operator*() const { return ptr.row_data[pos + 1]; }
        const idx_t row() const { return ptr.row_data[pos]; }
        const idx_t index() const { return pos; }

       private:
        CSC_NP& ptr;
        idx_t pos = 0;
    };

    iterator begin() { return iterator(*this, (idx_t)0); }
    iterator end() { return iterator(*this, row_data.size()); }

    std::vector<idx_t> view() {
        return row_data;
    }
};

#endif