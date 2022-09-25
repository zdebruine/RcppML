// This file is part of VSE, Vectorized Sparse Encodings,
//  a C++ library implementing scalable and compressed
//  structures for discrete sparse data
//
// Copyright (C) 2022 Zach DeBruine <debruinz@gvsu.edu>
//
// This source code is subject to the terms of the GPL
// Public License v. 3.0.

#ifndef VSE_TCSC_NP_H
#define VSE_TCSC_NP_H

//[[Rcpp::depends(RcppEigen)]]
#ifndef RcppEigen__RcppEigen__h
#include <RcppEigen.h>
#endif

// TCSC_NP ENCODING
// This is a vectorized tabulated CSC encoding, where values are stored as negative integers alongside indices as positive integers
// Note that values inherently cannot be negative
template <typename idx_t, typename val_t>
class TCSC_NP {
   private:
    std::vector<idx_t> row_data;
    size_t nrow;

   public:
    TCSC_NP(Eigen::SparseMatrix<val_t> ptr, const size_t col, const size_t num_row) : nrow(num_row) {
        // idx_t must be the same as val_t, because they are stored in the same vector
        static_assert(std::is_same<idx_t, val_t>::value, "idx_t and val_t must be the same for TCSC_NP encoding");

        // idx_t and val_t must be signed integral, because values are distinguished from indices by sign
        static_assert((std::is_integral<idx_t>::value && std::is_signed<idx_t>::value), "idx_t must be a signed integral type");

        // get all unique values
        std::vector<val_t> unique_values;
        for (typename Eigen::SparseMatrix<val_t>::InnerIterator it(ptr, col); it; ++it)
            if (std::find(unique_values.begin(), unique_values.end(), it.value()) == unique_values.end())
                unique_values.push_back(it.value());

        row_data.reserve(unique_values.size() + ptr.outerIndexPtr()[col + 1] - ptr.outerIndexPtr()[col]);

        // loop through all unique values, push back all row indices
        for (auto val : unique_values) {
            row_data.push_back(-std::abs(val));
            for (typename Eigen::SparseMatrix<val_t>::InnerIterator it(ptr, col); it; ++it)
                if (it.value() == val) row_data.push_back(it.row());
        }
        row_data.shrink_to_fit();
    }

    size_t mem_usage() {
        return row_data.capacity() * sizeof(idx_t) + sizeof(row_data);
    }

    // const sparse random access iterator
    class iterator {
       public:
        iterator(TCSC_NP& ptr, idx_t idx_pos) : ptr(ptr), pos(idx_pos) {
            if (idx_pos == 0) {
                pos = 1;
                curr_val = std::abs(ptr.row_data[0]);
            }
        }
        iterator operator++() {
            ++pos;
            if (pos < ptr.row_data.size()) {
                if (ptr.row_data[pos] < 0) {
                    curr_val = std::abs(ptr.row_data[pos]);
                    ++pos;
                }
            }
            return *this;
        }
        const bool operator!=(const iterator& other) const { return pos != other.index(); }
        const val_t& operator*() const { return curr_val; }
        const idx_t row() const { return ptr.row_data[pos]; }
        const idx_t index() const { return pos; }

       private:
        TCSC_NP& ptr;
        idx_t pos = 0;
        val_t curr_val = 0;
    };

    iterator begin() { return iterator(*this, (idx_t)0); }
    iterator end() { return iterator(*this, row_data.size()); }

    std::vector<idx_t> view() {
        return row_data;
    }
};

#endif