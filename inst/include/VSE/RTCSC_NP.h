// This file is part of VSE, Vectorized Sparse Encodings,
//  a C++ library implementing scalable and compressed
//  structures for discrete sparse data
//
// Copyright (C) 2022 Zach DeBruine <debruinz@gvsu.edu>
//
// This source code is subject to the terms of the GPL
// Public License v. 3.0.

#ifndef VSE_RTCSC_NP_H
#define VSE_RTCSC_NP_H

//[[Rcpp::depends(RcppEigen)]]
#ifndef RcppEigen__RcppEigen__h
#include <RcppEigen.h>
#endif

// TRCSC_NP ENCODING
// This is a vectorized run-length tabulated CSC encoding, where integral values are followed by number of indices with that value,
//  followed by all indices with that value
// Note that values can be signed, but must be integral and of the same type as indices
template <typename idx_t, typename val_t>
class TRCSC_NP {
   private:
    std::vector<idx_t> row_data;
    size_t nrow;

   public:
    TRCSC_NP(Eigen::SparseMatrix<val_t> ptr, const size_t col, const size_t num_row) : nrow(num_row) {
        // idx_t must be the same as val_t, because they are stored in the same vector
        static_assert(std::is_same<idx_t, val_t>::value, "idx_t and val_t must be the same for TRCSC_NP encoding");

        // idx_t and val_t must be integral
        static_assert((std::is_integral<idx_t>::value), "idx_t must be a signed integral type");

        // get all unique values
        std::vector<val_t> unique_values;
        for (typename Eigen::SparseMatrix<val_t>::InnerIterator it(ptr, col); it; ++it)
            if (std::find(unique_values.begin(), unique_values.end(), it.value()) == unique_values.end())
                unique_values.push_back(it.value());

        // tabulate up number of occurrences for each value
        std::vector<idx_t> values_table(unique_values.size());
        for (typename Eigen::SparseMatrix<val_t>::InnerIterator it(ptr, col); it; ++it) {
            idx_t ind = 0;
            while (unique_values[ind] != it.value()) ++ind;
            values_table[ind] += 1;
        }

        row_data.reserve(unique_values.size() * 2 + ptr.outerIndexPtr()[col + 1] - ptr.outerIndexPtr()[col]);

        // loop through all unique values, push back all row indices
        for (size_t i = 0; i < unique_values.size(); ++i) {
            row_data.push_back(unique_values[i]);
            row_data.push_back(values_table[i]);
            for (typename Eigen::SparseMatrix<val_t>::InnerIterator it(ptr, col); it; ++it)
                if (it.value() == unique_values[i]) row_data.push_back(it.row());
        }
        row_data.shrink_to_fit();
    }

    size_t mem_usage() {
        return row_data.capacity() * sizeof(idx_t) + sizeof(row_data);
    }

    // const sparse random access iterator
    class iterator {
       public:
        iterator(TRCSC_NP& ptr, idx_t idx_pos) : ptr(ptr), pos(idx_pos) {
            if (idx_pos == 0) {
                pos = 2;
                curr_val = ptr.row_data[0];
                curr_run = ptr.row_data[1];
            }
        }
        iterator operator++() {
            --curr_run;
            if (curr_run == 0 && pos != (ptr.row_data.size() - 1)) {
                curr_val = ptr.row_data[pos + 1];
                curr_run = ptr.row_data[pos + 2];
                pos += 3;
            } else {
                ++pos;
            }
            return *this;
        }
        const bool operator!=(const iterator& other) const { return pos != other.index(); }
        const val_t& operator*() const { return curr_val; }
        const idx_t row() const { return ptr.row_data[pos]; }
        const idx_t index() const { return pos; }

       private:
        TRCSC_NP& ptr;
        idx_t pos = 0;
        val_t curr_val = 0;
        idx_t curr_run = 0;
    };

    iterator begin() { return iterator(*this, (idx_t)0); }
    iterator end() { return iterator(*this, row_data.size()); }

    std::vector<idx_t> view() {
        return row_data;
    }
};

#endif