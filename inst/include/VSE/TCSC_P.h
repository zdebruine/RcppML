// This file is part of VSE, Vectorized Sparse Encodings,
//  a C++ library implementing scalable and compressed
//  structures for discrete sparse data
//
// Copyright (C) 2022 Zach DeBruine <debruinz@gvsu.edu>
//
// This source code is subject to the terms of the GPL
// Public License v. 3.0.

#ifndef VSE_TCSC_P_H
#define VSE_TCSC_P_H

//[[Rcpp::depends(RcppEigen)]]
#ifndef RcppEigen__RcppEigen__h
#include <RcppEigen.h>
#endif

// TCSC_P ENCODING
// This is a vectorized tabulated CSC encoding, where values are stored with all corresponding indices in a std::pair<val, std::vector<indices>>
template <typename idx_t, typename val_t>
class TCSC_P {
   private:
    // may also explore storing row_data as a std::deque of std::pair<val_t, std::vector<idx_t>>
    std::vector<std::pair<val_t, std::vector<idx_t>>> row_data;
    size_t nrow;

   public:
    TCSC_P(Eigen::SparseMatrix<val_t> ptr, const size_t col, const size_t num_row) : nrow(num_row) {
        // get all unique values
        std::vector<val_t> unique_values;
        for (typename Eigen::SparseMatrix<val_t>::InnerIterator it(ptr, col); it; ++it)
            if (std::find(unique_values.begin(), unique_values.end(), it.value()) == unique_values.end())
                unique_values.push_back(it.value());

        // loop through all unique values, push back all row indices
        for (auto val : unique_values) {
            std::vector<idx_t> indices_of_val;
            for (typename Eigen::SparseMatrix<val_t>::InnerIterator it(ptr, col); it; ++it)
                if (it.value() == val) indices_of_val.push_back(it.row());
            indices_of_val.shrink_to_fit();
            row_data.push_back(std::pair<val_t, std::vector<idx_t>>(val, indices_of_val));
        }
        row_data.shrink_to_fit();
    }

    size_t mem_usage() {
        // container sizes
        size_t mem_usage = (sizeof(row_data[0]) + sizeof(row_data[0].second)) * row_data.capacity();
        mem_usage += sizeof(row_data);

        // value usage
        mem_usage += sizeof(val_t) * row_data.size();
        for (size_t i = 0; i < row_data.size(); ++i) {
            mem_usage += sizeof(idx_t) * row_data[i].second.capacity();
        }
        return mem_usage;
    }

    // const sparse random access iterator
    class iterator {
       public:
        iterator(TCSC_P& ptr, idx_t idx_pos, idx_t val_pos) : ptr(ptr), index_pos(idx_pos), value_pos(val_pos) {}
        iterator operator++() {
            ++index_pos;
            if (index_pos == ptr.row_data[value_pos].second.size()) {
                ++value_pos;
                index_pos = 0;
            }
            return *this;
        }
        const bool operator!=(const iterator& other) const { return value_pos != other.index(); }
        const val_t& operator*() const { return ptr.row_data[value_pos].first; }
        const idx_t row() const { return ptr.row_data[value_pos].second[index_pos]; }
        const idx_t index() const { return value_pos; }

       private:
        TCSC_P& ptr;
        idx_t index_pos = 0, value_pos = 0;
    };

    iterator begin() { return iterator(*this, (idx_t)0, (idx_t)0); }
    iterator end() { return iterator(*this, row_data[row_data.size() - 1].second.size(), row_data.size()); }

    Rcpp::List view() {
        Rcpp::List l = Rcpp::List::create();
        for (size_t i = 0; i < row_data.size(); ++i) {
            l.push_back(
                Rcpp::List::create(
                    Rcpp::Named("value") = row_data[i].first,
                    Rcpp::Named("index") = row_data[i].second));
        }
        return l;
    }
};

#endif