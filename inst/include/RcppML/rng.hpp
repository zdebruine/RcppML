// This file is part of RcppML, a Rcpp Machine Learning library
//
// Copyright (C) 2022 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU
// Public License v. 2.0.

#ifndef RcppML_rng
#define RcppML_rng

#ifndef RcppML_common
#include <RcppMLCommon.hpp>
#endif

namespace RcppML {
template <bool transpose_identical>
class rng {
public:
 
  rng(uint32_t state) : state(state) {};
 
  inline uint32_t rand(uint32_t i, uint32_t j) {
    // enforce transpose-identity
    if(transpose_identical){
      if(j >= i){
        std::swap(i, j);
      }
    }
     
      // generate a unique hash of i and j, using (max(i, j))(max(i, j) + 1) / 2 + min(i, j)
      // https://math.stackexchange.com/questions/882877/produce-unique-number-given-two-integers
      // credit to user @JimmyK4542, and whoever published the original intuition
      // also add 1 to i and j to avoid issues where i == 0 || j == 0
      // transform to uint64_t to avoid issues with overflow during multiplication
      uint64_t ij = (i + 1) * (i + 2) / 2 + j + 1;
     
      // adapted from xorshift64, Marsaglia
      // https://en.wikipedia.org/wiki/Xorshift
      ij ^= ij << 13 | (i << 17);
      ij ^= ij >> 7 | (j << 5);
      ij ^= ij << 17;
     
      // adapted from xorshift128+
      // https://xoshiro.di.unimi.it/xorshift128plus.c
      uint64_t s = state ^ ij;
      s ^= s << 23;
      s = s ^ ij ^ (s >> 18) ^ (ij >> 5);
      return (uint32_t) ((s + ij));
  }
 
  template <typename T>
  inline T sample(uint32_t i, uint32_t j, const T max){
    return rand(i, j) % max;
  }
 
  template <typename T>
  inline T runif(uint32_t i, uint32_t j){
    T y = (T) rand(i, j) / UINT32_MAX;
    return y - std::floor(y);
  }
 
private:
  const uint32_t state;
};
}

template <typename T>
Eigen::Matrix<T, -1, -1> rti_matrix(uint32_t nrow, uint32_t ncol, uint32_t rng){
  Eigen::Matrix<T, -1, -1> m(nrow, ncol);
  RcppML::rng<true> s(rng);
 
  // symmetric part first
  uint32_t n_sym = (nrow < ncol) ? nrow : ncol;
 
  for(uint32_t i = 0; i < n_sym; ++i){
    for(uint32_t j = (i + 1); j < n_sym; ++j){
      T tmp = s.runif<T>(i, j);
      m(i, j) = tmp;
      m(j, i) = tmp;
    }
  }
 
  // populate the diagonal of the symmetric part
  for(uint32_t i = 0; i < n_sym; ++i)
    m(i, i) = s.runif<T>(i, i);
 
  // asymmetric part (but still transpose-identical)
  if(nrow > ncol)
    for(uint32_t i = n_sym; i < nrow; ++i)
      for(uint32_t j = 0; j < ncol; ++j)
        m(i, j) = s.runif<T>(i, j);
  else if (ncol > nrow)
    for(uint32_t i = 0; i < nrow; ++i)
      for(uint32_t j = n_sym; j < ncol; ++j)
        m(i, j) = s.runif<T>(i, j);
 
  return m;
}

template <typename T>
Eigen::Matrix<T, -1, -1> r_matrix(uint32_t nrow, uint32_t ncol, uint32_t rng){
  Eigen::Matrix<T, -1, -1> m(nrow, ncol);
  RcppML::rng<false> s(rng);
  for(uint32_t i = 0; i < nrow; ++i)
    for(uint32_t j = 0; j < ncol; ++j)
      m(i, j) = s.runif<T>(i, j);
  return m;
}

#endif
