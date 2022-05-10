// This file is part of RcppML, a Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU
// Public License v. 2.0.

#ifndef RcppML_nnls
#define RcppML_nnls

#ifndef RcppML_common
#include <RcppMLCommon.hpp>
#endif

namespace RcppML {
class rng {
public:

  rng(uint32_t state) : state(state) {};
  
  inline uint32_t rand(uint32_t i, uint32_t j) {
    // enforce transpose-identity
    if(j >= i)
      std::swap(i, j);
    
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

#endif
