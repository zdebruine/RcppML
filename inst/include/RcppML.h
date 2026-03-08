// Backward-compatibility shim: RcppML.h -> FactorNet.h
// The C++ library has been renamed to FactorNet.
// This header exists so that auto-generated RcppExports.cpp
// (which includes "RcppML.h") continues to compile.

#ifndef RcppML_H
#define RcppML_H

#include "FactorNet.h"

// Backward-compat namespace alias
namespace RcppML = FactorNet;

#endif  // RcppML_H
