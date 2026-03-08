// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file resource_tags.hpp
 * @brief Resource tag types for CPU/GPU dispatch
 *
 * Single definition point for resource tags, included by both
 * storage.hpp and primitives/primitives.hpp.
 */

#pragma once

namespace FactorNet {
namespace primitives {

/// CPU compute resource tag
struct CPU {};

/// GPU compute resource tag
struct GPU {};

}  // namespace primitives
}  // namespace FactorNet

