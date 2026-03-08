// This file is part of RcppML, a header-only C++ Machine Learning library
//
// Copyright (C) 2021-2026 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU Public License v. 2.0.

/**
 * @file exceptions.hpp
 * @brief Custom exception types for RcppML
 * 
 * Provides specific exception classes for different error conditions:
 * - InvalidArgument: Invalid parameter value
 * 
 * @author Zach DeBruine
 * @date 2026-01-24
 */

#ifndef FactorNet_CORE_EXCEPTIONS_HPP
#define FactorNet_CORE_EXCEPTIONS_HPP

#include <stdexcept>
#include <string>
#include <sstream>

namespace FactorNet {

/**
 * @brief Exception thrown for invalid argument values
 * 
 * @par Example
 * @code
 * if (k <= 0) {
 *     throw InvalidArgument("k must be positive", k);
 * }
 * @endcode
 */
class InvalidArgument : public std::invalid_argument {
public:
    /**
     * @brief Construct with error message
     * @param msg Description of invalid argument
     */
    explicit InvalidArgument(const std::string& msg)
        : std::invalid_argument("Invalid argument: " + msg) {}
    
    /**
     * @brief Construct with parameter name and value
     * @param param_name Parameter that is invalid
     * @param value Invalid value
     */
    template<typename T>
    InvalidArgument(const std::string& param_name, T value)
        : std::invalid_argument(format_message(param_name, value)) {}
    
private:
    template<typename T>
    static std::string format_message(const std::string& param, T value) {
        std::ostringstream oss;
        oss << "Invalid argument: " << param << " = " << value;
        return oss.str();
    }
};

}  // namespace FactorNet

#endif // FactorNet_CORE_EXCEPTIONS_HPP
