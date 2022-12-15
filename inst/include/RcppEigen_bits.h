// This file is adapted from RcppEigen headers
//
// It supports R bindings for
//   * Eigen::Matrix<T, -1, -1>
//   * Eigen::Matrix<T, -1, 1>
//
// Copyright (C)      2011 Douglas Bates, Dirk Eddelbuettel and Romain Francois
//

#ifndef RcppEigen_bits
#define RcppEigen_bits

#include <RcppCommon.h>

#include "EigenCore"

/* forward declarations */
namespace Rcpp {
namespace traits {
template <typename T>
class Exporter<Eigen::Matrix<T, -1, -1> >;
template <typename T>
class Exporter<Eigen::Matrix<T, -1, 1> >;
}  // namespace traits
}  // namespace Rcpp

#include <Rcpp.h>

namespace Rcpp {
namespace RcppEigen {

// helper trait to identify if T is a plain object type
template <typename T>
struct is_plain : Rcpp::traits::same_type<T, typename T::PlainObject> {};

// helper trait to identify if the object has dense storage
template <typename T>
struct is_dense : Rcpp::traits::same_type<typename T::StorageKind, Eigen::Dense> {};

// for plain dense objects
template <typename T>
SEXP eigen_wrap_plain_dense(const T& obj, Rcpp::traits::true_type) {
    typename Eigen::internal::conditional<T::IsRowMajor,
                                          Eigen::Matrix<typename T::Scalar,
                                                        T::RowsAtCompileTime,
                                                        T::ColsAtCompileTime>,
                                          const T&>::type objCopy(obj);
    int m = obj.rows(), n = obj.cols();
    R_xlen_t size = static_cast<R_xlen_t>(m) * n;
    SEXP ans = PROTECT(::Rcpp::wrap(objCopy.data(), objCopy.data() + size));
    if (T::ColsAtCompileTime != 1) {
        SEXP dd = PROTECT(::Rf_allocVector(INTSXP, 2));
        int* d = INTEGER(dd);
        d[0] = m;
        d[1] = n;
        ::Rf_setAttrib(ans, R_DimSymbol, dd);
        UNPROTECT(1);
    }
    UNPROTECT(1);
    return ans;
}

// plain object, so we can assume data() and size()
template <typename T>
inline SEXP eigen_wrap_is_plain(const T& obj, ::Rcpp::traits::true_type) {
    return eigen_wrap_plain_dense(obj, typename is_dense<T>::type());
}

// when the object is not plain, we need to eval()uate it
template <typename T>
inline SEXP eigen_wrap_is_plain(const T& obj, ::Rcpp::traits::false_type) {
    return eigen_wrap_is_plain(obj.eval(), Rcpp::traits::true_type());
}

template <typename T>
inline SEXP eigen_wrap(const T& obj) {
    return eigen_wrap_is_plain(obj,
                               typename is_plain<T>::type());
}

}  // namespace RcppEigen

namespace traits {
/* support for Rcpp::as */

template <typename T, typename value_type>
class MatrixExporterForEigen {
   public:
    typedef value_type r_export_type;

    MatrixExporterForEigen(SEXP x) : object(x) {}
    ~MatrixExporterForEigen() {}

    T get() {
        Shield<SEXP> dims(::Rf_getAttrib(object, R_DimSymbol));
        if (Rf_isNull(dims) || ::Rf_length(dims) != 2) {
            throw ::Rcpp::not_a_matrix();
        }
        int* dims_ = INTEGER(dims);
        T result(dims_[0], dims_[1]);
        value_type* data = result.data();
        ::Rcpp::internal::export_indexing<value_type*, value_type>(object, data);
        return result;
    }

   private:
    SEXP object;
};

// export a dense vector
template <typename T>
class Exporter<Eigen::Matrix<T, -1, 1> >
    : public IndexingExporter<Eigen::Matrix<T, -1, 1>, T> {
   public:
    Exporter(SEXP x) : IndexingExporter<Eigen::Matrix<T, -1, 1>, T>(x) {}
};

// export a dense matrix
template <typename T>
class Exporter<Eigen::Matrix<T, -1, -1> >
    : public MatrixExporterForEigen<Eigen::Matrix<T, -1, -1>, T> {
   public:
    Exporter(SEXP x) : MatrixExporterForEigen<Eigen::Matrix<T, -1, -1>, T>(x) {}
};

}  // namespace traits
}  // namespace Rcpp

#endif