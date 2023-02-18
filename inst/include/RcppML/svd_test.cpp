#include "svd.hpp"

#define RANK 5

// Generate 
Eigen::MatrixXd A = 100 * Eigen::MatrixXd::Random(100, 1000);

RcppML::svd model(A, RANK);
model.fit();

// plot the error as each K is fit
/*** R
 res <- model.debug_errs
*/