// This file is part of RcppML, an Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
// github.com/zdebruine/RcppML

#include "RcppML.h"

// calculate sort index of vector "d" in decreasing order
std::vector<int> sort_index(const Eigen::VectorXd& d) {
  std::vector<int> idx(d.size());
  std::iota(idx.begin(), idx.end(), 0);
  sort(idx.begin(), idx.end(), [&d](size_t i1, size_t i2) {return d[i1] > d[i2];});
  return idx;
}

// reorder rows in dynamic matrix "x" by integer vector "ind"
Eigen::MatrixXd reorder_rows(const Eigen::MatrixXd& x, const std::vector<int>& ind) {
  Eigen::MatrixXd x_reordered(x.rows(), x.cols());
  for (unsigned int i = 0; i < ind.size(); ++i)
    x_reordered.row(i) = x.row(ind[i]);
  return x_reordered;
}

// reorder elements in vector "x" by integer vector "ind"
Eigen::VectorXd reorder(const Eigen::VectorXd& x, const std::vector<int>& ind) {
  Eigen::VectorXd x_reordered(x.size());
  for (unsigned int i = 0; i < ind.size(); ++i)
    x_reordered(i) = x(ind[i]);
  return x_reordered;
}

double cor(Eigen::MatrixXd& x, Eigen::MatrixXd& y) {
  double x_i, y_i, sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0, sum_y2 = 0;
  const unsigned int n = x.size();
  for (unsigned int i = 0; i < n; ++i) {
    x_i = (*(x.data() + i));
    y_i = (*(y.data() + i));
    sum_x += x_i;
    sum_y += y_i;
    sum_xy += x_i * y_i;
    sum_x2 += x_i * x_i;
    sum_y2 += y_i * y_i;
  }
  return 1 - (n * sum_xy - sum_x * sum_y) / std::sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));
}

std::vector<double> getRandomValues(const unsigned int len, const unsigned int seed){

  if(seed > 0){
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(std::floor(std::fabs(seed)));
  }
  Rcpp::NumericVector R_RNG_random_values = Rcpp::runif(len);
  std::vector<double> random_values = Rcpp::as<std::vector<double> >(R_RNG_random_values);

  return random_values;
}

Eigen::MatrixXd randomMatrix(const unsigned int nrow, const unsigned int ncol, const std::vector<double>& random_values){

  Eigen::MatrixXd x(nrow, ncol);
  unsigned int indx = 0;
  for(unsigned int r = 0; r < nrow; ++r)
    for(unsigned int c = 0; c < ncol; ++c, ++indx)
      x(r, c) = random_values[indx];
  return x;
}