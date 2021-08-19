// This file is part of RcppML, a Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU
// Public License v. 2.0.

#ifndef RcppML_bipartition
#define RcppML_bipartition

#ifndef RcppML_common
#include <RcppMLCommon.hpp>
#endif

#ifndef RcppML_nmf2
#include <RcppML/nmf2.hpp>
#endif

// compute cluster centroid given an ipx sparse matrix and samples in the cluster center
std::vector<double> centroid(Rcpp::dgCMatrix& A, const std::vector<unsigned int>& samples) {

    std::vector<double> center(A.rows());
    for (unsigned int s = 0; s < samples.size(); ++s)
      for (Rcpp::dgCMatrix::InnerIterator it(A, samples[s]); it; ++it)
        center[it.row()] += it.value();
    for (unsigned int j = 0; j < A.rows(); ++j) center[j] /= samples.size();

    return center;
}

// dense version
std::vector<double> centroid(const Rcpp::NumericMatrix& A, const std::vector<unsigned int>& samples) {

    std::vector<double> center(A.rows());
    for (unsigned int s = 0; s < samples.size(); ++s)
      for (unsigned int r = 0; r < A.rows(); ++r)
        center[r] += A(r, samples[s]);
    for (unsigned int j = 0; j < A.rows(); ++j) center[j] /= samples.size();

    return center;
}

// cosine distance of cells in a cluster to assigned cluster center (in_center) vs. other cluster center (out_cluster),
// divided by the cosine distance to assigned cluster center
//
// tot_dist is given by sum for all samples of cosine distance to cognate cluster (ci) - distance to non-cognate cluster (cj)
//   divided by distance to cognate cluster (ci):
// cosine dist to c_i, dci = sqrt(x cross c_i) / (sqrt(c_i cross c_i) * sqrt(x cross x))
// cosine dist to c_j, dcj = sqrt(x cross c_j) / (sqrt(c_j cross c_j) * sqrt(x cross x))
// tot_dist = (dci - dcj) / dci
// this expression simplifies to 1 - (sqrt(c_j cross x) * sqrt(c_i cross c_i)) / (sqrt(c_i cross x) * sqrt(c_j cross c_j))
double rel_cosine(Rcpp::dgCMatrix& A, const std::vector<unsigned int>& samples1, const std::vector<unsigned int>& samples2,
             const std::vector<double>& center1, const std::vector<double>& center2) {

    double center1_innerprod = std::sqrt(std::inner_product(center1.begin(), center1.end(), center1.begin(), (double)0));
    double center2_innerprod = std::sqrt(std::inner_product(center2.begin(), center2.end(), center2.begin(), (double)0));
    double dist1 = 0, dist2 = 0;
    for (unsigned int s = 0; s < samples1.size(); ++s) {
        double x1_center1 = 0, x1_center2 = 0;
        for (Rcpp::dgCMatrix::InnerIterator it(A, samples1[s]); it; ++it){
          x1_center1 += center1[it.row()] * it.value();
          x1_center2 += center2[it.row()] * it.value();
        }
        dist1 += (std::sqrt(x1_center2) * center1_innerprod) / (std::sqrt(x1_center1) * center2_innerprod);
    }
    for (unsigned int s = 0; s < samples2.size(); ++s) {
        double x2_center1 = 0, x2_center2 = 0;
        for (Rcpp::dgCMatrix::InnerIterator it(A, samples2[s]); it; ++it){
          x2_center1 += center1[it.row()] * it.value();
          x2_center2 += center2[it.row()] * it.value();
        }
        dist2 += (std::sqrt(x2_center1) * center2_innerprod) / (std::sqrt(x2_center2) * center1_innerprod);
    }
    return (dist1 + dist2) / (2 * A.rows());
}

double rel_cosine(const Rcpp::NumericMatrix& A, const std::vector<unsigned int>& samples1, const std::vector<unsigned int>& samples2,
             const std::vector<double>& center1, const std::vector<double>& center2) {

    double center1_innerprod = std::sqrt(std::inner_product(center1.begin(), center1.end(), center1.begin(), (double)0));
    double center2_innerprod = std::sqrt(std::inner_product(center2.begin(), center2.end(), center2.begin(), (double)0));
    double dist1 = 0, dist2 = 0;
    for (unsigned int s = 0; s < samples1.size(); ++s) {
        double x1_center1 = 0, x1_center2 = 0;
        for (unsigned int r = 0; r < A.rows(); ++r){
          x1_center1 += center1[r] * A(r, samples1[s]);
          x1_center2 += center2[r] * A(r, samples1[s]);
        }
        dist1 += (std::sqrt(x1_center2) * center1_innerprod) / (std::sqrt(x1_center1) * center2_innerprod);
    }
    for (unsigned int s = 0; s < samples2.size(); ++s) {
        double x2_center1 = 0, x2_center2 = 0;
        for (unsigned int r = 0; r < A.rows(); ++r){
          x2_center1 += center1[r] * A(r, samples2[s]);
          x2_center2 += center2[r] * A(r, samples2[s]);
        }
        dist2 += (std::sqrt(x2_center1) * center2_innerprod) / (std::sqrt(x2_center2) * center1_innerprod);
    }
    return (dist1 + dist2) / (2 * A.rows());
}

bipartitionModel c_bipartition_sparse(
  Rcpp::dgCMatrix& A,
  const double tol,
  const bool nonneg,
  const std::vector<unsigned int> samples,
  bool calc_centers,
  bool calc_dist, 
  const unsigned int maxit,
  const bool verbose,
  const bool diag, 
  const std::vector<double>& random_values) {

  unsigned int size1 = 0, size2 = 0;
  std::vector<double> center1(A.rows()), center2(A.rows());
  double dist = -1;

  Eigen::MatrixXd w = randomMatrix(2, A.rows(), random_values);
  wdhmodel m = c_nmf2_sparse(A, w, tol, nonneg, maxit, verbose, diag, samples);
  std::vector<double> v(samples.size());
  for (unsigned int j = 0; j < samples.size(); ++j){
    v[j] = m.h(0, j) - m.h(1, j);
    v[j] > 0 ? ++size1 : ++size2;
  }
  std::vector<unsigned int> samples1(size1), samples2(size2);

  // get indices of samples in both clusters
  unsigned int s1 = 0, s2 = 0;
  for (unsigned int j = 0; j < samples.size(); ++j){
    if(v[j] > 0){ samples1[s1] = samples[j]; ++s1; } 
    else { samples2[s2] = samples[j]; ++s2; }
  }

  // calculate the centers of both clusters
  if(calc_centers){
    center1 = centroid(A, samples1);
    center2 = centroid(A, samples2);

    // calculate relative cosine similarity of all samples to ((assigned - other) / assigned) cluster
    if (calc_dist) 
      dist = rel_cosine(A, samples1, samples2, center1, center2);
  }

  return bipartitionModel {v, dist, size1, size2, samples1, samples2, center1, center2};
}

bipartitionModel c_bipartition_dense(
  const Rcpp::NumericMatrix& A,
  const double tol,
  const bool nonneg,
  const std::vector<unsigned int> samples,
  bool calc_centers,
  bool calc_dist, 
  const unsigned int maxit,
  const bool verbose,
  const bool diag,
  const std::vector<double>& random_values) {

  unsigned int size1 = 0, size2 = 0;
  std::vector<double> center1(A.rows()), center2(A.rows());
  double dist = -1;

  Eigen::MatrixXd w = randomMatrix(2, A.rows(), random_values);
  wdhmodel m = c_nmf2_dense(A, w, tol, nonneg, maxit, verbose, diag, samples);
  std::vector<double> v(samples.size());
  for (unsigned int j = 0; j < samples.size(); ++j){
    v[j] = m.h(0, j) - m.h(1, j);
    v[j] > 0 ? ++size1 : ++size2;
  }
  std::vector<unsigned int> samples1(size1), samples2(size2);

  // get indices of samples in both clusters
  unsigned int s1 = 0, s2 = 0;
  for (unsigned int j = 0; j < samples.size(); ++j){
    if(v[j] > 0){ samples1[s1] = samples[j]; ++s1; } 
    else { samples2[s2] = samples[j]; ++s2; }
  }

  // calculate the centers of both clusters
  if(calc_centers){
    center1 = centroid(A, samples1);
    center2 = centroid(A, samples2);

    // calculate relative cosine similarity of all samples to ((assigned - other) / assigned) cluster
    if (calc_dist) 
      dist = rel_cosine(A, samples1, samples2, center1, center2);
  }
  return bipartitionModel {v, dist, size1, size2, samples1, samples2, center1, center2};
}

#endif