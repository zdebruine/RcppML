// This file is part of RcppML, an Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
// github.com/zdebruine/RcppML

#include "RcppML.h"

std::vector<unsigned int> indices_that_are_not_leaves(std::vector<clusterModel>& clusters) {
    std::vector<unsigned int> ind;
    for (unsigned int i = 0; i < clusters.size(); ++i)
        if (!clusters[i].leaf) ind.push_back(i);
    return ind;
}

//[[Rcpp::export]]
Rcpp::List Rcpp_dclust_sparse(
  const Rcpp::S4& A_S4, 
  const double min_dist = 0, 
  const unsigned int min_samples = 100, 
  const bool verbose = true,
  const unsigned int threads = 0,
  const double bipartition_tol = 1e-4,
  const bool bipartition_nonneg = true,
  const unsigned int bipartition_maxit = 100,
  const bool calc_centers = true,
  const bool diag = true,
  const unsigned int seed = 0) {

  Rcpp::dgCMatrix A(A_S4);

  bool calc_dist = (min_dist > 0);
  std::vector<clusterModel> clusters;
  std::vector<unsigned int> samples(A.cols());
  std::iota(samples.begin(), samples.end(), (int)0);
  std::vector<double> random_values = getRandomValues(A.cols() * 2, seed);
  
  // initial bipartition
  if (verbose) Rprintf("\n# of divisions: ");
  bipartitionModel p0 = c_bipartition_sparse(A, bipartition_tol, bipartition_nonneg, samples, calc_centers, calc_dist, bipartition_maxit, false, diag, random_values);
  clusters.push_back(clusterModel{ "0", p0.samples1, p0.center1, 0, p0.size1 < min_samples * 2, false });
  clusters.push_back(clusterModel{ "1", p0.samples2, p0.center2, 0, p0.size2 < min_samples * 2, false });

  if (verbose) Rprintf(" 1");
  std::vector<unsigned int> to_split = indices_that_are_not_leaves(clusters), new_clusters();
  unsigned int n_splits = 1;
  while (n_splits > 0) { // attempt to bipartition all clusters that have not yet been determined to be leaves
    n_splits = 0;
    to_split = indices_that_are_not_leaves(clusters);
    std::vector<clusterModel> new_clusters(to_split.size());
    #pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (unsigned int i = 0; i < to_split.size(); ++i) {
      bipartitionModel p = c_bipartition_sparse(A, bipartition_tol, bipartition_nonneg, clusters[to_split[i]].samples, calc_centers, calc_dist, bipartition_maxit, false, diag, random_values);
      bool successful_split = p.size1 > min_samples && p.size2 > min_samples;
      if(calc_dist && successful_split && p.dist < min_dist) successful_split = false;
      if (successful_split) { // bipartition was successful
        new_clusters[i] = clusterModel{ clusters[to_split[i]].id + "1", p.samples2, p.center2, 0, p.size2 < min_samples * 2, false };
        clusters[to_split[i]] = clusterModel{ clusters[to_split[i]].id + "0", p.samples1, p.center1, 0, p.size1 < min_samples * 2, false };
        ++n_splits;
      } else { // bipartition was unsuccessful
        clusters[to_split[i]].dist = p.dist;
        clusters[to_split[i]].leaf = true;
      }
    }
    for (unsigned int i = 0; i < new_clusters.size(); ++i)
      if (new_clusters[i].id.size() > 0) clusters.push_back(new_clusters[i]);
    if (verbose) Rprintf(", %u", n_splits);
  }
  if (verbose) Rprintf("\n");

  Rcpp::List result(clusters.size());
  for (unsigned int i = 0; i < clusters.size(); ++i) {
    result[i] = Rcpp::List::create(
      Rcpp::Named("id") = clusters[i].id,
      Rcpp::Named("samples") = clusters[i].samples,
      Rcpp::Named("center") = clusters[i].center,
      Rcpp::Named("dist") = clusters[i].dist,
      Rcpp::Named("leaf") = clusters[i].leaf);
  }
  return result;
}

//[[Rcpp::export]]
Rcpp::List Rcpp_dclust_dense(
  const Rcpp::NumericMatrix& A, 
  const double min_dist = 0, 
  const unsigned int min_samples = 100, 
  const bool verbose = true,
  const unsigned int threads = 0,
  const double bipartition_tol = 1e-4,
  const bool bipartition_nonneg = true,
  const unsigned int bipartition_maxit = 100,
  const bool calc_centers = true,
  const bool diag = true,
  const unsigned int seed = 0) {

  bool calc_dist = (min_dist > 0);
  std::vector<clusterModel> clusters;
  std::vector<unsigned int> samples(A.cols());
  std::iota(samples.begin(), samples.end(), (int)0);
  std::vector<double> random_values = getRandomValues(A.cols() * 2, seed);
  
  // initial bipartition
  if (verbose) Rprintf("\n# of divisions: ");
  bipartitionModel p0 = c_bipartition_dense(A, bipartition_tol, bipartition_nonneg, samples, calc_centers, calc_dist, bipartition_maxit, false, diag, random_values);
  clusters.push_back(clusterModel{ "0", p0.samples1, p0.center1, 0, p0.size1 < min_samples * 2, false });
  clusters.push_back(clusterModel{ "1", p0.samples2, p0.center2, 0, p0.size2 < min_samples * 2, false });

  if (verbose) Rprintf(" 1");
  std::vector<unsigned int> to_split = indices_that_are_not_leaves(clusters), new_clusters();
  unsigned int n_splits = 1;
  while (n_splits > 0) { // attempt to bipartition all clusters that have not yet been determined to be leaves
    n_splits = 0;
    to_split = indices_that_are_not_leaves(clusters);
    std::vector<clusterModel> new_clusters(to_split.size());
    #pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (unsigned int i = 0; i < to_split.size(); ++i) {
      bipartitionModel p = c_bipartition_dense(A, bipartition_tol, bipartition_nonneg, clusters[to_split[i]].samples, calc_centers, calc_dist, bipartition_maxit, false, diag, random_values);
      bool successful_split = p.size1 > min_samples && p.size2 > min_samples;
      if(calc_dist && successful_split && p.dist < min_dist) successful_split = false;
      if (successful_split) { // bipartition was successful
        new_clusters[i] = clusterModel{ clusters[to_split[i]].id + "1", p.samples2, p.center2, 0, p.size2 < min_samples * 2, false };
        clusters[to_split[i]] = clusterModel{ clusters[to_split[i]].id + "0", p.samples1, p.center1, 0, p.size1 < min_samples * 2, false };
        ++n_splits;
      } else { // bipartition was unsuccessful
        clusters[to_split[i]].dist = p.dist;
        clusters[to_split[i]].leaf = true;
      }
    }
    for (unsigned int i = 0; i < new_clusters.size(); ++i)
      if (new_clusters[i].id.size() > 0) clusters.push_back(new_clusters[i]);
    if (verbose) Rprintf(", %u", n_splits);
  }
  if (verbose) Rprintf("\n");

  Rcpp::List result(clusters.size());
  for (unsigned int i = 0; i < clusters.size(); ++i) {
    result[i] = Rcpp::List::create(
      Rcpp::Named("id") = clusters[i].id,
      Rcpp::Named("samples") = clusters[i].samples,
      Rcpp::Named("center") = clusters[i].center,
      Rcpp::Named("dist") = clusters[i].dist,
      Rcpp::Named("leaf") = clusters[i].leaf);
  }
  return result;
}