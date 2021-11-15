// This file is part of RcppML, a Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU
// Public License v. 2.0.

#ifndef RcppML_dclust
#define RcppML_dclust

#ifndef RcppML_common
#include <RcppMLCommon.h>
#endif

#ifndef RcppML_bipartition
#include <RcppML/bipartition.hpp>
#endif

struct cluster {
  std::string id;
  std::vector<unsigned int> samples;
  std::vector<double> center;
  double dist;
  bool leaf;
  bool agg;
};

std::vector<unsigned int> indices_that_are_not_leaves(std::vector<cluster>& clusters) {
  std::vector<unsigned int> ind;
  for (unsigned int i = 0; i < clusters.size(); ++i) {
    if (!clusters[i].leaf) ind.push_back(i);
  }
  return ind;
}

namespace RcppML {
  class clusterModel {
  public:
    RcppSparse::Matrix A;
    unsigned int min_samples;
    double min_dist, tol;
    bool nonneg, verbose;
    unsigned int seed, maxit, threads;

    // constructor requiring min_samples and min_dist. All other parameters must be set individually.
    clusterModel(RcppSparse::Matrix& A, const unsigned int min_samples, const double min_dist) : A(A), min_samples(min_samples), min_dist(min_dist) {
      nonneg = true; verbose = true; tol = 1e-4; seed = 0; maxit = 100; threads = 0;
      w = randomMatrix(2, A.rows(), seed);
      calc_dist = (min_dist > 0);
    }

    std::vector<cluster> getClusters() { return clusters; }

    void dclust() {
      if (verbose) Rprintf("\n# of divisions: ");
      std::vector<unsigned int> samples = std::vector<unsigned int>(A.cols());
      std::iota(samples.begin(), samples.end(), (int)0);
      cluster parent_cluster{ "0", samples, centroid(A, samples), 0, samples.size() < min_samples * 2, false };
      clusters.push_back(parent_cluster); // master cluster
      unsigned int n_splits;
      do { // attempt to bipartition all clusters that have not yet been determined to be leaves
        Rcpp::checkUserInterrupt();
        n_splits = 0;
        std::vector<unsigned int> to_split = indices_that_are_not_leaves(clusters);
        std::vector<cluster> new_clusters(to_split.size());
        #ifdef _OPENMP
        #pragma omp parallel for num_threads(threads) schedule(dynamic)
        #endif
        for (unsigned int i = 0; i < to_split.size(); ++i) {
          bipartitionModel p = c_bipartition_sparse(A, w, clusters[to_split[i]].samples, tol, nonneg, calc_dist, maxit, false);
          bool successful_split = (p.size1 > min_samples && p.size2 > min_samples);
          if (calc_dist && successful_split && p.dist < min_dist) successful_split = false;
          if (successful_split) { // bipartition was successful
            new_clusters[i] = cluster{ clusters[to_split[i]].id + "1", p.samples2, p.center2, 0, p.size2 < min_samples * 2, false };
            clusters[to_split[i]] = cluster{ clusters[to_split[i]].id + "0", p.samples1, p.center1, 0, p.size1 < min_samples * 2, false };
            ++n_splits;
          } else { // bipartition was unsuccessful
            clusters[to_split[i]].dist = p.dist;
            clusters[to_split[i]].leaf = true;
          }
        }
        for (unsigned int i = 0; i < new_clusters.size(); ++i)
          if (new_clusters[i].id.size() > 0) clusters.push_back(new_clusters[i]);
        if (verbose) Rprintf(", %u", n_splits);
      } while (n_splits > 0);
      if (verbose) Rprintf("\n");

    }

  private:
    std::vector<cluster> clusters;
    Eigen::MatrixXd w;
    bool calc_dist;
  };
}

#endif