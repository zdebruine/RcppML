// This file is part of RcppML, a Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU
// Public License v. 2.0.

#ifndef RcppML_bipartition
#define RcppML_bipartition

#ifndef RcppML_nnls
#include <RcppML/nnls.hpp>
#endif

struct bipartitionModel {
  std::vector<double> v;
  double dist;
  unsigned int size1;
  unsigned int size2;
  std::vector<unsigned int> samples1;
  std::vector<unsigned int> samples2;
  std::vector<double> center1;
  std::vector<double> center2;
};

// compute cluster centroid given an ipx sparse matrix and samples in the cluster center
inline std::vector<double> centroid(RcppSparse::Matrix& A, const std::vector<unsigned int>& samples) {

  std::vector<double> center(A.rows());
  for (unsigned int s = 0; s < samples.size(); ++s)
    for (RcppSparse::Matrix::InnerIterator it(A, samples[s]); it; ++it)
      center[it.row()] += it.value();
  for (unsigned int j = 0; j < A.rows(); ++j) center[j] /= samples.size();

  return center;
}

// dense version
inline std::vector<double> centroid(const Eigen::MatrixXd& A, const std::vector<unsigned int>& samples) {

  std::vector<double> center(A.rows());
  for (unsigned int s = 0; s < samples.size(); ++s)
    for (int r = 0; r < A.rows(); ++r)
      center[r] += A(r, samples[s]);
  for (int j = 0; j < A.rows(); ++j) center[j] /= samples.size();

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
inline double rel_cosine(RcppSparse::Matrix& A, const std::vector<unsigned int>& samples1, const std::vector<unsigned int>& samples2,
                         const std::vector<double>& center1, const std::vector<double>& center2) {

  double center1_innerprod = std::sqrt(std::inner_product(center1.begin(), center1.end(), center1.begin(), (double)0));
  double center2_innerprod = std::sqrt(std::inner_product(center2.begin(), center2.end(), center2.begin(), (double)0));
  double dist1 = 0, dist2 = 0;
  for (unsigned int s = 0; s < samples1.size(); ++s) {
    double x1_center1 = 0, x1_center2 = 0;
    for (RcppSparse::Matrix::InnerIterator it(A, samples1[s]); it; ++it) {
      x1_center1 += center1[it.row()] * it.value();
      x1_center2 += center2[it.row()] * it.value();
    }
    dist1 += (std::sqrt(x1_center2) * center1_innerprod) / (std::sqrt(x1_center1) * center2_innerprod);
  }
  for (unsigned int s = 0; s < samples2.size(); ++s) {
    double x2_center1 = 0, x2_center2 = 0;
    for (RcppSparse::Matrix::InnerIterator it(A, samples2[s]); it; ++it) {
      x2_center1 += center1[it.row()] * it.value();
      x2_center2 += center2[it.row()] * it.value();
    }
    dist2 += (std::sqrt(x2_center1) * center2_innerprod) / (std::sqrt(x2_center2) * center1_innerprod);
  }
  return (dist1 + dist2) / (2 * A.rows());
}

inline double rel_cosine(const Eigen::MatrixXd& A, const std::vector<unsigned int>& samples1, const std::vector<unsigned int>& samples2,
                         const std::vector<double>& center1, const std::vector<double>& center2) {

  double center1_innerprod = std::sqrt(std::inner_product(center1.begin(), center1.end(), center1.begin(), (double)0));
  double center2_innerprod = std::sqrt(std::inner_product(center2.begin(), center2.end(), center2.begin(), (double)0));
  double dist1 = 0, dist2 = 0;
  for (unsigned int s = 0; s < samples1.size(); ++s) {
    double x1_center1 = 0, x1_center2 = 0;
    for (int r = 0; r < A.rows(); ++r) {
      x1_center1 += center1[r] * A(r, samples1[s]);
      x1_center2 += center2[r] * A(r, samples1[s]);
    }
    dist1 += (std::sqrt(x1_center2) * center1_innerprod) / (std::sqrt(x1_center1) * center2_innerprod);
  }
  for (unsigned int s = 0; s < samples2.size(); ++s) {
    double x2_center1 = 0, x2_center2 = 0;
    for (int r = 0; r < A.rows(); ++r) {
      x2_center1 += center1[r] * A(r, samples2[s]);
      x2_center2 += center2[r] * A(r, samples2[s]);
    }
    dist2 += (std::sqrt(x2_center1) * center2_innerprod) / (std::sqrt(x2_center2) * center1_innerprod);
  }
  return (dist1 + dist2) / (2 * A.rows());
}

void scale(Eigen::VectorXd& d, Eigen::MatrixXd& w) {
  d = w.rowwise().sum();
  d.array() += TINY_NUM;
  for (unsigned int i = 0; i < w.rows(); ++i)
    for (unsigned int j = 0; j < w.cols(); ++j)
      w(i, j) /= d(i);
}

inline bipartitionModel c_bipartition_sparse(
  RcppSparse::Matrix& A,
  Eigen::MatrixXd w,
  const std::vector<unsigned int> samples,
  const double tol,
  const bool nonneg,
  const bool calc_dist,
  const unsigned int maxit,
  const bool verbose) {

  // rank-2 nmf
  Eigen::MatrixXd w_it, h(w.rows(), samples.size());
  Eigen::VectorXd d = Eigen::VectorXd::Ones(2);
  if (verbose) Rprintf("\n%4s | %8s \n---------------\n", "iter", "tol");
  double tol_ = 1;
  for (unsigned int iter = 0; iter < maxit && tol_ > tol; ++iter) {
    w_it = w;

    // update h
    Eigen::Matrix2d a = w * w.transpose();
    double denom = a(0, 0) * a(1, 1) - a(0, 1) * a(0, 1);
    for (unsigned int i = 0; i < h.cols(); ++i) {
      Eigen::Vector2d b(0, 0);
      for (RcppSparse::Matrix::InnerIterator it(A, samples[i]); it; ++it) {
        const double val = it.value();
        const unsigned int r = it.row();
        b(0) += val * w(0, r);
        b(1) += val * w(1, r);
      }
      nnls2(a, b, denom, h, i, nonneg);
    }
    scale(d, h);

    // update w
    a = h * h.transpose();
    denom = a(0, 0) * a(1, 1) - a(0, 1) * a(0, 1);
    w.setZero();
    for (unsigned int i = 0; i < h.cols(); ++i) {
      for (RcppSparse::Matrix::InnerIterator it(A, samples[i]); it; ++it)
        for (unsigned int j = 0; j < 2; ++j)
          w(j, it.row()) += it.value() * h(j, i);
    }
    nnls2InPlace(a, denom, w, nonneg);
    scale(d, w);

    tol_ = cor(w, w_it);
    if (verbose) Rprintf("%4d | %8.2e\n", iter + 1, tol_);
  }

  // calculate bipartitioning vector
  unsigned int size1 = 0, size2 = 0;
  std::vector<double> v(h.cols()), center1(w.cols()), center2(w.cols());
  if (d(0) > d(1)) {
    for (unsigned int j = 0; j < h.cols(); ++j) {
      v[j] = h(0, j) - h(1, j);
      v[j] > 0 ? ++size1 : ++size2;
    }
  } else {
    for (unsigned int j = 0; j < h.cols(); ++j) {
      v[j] = h(1, j) - h(0, j);
      v[j] > 0 ? ++size1 : ++size2;
    }
  }

  std::vector<unsigned int> samples1(size1), samples2(size2);
  double dist = -1;

  // get indices of samples in both clusters
  unsigned int s1 = 0, s2 = 0;
  for (unsigned int j = 0; j < h.cols(); ++j) {
    if (v[j] > 0) { samples1[s1] = samples[j]; ++s1; } else { samples2[s2] = samples[j]; ++s2; }
  }

  if (calc_dist) {
    // calculate the centers of both clusters
    center1 = centroid(A, samples1);
    center2 = centroid(A, samples2);

    // calculate relative cosine similarity of all samples to ((assigned - other) / assigned) cluster
    dist = rel_cosine(A, samples1, samples2, center1, center2);
  }

  return bipartitionModel{ v, dist, size1, size2, samples1, samples2, center1, center2 };
}

inline bipartitionModel c_bipartition_dense(
  const Eigen::MatrixXd& A,
  Eigen::MatrixXd w,
  const std::vector<unsigned int> samples,
  const double tol,
  const bool nonneg,
  const bool calc_dist,
  const unsigned int maxit,
  const bool verbose) {

  // rank-2 nmf
  Eigen::MatrixXd w_it, h(w.rows(), samples.size());
  Eigen::VectorXd d = Eigen::VectorXd::Ones(2);
  if (verbose) Rprintf("\n%4s | %8s \n---------------\n", "iter", "tol");
  double tol_ = 1;
  for (unsigned int iter = 0; iter < maxit && tol_ > tol; ++iter) {
    w_it = w;

    // update h
    Eigen::Matrix2d a = w * w.transpose();
    double denom = a(0, 0) * a(1, 1) - a(0, 1) * a(0, 1);
    for (unsigned int i = 0; i < h.cols(); ++i) {
      Eigen::Vector2d b(0, 0);
      for (int j = 0; j < A.rows(); ++j) {
        const double val = A(j, samples[i]);
        b(0) += val * w(0, j);
        b(1) += val * w(1, j);
      }
      nnls2(a, b, denom, h, i, nonneg);
    }
    scale(d, h);

    // update w
    a = h * h.transpose();
    denom = a(0, 0) * a(1, 1) - a(0, 1) * a(0, 1);
    w.setZero();
    for (unsigned int i = 0; i < h.cols(); ++i) {
      for (int j = 0; j < A.rows(); ++j)
        for (unsigned int l = 0; l < 2; ++l)
          w(l, j) += A(j, samples[i]) * h(l, i);
    }
    nnls2InPlace(a, denom, w, nonneg);
    scale(d, w);

    tol_ = cor(w, w_it);
    if (verbose) Rprintf("%4d | %8.2e\n", iter + 1, tol_);
  }

  // calculate bipartitioning vector
  unsigned int size1 = 0, size2 = 0;
  std::vector<double> v(h.cols()), center1(w.cols()), center2(w.cols());
  if (d(0) > d(1)) {
    for (unsigned int j = 0; j < h.cols(); ++j) {
      v[j] = h(0, j) - h(1, j);
      v[j] > 0 ? ++size1 : ++size2;
    }
  } else {
    for (unsigned int j = 0; j < h.cols(); ++j) {
      v[j] = h(1, j) - h(0, j);
      v[j] > 0 ? ++size1 : ++size2;
    }
  }

  std::vector<unsigned int> samples1(size1), samples2(size2);
  double dist = -1;

  // get indices of samples in both clusters
  unsigned int s1 = 0, s2 = 0;
  for (unsigned int j = 0; j < h.cols(); ++j) {
    if (v[j] > 0) { samples1[s1] = samples[j]; ++s1; } else { samples2[s2] = samples[j]; ++s2; }
  }

  if (calc_dist) {
    // calculate the centers of both clusters
    center1 = centroid(A, samples1);
    center2 = centroid(A, samples2);

    // calculate relative cosine similarity of all samples to ((assigned - other) / assigned) cluster
    dist = rel_cosine(A, samples1, samples2, center1, center2);
  }

  return bipartitionModel{ v, dist, size1, size2, samples1, samples2, center1, center2 };
}

#endif