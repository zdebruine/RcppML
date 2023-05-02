// This file is part of RcppML, a Rcpp Machine Learning library
//
// Copyright (C) 2021 Zach DeBruine <zacharydebruine@gmail.com>
//
// This source code is subject to the terms of the GNU
// Public License v. 2.0.

#ifndef RcppML_svd
#define RcppML_svd

#define DIV_OFFSET 1e-15

namespace RcppML
{
  template <class T>
  class svd
  {
  private:
    T &A;
    T t_A;
    Eigen::MatrixXd u;
    Eigen::MatrixXd v;
    Eigen::VectorXd d;
    double tol_ = -1, mse_ = 0;
    unsigned int iter_ = 0, best_model_ = 0;

  public:
    bool verbose = false;
    unsigned int maxit = 100, threads = 0;
    std::vector<double> L1 = std::vector<double>(2), L2 = std::vector<double>(2);

    double tol = 1e-4;

    std::vector<double> debug_errs;

    // CONSTRUCTORS
    // constructor for initialization with a randomly generated "w" matrix
    svd(T &A, const unsigned int k, const unsigned int seed = 0) : A(A)
    {
      u = randomMatrix(A.rows(), k, seed);
      v = Eigen::MatrixXd(A.cols(), k);
      d = Eigen::VectorXd::Ones(k);
    }

    // constructor for initialization with an initial "u" matrix
    svd(T &A, Eigen::MatrixXd u) : A(A), u(u)
    {
      if (A.rows() != u.rows())
        Rcpp::stop("number of rows in 'A' and 'u' are not equal!");
      v = Eigen::MatrixXd(A.cols(), u.cols());
      d = Eigen::VectorXd::Ones(u.cols());
    }

    // constructor for initialization with a fully-specified model
    svd(T &A, Eigen::MatrixXd u, Eigen::MatrixXd v) : A(A), u(u), v(v)
    {
      if (A.rows() != u.rows())
        Rcpp::stop("dimensions of 'u' and 'A' are not compatible");
      if (A.cols() != v.rows())
        Rcpp::stop("dimensions of 'v' and 'A' are not compatible");
      if (u.cols() != v.cols())
        Rcpp::stop("rank of 'u' and 'v' are not equal!");
      d = Eigen::VectorXd::Ones(u.cols());
    }

    // GETTERS
    Eigen::MatrixXd matrixU() { return u; }
    Eigen::MatrixXd matrixV() { return v; }
    Eigen::VectorXd vectorD() { return d; }
    double fit_tol() { return tol_; }
    unsigned int fit_iter() { return iter_; }
    double fit_mse() { return mse_; }
    unsigned int best_model() { return best_model_; }

    // requires specialized dense and sparse backends
    double mse();

    void fit();
    template <int k>
    void fit_rank_k();

    // fit the model multiple times and return the best one
    void fit_restarts(Rcpp::List &u_init)
    {
      Eigen::MatrixXd u_best = u;
      Eigen::MatrixXd v_best = v;
      double tol_best = tol_;
      double mse_best = 0;
      for (unsigned int i = 0; i < u_init.length(); ++i)
      {
        if (verbose)
          Rprintf("Fitting model %i/%i:", i + 1, u_init.length());
        u = Rcpp::as<Eigen::MatrixXd>(u_init[i]);
        tol_ = 1;
        iter_ = 0;
        if (u.rows() != v.rows())
          Rcpp::stop("rank of 'u' is not equal to rank of 'v'");
        if (u.cols() != A.rows())
          Rcpp::stop("dimensions of 'u' and 'A' are not compatible");
        fit();
        mse_ = mse();
        if (verbose)
          Rprintf("MSE: %8.4e\n\n", mse_);
        if (i == 0 || mse_ < mse_best)
        {
          best_model_ = i;
          u_best = u;
          v_best = v;
          tol_best = tol_;
          mse_best = mse_;
        }
      }
      if (best_model_ != (u_init.length() - 1))
      {
        u = u_best;
        v = v_best;
        tol_ = tol_best;
        mse_ = mse_best;
      }
    }
  };

  // svd class methods with specialized dense/sparse backends
  template <>
  double svd<Rcpp::SparseMatrix>::mse()
  {
    Eigen::MatrixXd u0 = u.transpose();

    // compute losses across all samples in parallel
    Eigen::ArrayXd losses(v.cols());
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#endif
    for (unsigned int i = 0; i < v.cols(); ++i)
    {
      Eigen::VectorXd uv_i = u0 * v.col(i);
      for (Rcpp::SparseMatrix::InnerIterator iter(A, i); iter; ++iter)
        uv_i(iter.row()) -= iter.value();
      losses(i) += uv_i.array().square().sum();
    }

    // divide total loss by number of applicable measurements
    return losses.sum() / ((v.cols() * u.cols()));
  }

  template <>
  double svd<Eigen::MatrixXd>::mse()
  {
    Eigen::MatrixXd u0 = u.transpose();
    // compute losses across all samples in parallel
    Eigen::ArrayXd losses(v.cols());

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#endif
    for (unsigned int i = 0; i < v.cols(); ++i)
    {
      Eigen::VectorXd uv_i = u0 * v.col(i);
      for (unsigned int iter = 0; iter < A.rows(); ++iter)
      {
        uv_i(iter) -= A(iter, i);
      }
      losses(i) += uv_i.array().square().sum();
    }

    // divide total loss by number of applicable measurements
    return losses.sum() / ((v.cols() * u.cols()));
  }

  // Recursive static rank update for dense data
  template <>
  template <int k>
  void svd<Eigen::MatrixXd>::fit_rank_k()
  {
    fit_rank_k<k - 1>();
    // alternating least squares updates
    double d_k;
    for (; iter_ < maxit; ++iter_)
    {
      // Save current value of u to track convergence
      Eigen::VectorXd u_it = u.col(k);

      // Calculate relevant values of 'a' matrix
      double akk = u.col(k).dot(u.col(k));
      Eigen::Vector<double, k> a;
      for (int _k = 0; _k < k; ++_k)
        a(_k) = u.col(k).dot(u.col(_k));

      // Update each row of v
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#endif
      for (int i = 0; i < v.rows(); ++i)
      {
        // Calculate b
        v(i, k) = (u.col(k).dot(A.col(i)));
        // Apply L1 penalty
        if (L1[1] > 0)
          v(i, k) -= L1[1];
        
        // Remove contribution of lower ranks from b
        for (int _k = 0; _k < k; ++_k)
        {
          v(i, k) -= a(_k) * v(i, _k);
        }

        // Calculate x
        v(i, k) /= (akk + DIV_OFFSET);
      }

      // Scale V
      v.col(k) /= v.col(k).norm() + DIV_OFFSET;

      // Calculate relevant values of a for update of u
      akk = v.col(k).dot(v.col(k));
      for (int _k = 0; _k < k; ++_k)
        a(_k) = v.col(k).dot(v.col(_k));

      // Update each row of u
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#endif
      for (int i = 0; i < u.rows(); ++i)
      {
        // Calculate b
        u(i, k) = v.col(k).dot(A.row(i));
        // Apply L1 penalty
        if (L1[0] > 0)
          u(i, k) -= L1[0];

        // Remove contribution of lower ranks from b
        for (int _k = 0; _k < k; ++_k)
        {
          u(i, k) -= a(_k) * u(i, _k);
        }

        // Calculate x
        u(i, k) /= (akk + DIV_OFFSET);
      }

      // Scale U
      d_k = u.col(k).norm();
      u.col(k) /= (d_k + DIV_OFFSET);

      // Check early exit criteria
      if (d_k < tol)
      {
        d_k = 0.0;
        break;
      }

      // Check exit criteria
      tol_ = (u.col(k) - u_it).norm();
      if (verbose)
        Rprintf("iteration %4d, rank %4d | %8.2e\n", iter_ + 1, k + 1, tol_);

      if (tol_ < tol)
        break;
      Rcpp::checkUserInterrupt();
    }

    // Undo scaling of U to allow for calculation of next ranks, set d[k]
    u.col(k) *= d_k;
    d(k) = d_k;
  };

  // First rank specialization for dense data
  template <>
  template <>
  void svd<Eigen::MatrixXd>::fit_rank_k<0>()
  {
    // alternating least squares updates
    double d_k;
    for (; iter_ < maxit; ++iter_)
    {
      Eigen::VectorXd u_it = u.col(0);

      // Calculate scalar a
      double akk = u.col(0).dot(u.col(0));
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#endif
      for (int i = 0; i < v.rows(); ++i)
      {
        // Calculate b and apply L1 penalty
        v(i, 0) = (u.col(0).dot(A.col(i)));
        if (L1[1] > 0)
          v(i, 0) -= L1[1];
        // Calculate x
        v(i, 0) /= akk + DIV_OFFSET;
      }

      // Scale V
      v.col(0) /= v.col(0).norm() + DIV_OFFSET;

      // Calculate scalar a for update of u
      akk = v.col(0).dot(v.col(0));
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#endif
      for (int i = 0; i < u.rows(); ++i)
      {
        // Calculate b and apply L1 penalty
        u(i, 0) = v.col(0).dot(A.row(i));\
        if (L1[0] > 0)
          u(i, 0) -= L1[0];
        // Calculate x
        u(i, 0) /= akk + DIV_OFFSET;
      }

      // Scale U
      d_k = u.col(0).norm();
      u.col(0) /= (d_k + DIV_OFFSET);

      // Check early exit criteria
      if (d_k < tol)
      {
        d_k = 0.0;
        break;
      }

      // Check exit criteria
      tol_ = (u.col(0) - u_it).norm();
      if (verbose)
        Rprintf("iteration %4d, rank %4d | %8.2e\n", iter_ + 1, 1, tol_);

      if (tol_ < tol)
        break;
      Rcpp::checkUserInterrupt();
    }

    // "unscale" U
    u.col(0) *= d_k;
    d(0) = d_k;
  };

  // General fit method for dense data of any rank
  template <>
  void svd<Eigen::MatrixXd>::fit()
  {
    // Use static method for ranks 1 to 100
    switch (u.cols())
    {
    case 1:
      fit_rank_k<0>();
    case 2:
      fit_rank_k<1>();
      break;
    case 3:
      fit_rank_k<2>();
      break;
    case 4:
      fit_rank_k<3>();
      break;
    case 5:
      fit_rank_k<4>();
      break;
    case 6:
      fit_rank_k<5>();
      break;
    case 7:
      fit_rank_k<6>();
      break;
    case 8:
      fit_rank_k<7>();
      break;
    case 9:
      fit_rank_k<8>();
      break;
    case 10:
      fit_rank_k<9>();
      break;
    case 11:
      fit_rank_k<10>();
      break;
    case 12:
      fit_rank_k<11>();
      break;
    case 13:
      fit_rank_k<12>();
      break;
    case 14:
      fit_rank_k<13>();
      break;
    case 15:
      fit_rank_k<14>();
      break;
    case 16:
      fit_rank_k<15>();
      break;
    case 17:
      fit_rank_k<16>();
      break;
    case 18:
      fit_rank_k<17>();
      break;
    case 19:
      fit_rank_k<18>();
      break;
    case 20:
      fit_rank_k<19>();
      break;
    case 21:
      fit_rank_k<20>();
      break;
    case 22:
      fit_rank_k<21>();
      break;
    case 23:
      fit_rank_k<22>();
      break;
    case 24:
      fit_rank_k<23>();
      break;
    case 25:
      fit_rank_k<24>();
      break;
    case 26:
      fit_rank_k<25>();
      break;
    case 27:
      fit_rank_k<26>();
      break;
    case 28:
      fit_rank_k<27>();
      break;
    case 29:
      fit_rank_k<28>();
      break;
    case 30:
      fit_rank_k<29>();
      break;
    case 31:
      fit_rank_k<30>();
      break;
    case 32:
      fit_rank_k<31>();
      break;
    case 33:
      fit_rank_k<32>();
      break;
    case 34:
      fit_rank_k<33>();
      break;
    case 35:
      fit_rank_k<34>();
      break;
    case 36:
      fit_rank_k<35>();
      break;
    case 37:
      fit_rank_k<36>();
      break;
    case 38:
      fit_rank_k<37>();
      break;
    case 39:
      fit_rank_k<38>();
      break;
    case 40:
      fit_rank_k<39>();
      break;
    case 41:
      fit_rank_k<40>();
      break;
    case 42:
      fit_rank_k<41>();
      break;
    case 43:
      fit_rank_k<42>();
      break;
    case 44:
      fit_rank_k<43>();
      break;
    case 45:
      fit_rank_k<44>();
      break;
    case 46:
      fit_rank_k<45>();
      break;
    case 47:
      fit_rank_k<46>();
      break;
    case 48:
      fit_rank_k<47>();
      break;
    case 49:
      fit_rank_k<48>();
      break;
    case 50:
      fit_rank_k<49>();
      break;
    case 51:
      fit_rank_k<50>();
      break;
    case 52:
      fit_rank_k<51>();
      break;
    case 53:
      fit_rank_k<52>();
      break;
    case 54:
      fit_rank_k<53>();
      break;
    case 55:
      fit_rank_k<54>();
      break;
    case 56:
      fit_rank_k<55>();
      break;
    case 57:
      fit_rank_k<56>();
      break;
    case 58:
      fit_rank_k<57>();
      break;
    case 59:
      fit_rank_k<58>();
      break;
    case 60:
      fit_rank_k<59>();
      break;
    case 61:
      fit_rank_k<60>();
      break;
    case 62:
      fit_rank_k<61>();
      break;
    case 63:
      fit_rank_k<62>();
      break;
    case 64:
      fit_rank_k<63>();
      break;
    case 65:
      fit_rank_k<64>();
      break;
    case 66:
      fit_rank_k<65>();
      break;
    case 67:
      fit_rank_k<66>();
      break;
    case 68:
      fit_rank_k<67>();
      break;
    case 69:
      fit_rank_k<68>();
      break;
    case 70:
      fit_rank_k<69>();
      break;
    case 71:
      fit_rank_k<70>();
      break;
    case 72:
      fit_rank_k<71>();
      break;
    case 73:
      fit_rank_k<72>();
      break;
    case 74:
      fit_rank_k<73>();
      break;
    case 75:
      fit_rank_k<74>();
      break;
    case 76:
      fit_rank_k<75>();
      break;
    case 77:
      fit_rank_k<76>();
      break;
    case 78:
      fit_rank_k<77>();
      break;
    case 79:
      fit_rank_k<78>();
      break;
    case 80:
      fit_rank_k<79>();
      break;
    case 81:
      fit_rank_k<80>();
      break;
    case 82:
      fit_rank_k<81>();
      break;
    case 83:
      fit_rank_k<82>();
      break;
    case 84:
      fit_rank_k<83>();
      break;
    case 85:
      fit_rank_k<84>();
      break;
    case 86:
      fit_rank_k<85>();
      break;
    case 87:
      fit_rank_k<86>();
      break;
    case 88:
      fit_rank_k<87>();
      break;
    case 89:
      fit_rank_k<88>();
      break;
    case 90:
      fit_rank_k<89>();
      break;
    case 91:
      fit_rank_k<90>();
      break;
    case 92:
      fit_rank_k<91>();
      break;
    case 93:
      fit_rank_k<92>();
      break;
    case 94:
      fit_rank_k<93>();
      break;
    case 95:
      fit_rank_k<94>();
      break;
    case 96:
      fit_rank_k<95>();
      break;
    case 97:
      fit_rank_k<96>();
      break;
    case 98:
      fit_rank_k<97>();
      break;
    case 99:
      fit_rank_k<98>();
      break;
    case 100:
      fit_rank_k<99>();
      break;
    default:
      fit_rank_k<99>();
      // General case for rank 101+
      for (int k = 100; k < u.cols(); ++k)
      {
        // alternating least squares updates
        double d_k;
        for (; iter_ < maxit; ++iter_)
        {
          Eigen::VectorXd u_it = u.col(k);

          // Calculate values of a for update of v
          double akk = u.col(k).dot(u.col(k));
          Eigen::VectorXd a(k);
          for (int _k = 0; _k < k; ++_k)
            a(_k) = u.col(k).dot(u.col(_k));

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#endif
          for (int i = 0; i < v.rows(); ++i)
          {
            // Calculate b
            v(i, k) = (u.col(k).dot(A.col(i)));
            if (L1[1] > 0)
              v(i, k) -= L1[1];

            // Remove contributions of previous ranks
            for (int _k = 0; _k < k; ++_k)
            {
              v(i, k) -= a(_k) * v(i, _k);
            }

            // Calculate x
            v(i, k) /= (akk + DIV_OFFSET);
          }

          // Scale V
          v.col(k) /= v.col(k).norm() + DIV_OFFSET;

          // Update U
          akk = v.col(k).dot(v.col(k));
          for (int _k = 0; _k < k; ++_k)
            a(_k) = v.col(k).dot(v.col(_k));

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#endif
          for (int i = 0; i < u.rows(); ++i)
          {
            // Calculate b
            u(i, k) = v.col(k).dot(A.row(i));
            if (L1[0] > 0)
              u(i, k) -= L1[0];

            // Remove contribution of previous ranks
            for (int _k = 0; _k < k; ++_k)
            {
              u(i, k) -= a(_k) * u(i, _k);
            }

            // Calculate x
            u(i, k) /= (akk + DIV_OFFSET);
          }

          // Scale U
          d_k = u.col(k).norm();
          u.col(k) /= (d_k + DIV_OFFSET);

          // Check early exit criteria
          if (d_k < tol)
          {
            d_k = 0.0;
            break;
          }

          // Check exit criteria
          tol_ = (u.col(k) - u_it).cwiseAbs().sum();
          if (verbose)
            Rprintf("iteration %4d, rank %4d | %8.2e\n", iter_ + 1, k + 1, tol_);

          if (tol_ < tol)
            break;
          Rcpp::checkUserInterrupt();
        }

        // "unscale" U
        u.col(k) *= d_k;
        d(k) = d_k;
      }
      break;
    }

    // Scale u, since this is undone after each rank update
    for (int i = 0; i < u.cols(); ++i)
      u.col(i) /= (d(i) + DIV_OFFSET);

    if (tol_ > tol && iter_ == maxit && verbose)
      Rprintf(" convergence not reached in %d iterations\n  (actual tol = %4.2e, target tol = %4.2e)\n", iter_, tol_, tol);
  };

  // Recursive static rank update for sparse data
  template <>
  template <int k>
  void svd<Rcpp::SparseMatrix>::fit_rank_k()
  {
    fit_rank_k<k - 1>();

    Rcpp::SparseMatrix At = A.transpose();

    // alternating least squares updates
    double d_k;
    for (; iter_ < maxit; ++iter_)
    {
      Eigen::VectorXd u_it = u.col(k);

      double akk = u.col(k).dot(u.col(k));
      Eigen::Vector<double, k> a;
      for (int _k = 0; _k < k; ++_k)
        a(_k) = u.col(k).dot(u.col(_k));

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#endif
      for (int i = 0; i < v.rows(); ++i)
      {
        // Calculate b element-by-element
        v(i, k) = 0.0;
        for (Rcpp::SparseMatrix::InnerIterator iter(A, i); iter; ++iter)
          v(i, k) += u(iter.row(), k) * iter.value();

        if (L1[1] > 0)
          v(i, k) -= L1[1];

        for (int _k = 0; _k < k; ++_k)
        {
          v(i, k) -= a(_k) * v(i, _k);
        }

        v(i, k) /= akk + DIV_OFFSET;
      }

      // Scale V
      v.col(k) /= v.col(k).norm() + DIV_OFFSET;

      // Update U
      akk = v.col(k).dot(v.col(k));
      for (int _k = 0; _k < k; ++_k)
        a(_k) = v.col(k).dot(v.col(_k));

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#endif
      for (int i = 0; i < u.rows(); ++i)
      {
        // Calculate b element-by-element
        u(i, k) = 0.0;
        for (Rcpp::SparseMatrix::InnerIterator iter(At, i); iter; ++iter)
          u(i, k) += v(iter.row(), k) * iter.value();

        if (L1[0] > 0)
          u(i, k) -= L1[0];

        for (int _k = 0; _k < k; ++_k)
        {
          u(i, k) -= a(_k) * u(i, _k);
        }

        u(i, k) /= akk + DIV_OFFSET;
      }

      // Scale U
      d_k = u.col(k).norm();
      u.col(k) /= (d_k + DIV_OFFSET);

      // Check early exit criteria
      if (d_k < tol)
      {
        d_k = 0.0;
        break;
      }

      // Check exit criteria
      tol_ = (u.col(k) - u_it).norm();
      if (verbose)
        Rprintf("iteration %4d, rank %4d | %8.2e\n", iter_ + 1, k + 1, tol_);

      if (tol_ < tol)
        break;
      Rcpp::checkUserInterrupt();
    }

    // "unscale" U
    u.col(k) *= d_k;
    d(k) = d_k;
  };

  // First rank specialization for sparse data
  template <>
  template <>
  void svd<Rcpp::SparseMatrix>::fit_rank_k<0>()
  {
    Rcpp::SparseMatrix At = A.transpose();

    // alternating least squares updates
    double d_k;
    for (; iter_ < maxit; ++iter_)
    {
      Eigen::VectorXd u_it = u.col(0);

      double akk = u.col(0).dot(u.col(0));
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#endif
      for (int i = 0; i < v.rows(); ++i)
      {
        v(i, 0) = 0.0;
        for (Rcpp::SparseMatrix::InnerIterator iter(A, i); iter; ++iter)
          v(i, 0) += u(iter.row(), 0) * iter.value();
        if (L1[1] > 0)
          v(i, 0) -= L1[1];

        v(i, 0) /= akk + DIV_OFFSET;
      }

      // Scale V
      v.col(0) /= v.col(0).norm() + DIV_OFFSET;

      // Update U
      akk = v.col(0).dot(v.col(0));

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#endif
      for (int i = 0; i < u.rows(); ++i)
      {
        u(i, 0) = 0.0;
        for (Rcpp::SparseMatrix::InnerIterator iter(At, i); iter; ++iter)
          u(i, 0) += v(iter.row(), 0) * iter.value();

        if (L1[0] > 0)
          u(i, 0) -= L1[0];

        u(i, 0) /= akk + DIV_OFFSET;
      }

      // Scale U
      d_k = u.col(0).norm();
      u.col(0) /= (d_k + DIV_OFFSET);

      // Check early exit criteria
      if (d_k < tol)
      {
        d_k = 0.0;
        break;
      }

      // Check exit criteria
      tol_ = (u.col(0) - u_it).norm();
      if (verbose)
        Rprintf("iteration %4d, rank %4d | %8.2e\n", iter_ + 1, 1, tol_);

      if (tol_ < tol)
        break;
      Rcpp::checkUserInterrupt();
    }

    // "unscale" U
    u.col(0) *= d_k;
    d(0) = d_k;
  }

  // General fit method for sparse data of any rank
  template <>
  void svd<Rcpp::SparseMatrix>::fit()
  {
    switch (u.cols())
    {
    case 1:
      fit_rank_k<0>();
    case 2:
      fit_rank_k<1>();
      break;
    case 3:
      fit_rank_k<2>();
      break;
    case 4:
      fit_rank_k<3>();
      break;
    case 5:
      fit_rank_k<4>();
      break;
    case 6:
      fit_rank_k<5>();
      break;
    case 7:
      fit_rank_k<6>();
      break;
    case 8:
      fit_rank_k<7>();
      break;
    case 9:
      fit_rank_k<8>();
      break;
    case 10:
      fit_rank_k<9>();
      break;
    case 11:
      fit_rank_k<10>();
      break;
    case 12:
      fit_rank_k<11>();
      break;
    case 13:
      fit_rank_k<12>();
      break;
    case 14:
      fit_rank_k<13>();
      break;
    case 15:
      fit_rank_k<14>();
      break;
    case 16:
      fit_rank_k<15>();
      break;
    case 17:
      fit_rank_k<16>();
      break;
    case 18:
      fit_rank_k<17>();
      break;
    case 19:
      fit_rank_k<18>();
      break;
    case 20:
      fit_rank_k<19>();
      break;
    case 21:
      fit_rank_k<20>();
      break;
    case 22:
      fit_rank_k<21>();
      break;
    case 23:
      fit_rank_k<22>();
      break;
    case 24:
      fit_rank_k<23>();
      break;
    case 25:
      fit_rank_k<24>();
      break;
    case 26:
      fit_rank_k<25>();
      break;
    case 27:
      fit_rank_k<26>();
      break;
    case 28:
      fit_rank_k<27>();
      break;
    case 29:
      fit_rank_k<28>();
      break;
    case 30:
      fit_rank_k<29>();
      break;
    case 31:
      fit_rank_k<30>();
      break;
    case 32:
      fit_rank_k<31>();
      break;
    case 33:
      fit_rank_k<32>();
      break;
    case 34:
      fit_rank_k<33>();
      break;
    case 35:
      fit_rank_k<34>();
      break;
    case 36:
      fit_rank_k<35>();
      break;
    case 37:
      fit_rank_k<36>();
      break;
    case 38:
      fit_rank_k<37>();
      break;
    case 39:
      fit_rank_k<38>();
      break;
    case 40:
      fit_rank_k<39>();
      break;
    case 41:
      fit_rank_k<40>();
      break;
    case 42:
      fit_rank_k<41>();
      break;
    case 43:
      fit_rank_k<42>();
      break;
    case 44:
      fit_rank_k<43>();
      break;
    case 45:
      fit_rank_k<44>();
      break;
    case 46:
      fit_rank_k<45>();
      break;
    case 47:
      fit_rank_k<46>();
      break;
    case 48:
      fit_rank_k<47>();
      break;
    case 49:
      fit_rank_k<48>();
      break;
    case 50:
      fit_rank_k<49>();
      break;
    case 51:
      fit_rank_k<50>();
      break;
    case 52:
      fit_rank_k<51>();
      break;
    case 53:
      fit_rank_k<52>();
      break;
    case 54:
      fit_rank_k<53>();
      break;
    case 55:
      fit_rank_k<54>();
      break;
    case 56:
      fit_rank_k<55>();
      break;
    case 57:
      fit_rank_k<56>();
      break;
    case 58:
      fit_rank_k<57>();
      break;
    case 59:
      fit_rank_k<58>();
      break;
    case 60:
      fit_rank_k<59>();
      break;
    case 61:
      fit_rank_k<60>();
      break;
    case 62:
      fit_rank_k<61>();
      break;
    case 63:
      fit_rank_k<62>();
      break;
    case 64:
      fit_rank_k<63>();
      break;
    case 65:
      fit_rank_k<64>();
      break;
    case 66:
      fit_rank_k<65>();
      break;
    case 67:
      fit_rank_k<66>();
      break;
    case 68:
      fit_rank_k<67>();
      break;
    case 69:
      fit_rank_k<68>();
      break;
    case 70:
      fit_rank_k<69>();
      break;
    case 71:
      fit_rank_k<70>();
      break;
    case 72:
      fit_rank_k<71>();
      break;
    case 73:
      fit_rank_k<72>();
      break;
    case 74:
      fit_rank_k<73>();
      break;
    case 75:
      fit_rank_k<74>();
      break;
    case 76:
      fit_rank_k<75>();
      break;
    case 77:
      fit_rank_k<76>();
      break;
    case 78:
      fit_rank_k<77>();
      break;
    case 79:
      fit_rank_k<78>();
      break;
    case 80:
      fit_rank_k<79>();
      break;
    case 81:
      fit_rank_k<80>();
      break;
    case 82:
      fit_rank_k<81>();
      break;
    case 83:
      fit_rank_k<82>();
      break;
    case 84:
      fit_rank_k<83>();
      break;
    case 85:
      fit_rank_k<84>();
      break;
    case 86:
      fit_rank_k<85>();
      break;
    case 87:
      fit_rank_k<86>();
      break;
    case 88:
      fit_rank_k<87>();
      break;
    case 89:
      fit_rank_k<88>();
      break;
    case 90:
      fit_rank_k<89>();
      break;
    case 91:
      fit_rank_k<90>();
      break;
    case 92:
      fit_rank_k<91>();
      break;
    case 93:
      fit_rank_k<92>();
      break;
    case 94:
      fit_rank_k<93>();
      break;
    case 95:
      fit_rank_k<94>();
      break;
    case 96:
      fit_rank_k<95>();
      break;
    case 97:
      fit_rank_k<96>();
      break;
    case 98:
      fit_rank_k<97>();
      break;
    case 99:
      fit_rank_k<98>();
      break;
    case 100:
      fit_rank_k<99>();
      break;
    default:
      fit_rank_k<99>();
      Rcpp::SparseMatrix At = A.transpose();
      for (int k = 100; k < u.cols(); ++k)
      {
        // alternating least squares updates
        double d_k;
        for (; iter_ < maxit; ++iter_)
        {
          Eigen::VectorXd u_it = u.col(k);

          double akk = u.col(k).dot(u.col(k));
          Eigen::VectorXd a(k);
          for (int _k = 0; _k < k; ++_k)
            a(_k) = u.col(k).dot(u.col(_k));

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#endif
          for (int i = 0; i < v.rows(); ++i)
          {
            v(i, k) = 0.0;
            for (Rcpp::SparseMatrix::InnerIterator iter(A, i); iter; ++iter)
              v(i, k) += u(iter.row(), k) * iter.value();
            if (L1[1] > 0)
              v(i, k) -= L1[1];

            for (int _k = 0; _k < k; ++_k)
            {
              v(i, k) -= a(_k) * v(i, _k);
            }

            v(i, k) /= akk + DIV_OFFSET;
          }

          // Scale V
          v.col(k) /= v.col(k).norm() + DIV_OFFSET;

          // Update U
          akk = v.col(k).dot(v.col(k));
          for (int _k = 0; _k < k; ++_k)
            a(_k) = v.col(k).dot(v.col(_k));

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#endif
          for (int i = 0; i < u.rows(); ++i)
          {
            u(i, k) = 0.0;
            for (Rcpp::SparseMatrix::InnerIterator iter(At, i); iter; ++iter)
              u(i, k) += v(iter.row(), k) * iter.value();            
            if (L1[0] > 0)
              u(i, k) -= L1[0];


            for (int _k = 0; _k < k; ++_k)
            {
              u(i, k) -= a(_k) * u(i, _k);
            }

            u(i, k) /= akk + DIV_OFFSET;
          }

          // Scale U
          d_k = u.col(k).norm();
          u.col(k) /= (d_k + DIV_OFFSET);

          // Check early exit criteria
          if (d_k < tol)
          {
            d_k = 0.0;
            break;
          }

          // Check exit criteria
          tol_ = (u.col(k) - u_it).cwiseAbs().sum();
          if (verbose)
            Rprintf("iteration %4d, rank %4d | %8.2e\n", iter_ + 1, k + 1, tol_);

          if (tol_ < tol)
            break;
          Rcpp::checkUserInterrupt();
        }

        // "unscale" U
        u.col(k) *= d_k;
        d(k) = d_k;
      }
    }
    // Scale u
    for (int i = 0; i < u.cols(); ++i)
      u.col(i) /= (d(i) + DIV_OFFSET);

    if (tol_ > tol && iter_ == maxit && verbose)
      Rprintf(" convergence not reached in %d iterations\n  (actual tol = %4.2e, target tol = %4.2e)\n", iter_, tol_, tol);
  };

} // namespace RcppML

#endif