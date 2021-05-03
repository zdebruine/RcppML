// 4/12/2021 Zach DeBruine <zach.debruine@vai.org
// Please raise issues on github.com/zdebruine/divvy/issues

// [[Rcpp::plugins(openmp)]]
#include <omp.h>
#include <rcpp.h>

// Rcpp for sparse matrices (spRcpp)
namespace Rcpp {
    class SparseMatrix {
    public:
        Rcpp::IntegerVector i, p;
        Rcpp::NumericVector x;
        int n_rows, n_cols;

        // constructors
        SparseMatrix(unsigned int nrow, unsigned int ncol) { n_rows = nrow; n_cols = ncol; p = Rcpp::IntegerVector(ncol); };
        SparseMatrix(Rcpp::IntegerVector& A_i, Rcpp::IntegerVector& A_p, Rcpp::NumericVector& A_x, unsigned int nrow) {
            i = A_i;
            p = A_p;
            x = A_x;
            n_rows = nrow;
            n_cols = A_p.size() - 1;
        };
        SparseMatrix(Rcpp::S4 mat) {
            Rcpp::IntegerVector dim = mat.slot("Dim");
            i = mat.slot("i");
            p = mat.slot("p");
            x = mat.slot("x");
            n_rows = (int)dim[0];
            n_cols = (int)dim[1];
        };

        unsigned int begin_col(unsigned int x) { return p[x]; };
        unsigned int end_col(unsigned int x) { return p[x + 1]; };
        /*
        class const_iterator {
        public:
            unsigned int index;
            /*
            const_iterator& operator++() { const_iterator it; it.index = this.index + 1; return it; }
            bool operator!=(unsigned int x) { return this.index != x; };
            double operator*() { return x[this.index]; };
            unsigned int row() { return i[this.index]; };
            unsigned int col() {
                if (p.size() < 2) return 0;
                const unsigned int target = this.index;
                for (unsigned int j = 1; j < p.size(); ++j)
                    if (target > p[j]) return j - 1;
            };

        };*/
    };
}

// convert S4 R object to SpRcpp::SpMatrix
namespace Rcpp {
    template <> Rcpp::SparseMatrix as(SEXP mat) {
        return Rcpp::SparseMatrix(mat);
    }
}

/*
for (unsigned int j = p[samples[s]]; j < p[samples[s] + 1]; ++j) {
    wb1[it.row()] += *it * h1[col];
    wb2[it.row()] += *it * h2[col];
}

wb += A.col(i);
wb *= h.row(i);
*/

//[[Rcpp::export]]
Rcpp::NumericVector toRcppSparseMatrix(Rcpp::SparseMatrix& A) {
    return A.x;
}

/*
//[[Rcpp::export]]
std::vector<double> SpRcpp_colsums(Rcpp::S4& mat) {
    SpRcpp::SpMatrix<double> A(mat);
    std::vector<double> sums(A.n_col);
    for (unsigned int col = 0; col < A.n_col; ++col) {
        double sum = 0;
        for (SpRcpp::SpMatrix<double>::const_iterator it = A.begin_col(col); it != A.end_col(col); ++it)
            sum += *it;
        sums[col] = sum;
    }
    return sums;
}*/

// structure definitions
template <typename T>
struct wh2 {
    std::vector<T> w1;
    std::vector<T> w2;
    std::vector<T> h1;
    std::vector<T> h2;
    std::vector<T> tol;
};

template <typename T>
struct uv {
    std::vector<T> u;
    std::vector<T> v;
};

template <typename T>
struct bipartitionModel {
    std::vector<bool> v;
    T dist;
    unsigned int size1;
    unsigned int size2;
    std::vector<unsigned int> samples1;
    std::vector<unsigned int> samples2;
    std::vector<T> center1;
    std::vector<T> center2;
};

template <typename T>
struct clusterModel {
    std::string id;
    std::vector<unsigned int> samples;
    std::vector<T> center;
    T dist;
    bool leaf;
    bool agg;
};

// 1 - R^2 between two std::vector objects
template<typename T>
T cor(std::vector<T>& x, std::vector<T>& y) {
    T sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0, sum_y2 = 0;
    for (unsigned int i = 0; i < x.size(); ++i) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
        sum_y2 += y[i] * y[i];
    }
    return 1 - (x.size() * sum_xy - sum_x * sum_y) / std::sqrt((x.size() * sum_x2 - sum_x * sum_x) * (x.size() * sum_y2 - sum_y * sum_y));
}

// ******************************************************************************************** HELPER FUNCTIONS THAT ARE ALWAYS SINGLE-THREADED
// implicit diagonalization of h and w to l2-norm, then multiplication through by the diagonal
template<typename T>
void l2norm(std::vector<T>& h1, std::vector<T>& h2, std::vector<T>& w1, std::vector<T>& w2) {
    T h1_sum = std::sqrt(std::accumulate(h1.begin(), h1.end(), (T)0));
    T h2_sum = std::sqrt(std::accumulate(h2.begin(), h2.end(), (T)0));
    T w1_sum = std::sqrt(std::accumulate(w1.begin(), w1.end(), (T)0));
    T w2_sum = std::sqrt(std::accumulate(w2.begin(), w2.end(), (T)0));
    for (unsigned int i = 0; i < h1.size(); ++i) {
        h1[i] *= std::sqrt(h1_sum * w1_sum) / h1_sum;
        h2[i] *= std::sqrt(h2_sum * w2_sum) / h2_sum;
    }
    for (unsigned int i = 0; i < w1.size(); ++i) {
        w1[i] *= std::sqrt(w1_sum * h1_sum) / w1_sum;
        w2[i] *= std::sqrt(w2_sum * h2_sum) / w2_sum;
    }
}

// cross-product of rank-2 matrix, given as a set of two vectors
template<typename T>
std::vector<std::vector<T>> cross(const std::vector<T>& h1, const std::vector<T>& h2) {
    std::vector<std::vector<T>> a(2, std::vector<T>(2));
    a[0][0] = std::inner_product(h1.begin(), h1.end(), h1.begin(), (T)0);
    a[0][1] = std::inner_product(h1.begin(), h1.end(), h2.begin(), (T)0);
    a[1][1] = std::inner_product(h2.begin(), h2.end(), h2.begin(), (T)0);
    return a;
}

// non-negative least squares to solve the equation a*x = b for rank-2 systems:
//
// [a11 a12] [x1] = [b1]    x1 = (a22b1 - a12b2) / (a11a22 - a12a12)
// [a12 a22] [x2] = [b2]    x2 = (a11b2 - a12b1) / (a11a22 - a12a12)
// denom is constant and equal to (a11a22 - a12a12)
// x1 and x2 hold "b" and are returned as "x"
template<typename T>
void ls2(T& x1, T& x2, const std::vector<std::vector<T>>& a, const T& denom, const bool nonneg) {
    if (nonneg) {
        T a01b1 = a[0][1] * x2;
        T a11b0 = a[1][1] * x1;
        if (a11b0 < a01b1) {
            x1 = 0;
            x2 /= a[1][1];
        } else {
            T a01b0 = a[0][1] * x1;
            T a00b1 = a[0][0] * x2;
            if (a00b1 < a01b0) {
                x1 /= a[0][0];
                x2 = 0;
            } else {
                x1 = (a11b0 - a01b1) / denom;
                x2 = (a00b1 - a01b0) / denom;
            }
        }
    } else {
        T b1 = x1;
        x1 = (a[1][1] * x1 - a[0][1] * x2) / denom;
        x2 = (a[0][0] * x2 - a[0][1] * b1) / denom;
    }
}

// compute cluster centroid given an ipx sparse matrix and samples in the cluster center
template <typename T>
std::vector<T> centroid(const std::vector<unsigned int>& i, const std::vector<unsigned int>& p, const std::vector<T>& x,
                        const unsigned int n_rows, const std::vector<unsigned int>& samples) {

    std::vector<T> center(n_rows);
    const unsigned int n_samples = samples.size();
    for (unsigned int s = 0; s < n_samples; ++s)
        for (unsigned int j = p[samples[s]]; j < p[samples[s] + 1]; ++j)
            center[i[j]] += x[j];
    for (unsigned int j = 0; j < n_rows; ++j) center[j] /= n_samples;
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
template <typename T>
T rel_cosine(const std::vector<unsigned int>& i, const std::vector<unsigned int>& p, const std::vector<T>& x,
             const unsigned int n_rows, const std::vector<unsigned int>& samples1, const std::vector<unsigned int>& samples2,
             const std::vector<T> center1, const std::vector<T> center2) {

    T center1_innerprod = std::sqrt(std::inner_product(center1.begin(), center1.end(), center1.begin(), (T)0));
    T center2_innerprod = std::sqrt(std::inner_product(center2.begin(), center2.end(), center2.begin(), (T)0));
    T dist1 = 0, dist2 = 0;
    for (unsigned int s = 0; s < samples1.size(); ++s) {
        T x1_center1 = 0, x1_center2 = 0;
        for (unsigned int j = p[samples1[s]]; j < p[samples1[s] + 1]; ++j) {
            x1_center1 += center1[i[j]] * x[j];
            x1_center2 += center2[i[j]] * x[j];
        }
        dist1 += (std::sqrt(x1_center2) * center1_innerprod) / (std::sqrt(x1_center1) * center2_innerprod);
    }
    for (unsigned int s = 0; s < samples2.size(); ++s) {
        T x2_center1 = 0, x2_center2 = 0;
        for (unsigned int j = p[samples2[s]]; j < p[samples2[s] + 1]; ++j) {
            x2_center1 += center1[i[j]] * x[j];
            x2_center2 += center2[i[j]] * x[j];
        }
        dist2 += (std::sqrt(x2_center1) * center2_innerprod) / (std::sqrt(x2_center2) * center1_innerprod);
    }
    return (dist1 + dist2) / (2 * n_rows);
}

template <typename T>
wh2<T> mf2(const std::vector<unsigned int>& i, const std::vector<unsigned int>& p, const std::vector<T>& x,
           const unsigned int n_rows, const std::vector<unsigned int>& samples,
           const T tol, const bool nonneg_w, const bool nonneg_h, const unsigned int maxit, const bool norm,
           const bool verbose, const bool symmetric, const unsigned int seed) {

    if (seed == 0) srand((unsigned int)time(0));
    else srand(seed);

    const unsigned int n_samples = samples.size();
    std::vector<T> w1(n_rows), w2(n_rows), h1(n_samples), h2(n_samples), h_it(n_samples), tols;

    // randomly initialize h
    std::generate(h1.begin(), h1.end(), rand);
    std::generate(h2.begin(), h2.end(), rand);
    // "block-pivoting" alternating updates of w and h
    for (unsigned int it = 0; it < maxit; ++it) {
        // calculate "a", sympd matrix giving left-hand side of linear systems, for "w" updates
        // a is designated here as [a11 a12] and is the cross-product of h * h.transpose()
        //                         [a12 a22]
        std::vector<std::vector<T>> a = cross(h1, h2);
        T denom = a[0][0] * a[1][1] - a[0][1] * a[0][1]; // preconditioned "cholesky" for least squares equations

        // calculate "b", right-hand vectors giving right-hand side of linear systems, for "w" updates
        // b is calculated here while "implicitly" transposing sparse matrix "A"
        if (it > 0) {
            std::fill(w1.begin(), w1.end(), 0);
            std::fill(w2.begin(), w2.end(), 0);
        }
        for (unsigned int s = 0; s < n_samples; ++s) {
            for (unsigned int j = p[samples[s]]; j < p[samples[s] + 1]; ++j) {
                w1[i[j]] += x[j] * h1[s];
                w2[i[j]] += x[j] * h2[s];
            }
        }
        // least squares updates for "w" given "a" and "b"
        for (unsigned int f = 0; f < n_rows; ++f)
            ls2(w1[f], w2[f], a, denom, nonneg_w);

        // "a" for left-hand side of linear systems for "h" updates
        a = cross(w1, w2);
        denom = a[0][0] * a[1][1] - a[0][1] * a[0][1];

        if (it > 0 && !symmetric) h_it = h1; // vector for tracking convergence
        for (unsigned int s = 0; s < n_samples; ++s) {
            // calculate b for each least squares update
            T b1 = 0, b2 = 0;
            for (unsigned int j = p[samples[s]]; j < p[samples[s] + 1]; ++j) {
                b1 += w1[i[j]] * x[j];
                b2 += w2[i[j]] * x[j];
            }
            // least squares updates for "h" given "a" and "b"
            ls2(b1, b2, a, denom, nonneg_h);
            h1[s] = b1;
            h2[s] = b2;
        }

        // l2-norm of "h", which gives the diagonal values
        if (norm) l2norm(h1, h2, w1, w2);

        // calculate tolerance as correlation between first "h" vectors
        if (it > 0) {
            T tol_it = symmetric ? cor(h1, w1) : cor(h1, h_it);
            if (verbose) { // BEGIN REMOVE FOR C++ USE ONLY
                Rcpp::checkUserInterrupt();
                Rprintf("%4d %8.2e\n", it + 1, tol_it);
            } // END REMOVE FOR C++ USE ONLY
            tols.push_back(tol_it);
            if (tol_it < tol) break;
        }
    }
    return wh2<T>{w1, w2, h1, h2, tols};
}

// optimized single-threaded method for returning the bipartition with optional nonnegativity constraints
template <typename T>
bipartitionModel<T> bipartition(const std::vector<unsigned int>& i, const std::vector<unsigned int>& p, const std::vector<T>& x,
                                const unsigned int n_rows, const std::vector<unsigned int>& samples, const T tol, const bool nonneg,
                                unsigned int maxit, const bool detail, const bool calc_dist) {

    const unsigned int n_samples = samples.size();
    std::vector<T> w1(n_rows), w2(n_rows), h1(n_samples), h2(n_samples), h_it(n_samples);
    std::generate(h1.begin(), h1.end(), rand);
    std::generate(h2.begin(), h2.end(), rand);
    for (unsigned int it = 0; it < maxit; ++it) {
        T a00 = std::inner_product(h1.begin(), h1.end(), h1.begin(), (T)0);
        T a01 = std::inner_product(h1.begin(), h1.end(), h2.begin(), (T)0);
        T a11 = std::inner_product(h2.begin(), h2.end(), h2.begin(), (T)0);
        T denom = a00 * a11 - a01 * a01;
        std::vector<T> wb1(n_rows), wb2(n_rows);
        for (unsigned int s = 0; s < n_samples; ++s) {
            for (unsigned int j = p[samples[s]]; j < p[samples[s] + 1]; ++j) {
                wb1[i[j]] += x[j] * h1[s];
                wb2[i[j]] += x[j] * h2[s];
            }
        }
        for (unsigned int f = 0; f < n_rows; ++f) {
            if (!nonneg) {
                w1[f] = (a11 * wb1[f] - a01 * wb2[f]) / denom;
                w2[f] = (a00 * wb2[f] - a01 * wb1[f]) / denom;
            } else {
                T a01b2 = a01 * wb2[f];
                T a11b1 = a11 * wb1[f];
                if (a11b1 < a01b2) {
                    w1[f] = 0;
                    w2[f] = wb2[f] / a11;
                } else {
                    T a01b1 = a01 * wb1[f];
                    T a00b2 = a00 * wb2[f];
                    if (a00b2 < a01b1) {
                        w1[f] = wb1[f] / a00;
                        w2[f] = 0;
                    } else {
                        w1[f] = (a11b1 - a01b2) / denom;
                        w2[f] = (a00b2 - a01b1) / denom;
                    }
                }
            }
        }

        a00 = std::inner_product(w1.begin(), w1.end(), w1.begin(), (T)0);
        a01 = std::inner_product(w1.begin(), w1.end(), w2.begin(), (T)0);
        a11 = std::inner_product(w2.begin(), w2.end(), w2.begin(), (T)0);
        denom = a00 * a11 - a01 * a01;
        if (it > 0) h_it = h1;
        for (unsigned int s = 0; s < n_samples; ++s) {
            T b1 = 0, b2 = 0;
            for (unsigned int j = p[samples[s]]; j < p[samples[s] + 1]; ++j) {
                b1 += w1[i[j]] * x[j];
                b2 += w2[i[j]] * x[j];
            }
            if (!nonneg) {
                h1[s] = (a11 * b1 - a01 * b2) / denom;
                h2[s] = (a00 * b2 - a01 * b1) / denom;
            } else {
                T a01b2 = a01 * b2;
                T a11b1 = a11 * b1;
                if (a11b1 < a01b2) {
                    h1[s] = 0;
                    h2[s] = b2 / a11;
                } else {
                    T a01b1 = a01 * b1;
                    T a00b2 = a00 * b2;
                    if (a00b2 < a01b1) {
                        h1[s] = b1 / a00;
                        h2[s] = 0;
                    } else {
                        h1[s] = (a11b1 - a01b2) / denom;
                        h2[s] = (a00b2 - a01b1) / denom;
                    }
                }
            }
        }
        if (nonneg) l2norm(h1, h2, w1, w2);
        if (it > 0 && cor(h1, h_it) < tol) break;
    }
    std::vector<bool> v(n_samples);
    for (unsigned int j = 0; j < n_samples; ++j) v[j] = h1[j] > h2[j];

    if (!detail) {
        bipartitionModel<T> mod;
        mod.v = v;
        return mod;
    }

    // calculate details
    unsigned int size1 = std::count(v.begin(), v.end(), true);
    unsigned int size2 = n_samples - size1;

    // get indices of samples in both clusters
    std::vector<unsigned int> samples1, samples2;
    samples1.reserve(size1);
    samples2.reserve(size2);
    for (unsigned int j = 0; j < n_samples; ++j)
        (v[j] > 0) ? samples1.emplace_back(samples[j]) : samples2.emplace_back(samples[j]);

    // calculate the centers of both clusters
    std::vector<T> center1 = centroid(i, p, x, n_rows, samples1);
    std::vector<T> center2 = centroid(i, p, x, n_rows, samples2);

    // calculate relative cosine similarity of all samples to ((assigned - other) / assigned) cluster
    T dist = 0;
    if (calc_dist) dist = rel_cosine(i, p, x, n_rows, samples1, samples2, center1, center2);

    return bipartitionModel<T> {v, dist, size1, size2, samples1, samples2, center1, center2};
}

// solve a rank-2 (or stop at rank-1) svd by rank-1 mf, fixing first factors, then solving for second factors
template <typename T>
uv<T> svd2(const std::vector<unsigned int>& i, const std::vector<unsigned int>& p, const std::vector<T>& x,
           const unsigned int n_rows, const std::vector<unsigned int>& samples,
           const T tol, const unsigned int maxit, const unsigned int k) {

    // calculate first singular vector
    const unsigned int n_samples = samples.size();
    std::vector<T> u1(n_rows), u2(n_rows), v1(n_samples), v2(n_samples), v_it(n_samples);
    std::generate(v1.begin(), v1.end(), rand);
    for (unsigned int it = 0; it < maxit; ++it) {
        // update u
        T a = std::inner_product(v1.begin(), v1.end(), v1.begin(), (T)0);
        if (it > 0) std::fill(u1.begin(), u1.end(), 0);
        for (unsigned int s = 0; s < n_samples; ++s)
            for (unsigned int j = p[samples[s]]; j < p[samples[s] + 1]; ++j)
                u1[i[j]] += x[j] * v1[s];
        for (unsigned int j = 0; j < n_rows; ++j) u1[j] /= a;

        // update v
        if (it > 0) v_it = v1;
        a = std::inner_product(u1.begin(), u1.end(), u1.begin(), (T)0);
        std::fill(v1.begin(), v1.end(), 0);
        for (unsigned int s = 0; s < n_samples; ++s) {
            for (unsigned int j = p[samples[s]]; j < p[samples[s] + 1]; ++j)
                v1[s] += u1[i[j]] * x[j];
            v1[s] /= a;
        }
        if (it > 0 && cor(v1, v_it) < tol) break;
    }
    if (k == 1) return uv<T> {u1, v1};

    // calculate second singular vector
    std::generate(v2.begin(), v2.end(), rand);
    for (unsigned int it = 0; it < maxit; ++it) {
        // update u2
        T a1 = std::inner_product(v2.begin(), v2.end(), v1.begin(), (T)0);
        T a2 = std::inner_product(v2.begin(), v2.end(), v2.begin(), (T)0);
        if (it > 0) std::fill(u2.begin(), u2.end(), 0);
        for (unsigned int s = 0; s < n_samples; ++s)
            for (unsigned int j = p[samples[s]]; j < p[samples[s] + 1]; ++j)
                u2[i[j]] += x[j] * v2[s];
        for (unsigned int j = 0; j < n_rows; ++j)
            u2[j] = (u2[j] - u1[j] * a1) / a2;

        // update v2
        if (it > 0) v_it = v2;
        a1 = std::inner_product(u2.begin(), u2.end(), u1.begin(), (T)0);
        a2 = std::inner_product(u2.begin(), u2.end(), u2.begin(), (T)0);
        std::fill(v2.begin(), v2.end(), 0);
        for (unsigned int s = 0; s < n_samples; ++s) {
            for (unsigned int j = p[samples[s]]; j < p[samples[s] + 1]; ++j)
                v2[s] += u2[i[j]] * x[j];
            v2[s] = (v2[s] - v1[s] * a1) / a2;
        }
        if (it > 0 && cor(v_it, v2) < tol) break;
    }
    return uv<T> {u2, v2};
}

// **************************************************************************** ADACLUST FUNCTIONS

// seeding phase: initial cluster center, bipartition...

// divisive phase: given an initial array of cluster models, divide until no more divisions are possible given min_samples and min_dist
// parallelized for loop for each split generation, iterating through all cluster objects, pushing back #2 and replacing parent with #1
// parallelization is applied across the bipartitioning, otherwise to the for loop

// when no clusters can be divided or agglomerated, adaclust is done (or when maxit is reached)

template <typename T>
std::vector<unsigned int> indices_that_are_not_leaves(std::vector<clusterModel<T>>& clusters) {
    std::vector<unsigned int> ind;
    for (unsigned int i = 0; i < clusters.size(); ++i)
        if (!clusters[i].leaf) ind.push_back(i);
    return ind;
}

// merge cluster j into cluster i in "clusters" vector of clusterModels
template <typename T>
void merge(std::vector<clusterModel<T>>& clusters, const unsigned int i, const unsigned int j, const unsigned int min_samples) {
    unsigned int i_size = clusters[i].samples.size();
    unsigned int j_size = clusters[j].samples.size();
    unsigned int new_size = i_size + j_size;
    clusters[i].id = "(" + clusters[i].id + "m" + clusters[j].id + ")";
    clusters[i].leaf = new_size > (2 * min_samples);
    unsigned int n_rows = clusters[i].center.size();
    for (unsigned int r = 0; r < n_rows; ++r) // weighted average to give new cluster center
        clusters[i].center[r] = (clusters[i].center[r] * i_size) + (clusters[j].center[r] * j_size) / new_size;
    clusters[i].samples.insert(clusters[i].samples.end(), clusters[j].samples.begin(), clusters[j].samples.end());
    clusters.erase(clusters.begin() + j);
}

// simple division to min_samples without any distance checks or agglomerations
template <typename T>
std::vector<clusterModel<T>> sketch(const std::vector<unsigned int>& Ai, const std::vector<unsigned int>& Ap, const std::vector<T>& Ax,
                                    const unsigned int n_rows, const unsigned int min_samples, const bool verbose,
                                    const unsigned int threads, const T bipartition_tol, const bool bipartition_nonneg,
                                    const unsigned int bipartition_maxit) {

    std::vector<clusterModel<T>> clusters;
    std::vector<unsigned int> samples(Ap.size() - 1);
    std::iota(samples.begin(), samples.end(), (unsigned int)0);

    // initial bipartition
    if (verbose) Rprintf("\nsplits: ");
    bipartitionModel<T> p0 = bipartition(Ai, Ap, Ax, n_rows, samples, bipartition_tol, bipartition_nonneg, bipartition_maxit, true, false);
    clusters.push_back(clusterModel<T>{ "0", p0.samples1, p0.center1, 0, p0.size1 < min_samples * 2, false });
    clusters.push_back(clusterModel<T>{ "1", p0.samples2, p0.center2, 0, p0.size2 < min_samples * 2, false });

    if (verbose) Rprintf(" 1");
    std::vector<unsigned int> to_split = indices_that_are_not_leaves(clusters), new_clusters();
    unsigned int n_splits = 1;
    while (n_splits > 0) { // attempt to bipartition all clusters that have not yet been determined to be leaves
        Rcpp::checkUserInterrupt();
        n_splits = 0;
        to_split = indices_that_are_not_leaves(clusters);
        std::vector<clusterModel<T>> new_clusters(to_split.size());
        #pragma omp parallel for num_threads(threads) schedule(dynamic)
        for (unsigned int i = 0; i < to_split.size(); ++i) {
            bipartitionModel<T> p = bipartition(Ai, Ap, Ax, n_rows, clusters[to_split[i]].samples, bipartition_tol, bipartition_nonneg, bipartition_maxit, true, false);
            if (p.size1 > min_samples && p.size2 > min_samples) { // bipartition was successful
                new_clusters[i] = clusterModel<T>{ clusters[to_split[i]].id + "1", p.samples2, p.center2, 0, p.size2 < min_samples * 2, false };
                clusters[to_split[i]] = clusterModel<T>{ clusters[to_split[i]].id + "0", p.samples1, p.center1, 0, p.size1 < min_samples * 2, false };
                ++n_splits;
            } else clusters[to_split[i]].leaf = true;
        }
        for (unsigned int i = 0; i < new_clusters.size(); ++i)
            if (new_clusters[i].id.size() > 0) clusters.push_back(new_clusters[i]);
        if (verbose) Rprintf(", %u", n_splits);
    }
    if (verbose) Rprintf("\n");
    return clusters;
}

template <typename T>
std::vector<clusterModel<T>> dclust(const std::vector<unsigned int>& Ai, const std::vector<unsigned int>& Ap, const std::vector<T>& Ax,
                                    const unsigned int n_rows, const T min_dist, const unsigned int min_samples, const bool verbose,
                                    const unsigned int threads, const T bipartition_tol, const bool bipartition_nonneg,
                                    const unsigned int bipartition_maxit) {

    std::vector<clusterModel<T>> clusters;
    std::vector<unsigned int> samples(Ap.size() - 1);
    std::iota(samples.begin(), samples.end(), (unsigned int)0);

    // initial bipartition
    if (verbose) Rprintf("\nsplits: ");
    bipartitionModel<T> p0 = bipartition(Ai, Ap, Ax, n_rows, samples, bipartition_tol, bipartition_nonneg, bipartition_maxit, true, true);
    clusters.push_back(clusterModel<T>{ "0", p0.samples1, p0.center1, 0, p0.size1 < min_samples * 2, false });
    clusters.push_back(clusterModel<T>{ "1", p0.samples2, p0.center2, 0, p0.size2 < min_samples * 2, false });

    if (verbose) Rprintf(" 1");
    std::vector<unsigned int> to_split = indices_that_are_not_leaves(clusters), new_clusters();
    unsigned int n_splits = 1;
    while (n_splits > 0) { // attempt to bipartition all clusters that have not yet been determined to be leaves
        Rcpp::checkUserInterrupt();
        n_splits = 0;
        to_split = indices_that_are_not_leaves(clusters);
        std::vector<clusterModel<T>> new_clusters(to_split.size());
        #pragma omp parallel for num_threads(threads) schedule(dynamic)
        for (unsigned int i = 0; i < to_split.size(); ++i) {
            bipartitionModel<T> p = bipartition(Ai, Ap, Ax, n_rows, clusters[to_split[i]].samples, bipartition_tol, bipartition_nonneg, bipartition_maxit, true, true);
            if (p.size1 > min_samples && p.size2 > min_samples && p.dist > min_dist) { // bipartition was successful
                new_clusters[i] = clusterModel<T>{ clusters[to_split[i]].id + "1", p.samples2, p.center2, 0, p.size2 < min_samples * 2, false };
                clusters[to_split[i]] = clusterModel<T>{ clusters[to_split[i]].id + "0", p.samples1, p.center1, 0, p.size1 < min_samples * 2, false };
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
    return clusters;
}

template <typename T>
std::vector<clusterModel<T>> adaclust(const std::vector<unsigned int>& Ai, const std::vector<unsigned int>& Ap, const std::vector<T>& Ax,
                                      const unsigned int n_rows, const T min_dist, const unsigned int min_samples, const bool verbose,
                                      const unsigned int threads, const T bipartition_tol, const bool bipartition_nonneg,
                                      const unsigned int bipartition_maxit) {

    std::vector<clusterModel<T>> clusters;
    std::vector<unsigned int> samples(Ap.size() - 1);
    std::iota(samples.begin(), samples.end(), (unsigned int)0);

    // initial bipartition
    bipartitionModel<T> p0 = bipartition(Ai, Ap, Ax, n_rows, samples, bipartition_tol, bipartition_nonneg, bipartition_maxit, true, threads);
    clusters.push_back(clusterModel<T>{ "0", p0.samples1, p0.center1, 0, p0.size1 < min_samples * 2, false });
    clusters.push_back(clusterModel<T>{ "1", p0.samples2, p0.center2, 0, p0.size2 < min_samples * 2, false });
    unsigned int n_merges;

    // alternating divisive and agglomerative subroutines
    do {
        // divisive subroutine
        if (verbose) Rprintf("\nsplits: 1");
        std::vector<unsigned int> to_split = indices_that_are_not_leaves(clusters), new_clusters();
        unsigned int n_splits = 1;
        while (n_splits > 0) { // attempt to bipartition all clusters that have not yet been determined to be leaves
            Rcpp::checkUserInterrupt();
            n_splits = 0;
            to_split = indices_that_are_not_leaves(clusters);
            std::vector<clusterModel<T>> new_clusters(to_split.size());
            #pragma omp parallel for num_threads(threads) schedule(dynamic)
            for (unsigned int i = 0; i < to_split.size(); ++i) {
                bipartitionModel<T> p = bipartition(Ai, Ap, Ax, n_rows, clusters[to_split[i]].samples, bipartition_tol, bipartition_nonneg, bipartition_maxit, true, true);
                if (p.size1 > min_samples && p.size2 > min_samples && p.dist > min_dist) { // bipartition was successful
                    new_clusters[i] = clusterModel<T>{ clusters[to_split[i]].id + "1", p.samples2, p.center2, 0, p.size2 < min_samples * 2, false };
                    clusters[to_split[i]] = clusterModel<T>{ clusters[to_split[i]].id + "0", p.samples1, p.center1, 0, p.size1 < min_samples * 2, false };
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

        // agglomerative subroutine
        // calculate distances between all pairs of new/new and new/old clusters (new clusters designated by agg = "false")
        n_merges = 0;
        std::vector<bool> to_agg;
        for (unsigned int i = 0; i < clusters.size(); ++i) if (!clusters[i].agg) to_agg.push_back(i);
        std::vector<std::vector<T>> dists(clusters.size(), std::vector<T>(clusters.size(), 1));
        for (unsigned int i = 0; i < to_agg.size(); ++i) {
            unsigned int ci = to_agg[i];
            for (unsigned int j = 0; j < clusters.size(); ++j) {
                if (dists[j][ci] == 1 && j != ci) {
                    dists[j][ci] = rel_cosine(Ai, Ap, Ax, n_rows, clusters[ci].samples, clusters[j].samples, clusters[ci].center, clusters[j].center);
                    dists[ci][j] = dists[j][ci];
                }
            }
        }
        while (min_dist > 0) {
            // find the cluster pair (ci, cj) with the least distance that is also less than min_dist
            Rcpp::checkUserInterrupt();
            unsigned int ci = 0, cj = 0;
            T min_val = 1;
            #pragma omp parallel for num_threads(threads) schedule(dynamic)
            for (unsigned int i = 0; i < (clusters.size() - 1); ++i) {
                for (unsigned int j = i + 1; j < clusters.size(); ++j) {
                    if (dists[i][j] < min_val) min_val = dists[i][j];
                    ci = i;
                    cj = j;
                }
            }
            if (min_val >= min_dist) break;
            merge(clusters, ci, cj, min_samples); // merge cj into ci and erase cj from distance matrix
            dists.erase(dists.begin() + cj);
            for (unsigned int c = 0; c < dists.size(); ++c) dists[c].erase(dists[c].begin() + cj);
            for (unsigned int c = 0; c < dists.size(); ++c) {
                dists[ci][c] = rel_cosine(Ai, Ap, Ax, n_rows, clusters[c].samples, clusters[ci].samples, clusters[c].center, clusters[ci].center);
                dists[c][ci] = dists[ci][c];
            }
            ++n_merges;
            Rprintf("  n_merges = %u\n", n_merges);
        }
        if (verbose) Rprintf(", merges: %u", n_merges);
    } while (n_merges > 0);

    return clusters;
}


// RCPP WRAPPER FUNCTIONS

//[[Rcpp::export]]
Rcpp::List Rcpp_mf2_double(const std::vector<unsigned int>& i, const std::vector<unsigned int>& p, const std::vector<double>& x,
                           const unsigned int n_rows, const std::vector<unsigned int>& samples,
                           const double tol, const bool nonneg_w, const bool nonneg_h, const unsigned int maxit, const bool norm,
                           const bool verbose, const bool symmetric, const unsigned int seed) {

    wh2<double> m = mf2(i, p, x, n_rows, samples, tol, nonneg_w, nonneg_h, maxit, norm, verbose, symmetric, seed);

    return(Rcpp::List::create(
        Rcpp::Named("w1") = m.w1,
        Rcpp::Named("w2") = m.w2,
        Rcpp::Named("h1") = m.h1,
        Rcpp::Named("h2") = m.h2,
        Rcpp::Named("tol") = m.tol));
}

//[[Rcpp::export]]
Rcpp::List Rcpp_mf2_float(const std::vector<unsigned int>& i, const std::vector<unsigned int>& p, const std::vector<float>& x,
                          const unsigned int n_rows, const std::vector<unsigned int>& samples,
                          const float tol, const bool nonneg_w, const bool nonneg_h, const unsigned int maxit, const bool norm,
                          const bool verbose, const bool symmetric, const unsigned int seed) {

    wh2<float> m = mf2(i, p, x, n_rows, samples, tol, nonneg_w, nonneg_h, maxit, norm, verbose, symmetric, seed);

    return(Rcpp::List::create(
        Rcpp::Named("w1") = m.w1,
        Rcpp::Named("w2") = m.w2,
        Rcpp::Named("h1") = m.h1,
        Rcpp::Named("h2") = m.h2,
        Rcpp::Named("tol") = m.tol));
}

//[[Rcpp::export]]
Rcpp::List Rcpp_svd2_double(const std::vector<unsigned int>& i, const std::vector<unsigned int>& p, const std::vector<double>& x,
                            const unsigned int n_rows, const std::vector<unsigned int>& samples, const double tol,
                            const unsigned int maxit, const unsigned int k) {

    uv<double> uv = svd2(i, p, x, n_rows, samples, tol, maxit, k);
    return(Rcpp::List::create(Rcpp::Named("u") = uv.u, Rcpp::Named("v") = uv.v));
}

//[[Rcpp::export]]
Rcpp::List Rcpp_svd2_float(const std::vector<unsigned int>& i, const std::vector<unsigned int>& p, const std::vector<float>& x,
                           const unsigned int n_rows, const std::vector<unsigned int>& samples, const float tol,
                           const unsigned int maxit, const unsigned int k) {

    uv<float> uv = svd2(i, p, x, n_rows, samples, tol, maxit, k);

    return(Rcpp::List::create(Rcpp::Named("u") = uv.u, Rcpp::Named("v") = uv.v));
}

//[[Rcpp::export]]
Rcpp::List Rcpp_bipartition_double(const std::vector<unsigned int>& i, const std::vector<unsigned int>& p, const std::vector<double>& x,
                                   const unsigned int n_rows, const std::vector<unsigned int>& samples, const double tol, const bool nonneg,
                                   unsigned int maxit, const bool detail) {

    bipartitionModel<double> m;
    m = bipartition(i, p, x, n_rows, samples, tol, nonneg, maxit, detail, true);

    return(Rcpp::List::create(
        Rcpp::Named("v") = m.v,
        Rcpp::Named("dist") = m.dist,
        Rcpp::Named("size1") = m.size1,
        Rcpp::Named("size2") = m.size2,
        Rcpp::Named("samples1") = m.samples1,
        Rcpp::Named("samples2") = m.samples2,
        Rcpp::Named("center1") = m.center1,
        Rcpp::Named("center2") = m.center2));
}

//[[Rcpp::export]]
Rcpp::List Rcpp_bipartition_float(const std::vector<unsigned int>& i, const std::vector<unsigned int>& p, const std::vector<float>& x,
                                  const unsigned int n_rows, const std::vector<unsigned int>& samples, const float tol, const bool nonneg,
                                  unsigned int maxit, const bool detail) {

    bipartitionModel<float> m = bipartition(i, p, x, n_rows, samples, tol, nonneg, maxit, detail, true);

    return(Rcpp::List::create(
        Rcpp::Named("v") = m.v,
        Rcpp::Named("dist") = m.dist,
        Rcpp::Named("size1") = m.size1,
        Rcpp::Named("size2") = m.size2,
        Rcpp::Named("samples1") = m.samples1,
        Rcpp::Named("samples2") = m.samples2,
        Rcpp::Named("center1") = m.center1,
        Rcpp::Named("center2") = m.center2));
}

//[[Rcpp::export]]
Rcpp::List Rcpp_adaclust_double(const std::vector<unsigned int>& i, const std::vector<unsigned int>& p, const std::vector<double>& x,
                                const unsigned int n_rows, const double min_dist, const unsigned int min_samples, const bool verbose,
                                const unsigned int threads, const double bipartition_tol, const bool bipartition_nonneg,
                                const unsigned int bipartition_maxit) {

    std::vector<clusterModel<double>> clusters = adaclust(i, p, x, n_rows, min_dist, min_samples, verbose, threads, bipartition_tol, bipartition_nonneg, bipartition_maxit);

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
Rcpp::List Rcpp_adaclust_float(const std::vector<unsigned int>& i, const std::vector<unsigned int>& p, const std::vector<float>& x,
                               const unsigned int n_rows, const float min_dist, const unsigned int min_samples, const bool verbose,
                               const unsigned int threads, const float bipartition_tol, const bool bipartition_nonneg,
                               const unsigned int bipartition_maxit) {

    std::vector<clusterModel<float>> clusters = adaclust(i, p, x, n_rows, min_dist, min_samples, verbose, threads, bipartition_tol, bipartition_nonneg, bipartition_maxit);

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
