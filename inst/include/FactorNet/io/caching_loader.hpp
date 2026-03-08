/**
 * @file io/caching_loader.hpp
 * @brief CachingLoader — wraps any DataLoader to cache forward chunks.
 *
 * On the first forward pass, decompressed chunks are stored in a vector.
 * Subsequent reset_forward() + next_forward() calls return cached data
 * instead of re-decompressing.
 *
 * This is especially beneficial for:
 *   - GPU chunked NMF which re-streams forward chunks for W-update
 *   - Streaming NMF which re-streams forward chunks for loss computation
 *
 * Memory: O(nnz) — holds all chunks decompressed simultaneously.
 * Use only when total matrix fits in RAM (streaming benefit is the
 * avoidance of re-decompression, not memory reduction).
 *
 * @author RcppML Agent 7
 * @date 2026
 */

#pragma once

#include <FactorNet/io/loader.hpp>
#include <vector>
#include <memory>

namespace FactorNet {
namespace io {

/**
 * @brief Caching wrapper for DataLoader.
 *
 * First forward pass delegates to the wrapped loader and caches chunks.
 * Subsequent forward passes replay from cache.
 * Transpose pass always delegates (no caching needed — only traversed once).
 *
 * @tparam Scalar float or double
 */
template<typename Scalar>
class CachingLoader : public DataLoader<Scalar> {
public:
    using SpMat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, int>;

    /**
     * @brief Wrap an existing DataLoader.
     *
     * @param inner  The underlying loader (takes ownership)
     */
    explicit CachingLoader(std::unique_ptr<DataLoader<Scalar>> inner)
        : inner_(std::move(inner)),
          fwd_cached_(false),
          trans_cached_(false),
          fwd_replay_idx_(0),
          trans_replay_idx_(0)
    {}

    uint32_t rows() const override { return inner_->rows(); }
    uint32_t cols() const override { return inner_->cols(); }
    uint64_t nnz()  const override { return inner_->nnz(); }

    uint32_t num_forward_chunks() const override {
        return inner_->num_forward_chunks();
    }

    uint32_t num_transpose_chunks() const override {
        return inner_->num_transpose_chunks();
    }

    bool next_forward(Chunk<Scalar>& out) override {
        if (fwd_cached_) {
            // Replay from cache
            if (fwd_replay_idx_ >= fwd_cache_.size()) return false;
            out = fwd_cache_[fwd_replay_idx_];
            // Share the sparse matrix (Eigen uses COW/shared pointers internally)
            ++fwd_replay_idx_;
            return true;
        }
        // First pass: delegate and cache
        if (!inner_->next_forward(out)) {
            fwd_cached_ = true;
            return false;
        }
        fwd_cache_.push_back(out);
        return true;
    }

    bool next_transpose(Chunk<Scalar>& out) override {
        if (trans_cached_) {
            if (trans_replay_idx_ >= trans_cache_.size()) return false;
            out = trans_cache_[trans_replay_idx_++];
            return true;
        }
        if (!inner_->next_transpose(out)) {
            trans_cached_ = true;
            return false;
        }
        trans_cache_.push_back(out);
        return true;
    }

    void reset_forward() override {
        if (fwd_cached_) {
            fwd_replay_idx_ = 0;
        } else {
            inner_->reset_forward();
        }
    }

    void reset_transpose() override {
        if (trans_cached_) {
            trans_replay_idx_ = 0;
        } else {
            inner_->reset_transpose();
        }
    }

    /// Access to the underlying loader
    DataLoader<Scalar>* inner() { return inner_.get(); }

    /// Check if forward chunks are cached
    bool is_cached() const { return fwd_cached_; }

    /// Total cached memory (approximate)
    size_t cached_nnz() const {
        size_t total = 0;
        for (const auto& c : fwd_cache_)
            total += c.nnz;
        return total;
    }

private:
    std::unique_ptr<DataLoader<Scalar>> inner_;
    std::vector<Chunk<Scalar>> fwd_cache_;
    std::vector<Chunk<Scalar>> trans_cache_;
    bool fwd_cached_, trans_cached_;
    size_t fwd_replay_idx_, trans_replay_idx_;
};

}  // namespace io
}  // namespace FactorNet
