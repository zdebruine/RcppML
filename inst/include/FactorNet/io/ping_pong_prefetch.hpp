/**
 * @file ping_pong_prefetch.hpp
 * @brief Persistent-thread double-buffered prefetcher for streaming I/O.
 *
 * Replaces per-chunk std::async with a single background thread that
 * pre-loads the next chunk into a ping-pong buffer while the main thread
 * processes the current chunk. Eliminates thread creation overhead
 * (~10-50 µs per spawn) which matters for fast NVMe or many small chunks.
 *
 * Usage:
 *   PingPongPrefetcher<Chunk<float>> pp([&](Chunk<float>& c) {
 *       return loader.next_forward(c);
 *   });
 *   Chunk<float>* chunk;
 *   while (pp.next(chunk)) {
 *       // process *chunk
 *   }
 *   // destructor joins background thread
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <array>
#include <atomic>
#include <cstdio>

namespace FactorNet {
namespace io {

template<typename ChunkType>
class PingPongPrefetcher {
public:
    using LoadFn = std::function<bool(ChunkType&)>;

    /**
     * @brief Construct and start the background prefetch thread.
     *
     * Both slots are loaded synchronously in the constructor to avoid a
     * race between next() and the worker on write_idx_. The worker thread
     * is only started when there are 3+ chunks of data to process.
     *
     * @param load_fn  Function that loads the next chunk. Returns false when no more data.
     */
    explicit PingPongPrefetcher(LoadFn load_fn)
        : load_fn_(std::move(load_fn))
        , done_(false)
        , finished_(false)
        , write_idx_(0)
        , read_ready_{false, false}
        , read_has_data_{false, false}
    {
        // Pre-load slot 0 synchronously
        read_has_data_[0] = load_fn_(slots_[0]);
        read_ready_[0] = true;

        if (!read_has_data_[0]) {
            // 0 chunks: mark slot 1 as ready-but-empty so next() returns false
            finished_ = true;
            read_ready_[1] = true;
            read_has_data_[1] = false;
            return;
        }

        // Pre-load slot 1 synchronously (eliminates the constructor/next race)
        write_idx_ = 1;
        read_has_data_[1] = load_fn_(slots_[1]);
        read_ready_[1] = true;

        if (!read_has_data_[1]) {
            // Only 1 chunk: next() will return slot 0 then discover slot 1 empty
            finished_ = true;
            return;
        }

        // 2+ chunks loaded. Start worker to refill consumed slots.
        worker_ = std::thread(&PingPongPrefetcher::worker_loop, this);
    }

    ~PingPongPrefetcher() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            done_ = true;
        }
        cv_work_.notify_one();
        if (worker_.joinable())
            worker_.join();
    }

    // Non-copyable, non-movable
    PingPongPrefetcher(const PingPongPrefetcher&) = delete;
    PingPongPrefetcher& operator=(const PingPongPrefetcher&) = delete;

    /**
     * @brief Get the next chunk. Blocks until prefetch completes.
     * @param[out] chunk  Pointer set to the internal buffer holding the chunk.
     * @return true if a chunk was available, false if stream is exhausted.
     */
    bool next(ChunkType*& chunk) {
        int idx = 1 - write_idx_; // read from opposite slot

        // Wait for read slot to be ready
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_ready_.wait(lk, [&]{ return read_ready_[idx]; });
        }

        if (!read_has_data_[idx]) {
            return false;
        }

        chunk = &slots_[idx];

        // Signal that we've consumed this slot, background can write to it
        {
            std::lock_guard<std::mutex> lk(mu_);
            read_ready_[idx] = false;
            write_idx_ = idx; // background thread targets consumed slot
        }
        cv_work_.notify_one();

        return true;
    }

private:
    void worker_loop() {
        while (true) {
            int target;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_work_.wait(lk, [&]{
                    return done_ || !read_ready_[write_idx_];
                });
                if (done_) return;
                target = write_idx_;
            }

            bool has_data = load_fn_(slots_[target]);

            {
                std::lock_guard<std::mutex> lk(mu_);
                read_has_data_[target] = has_data;
                read_ready_[target] = true;
                if (!has_data) {
                    finished_ = true;
                }
            }
            cv_ready_.notify_one();

            if (!has_data) return; // no more data, exit thread
        }
    }

    LoadFn load_fn_;
    std::array<ChunkType, 2> slots_;
    std::thread worker_;
    std::mutex mu_;
    std::condition_variable cv_ready_;  // signaled when a slot is ready to read
    std::condition_variable cv_work_;   // signaled when a slot is free to write
    bool done_;
    bool finished_;
    int write_idx_;
    bool read_ready_[2];
    bool read_has_data_[2];
};

}  // namespace io
}  // namespace FactorNet
