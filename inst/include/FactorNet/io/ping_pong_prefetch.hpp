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

// Temporary debug: write to a file for visibility
#include <fstream>
static std::ofstream& pp_debug_log() {
    static std::ofstream f("/tmp/pp_debug.log", std::ios::app);
    return f;
}
#define PP_DBG(...) do { char buf[256]; std::snprintf(buf, sizeof(buf), __VA_ARGS__); pp_debug_log() << buf; pp_debug_log().flush(); } while(0)

namespace FactorNet {
namespace io {

template<typename ChunkType>
class PingPongPrefetcher {
public:
    using LoadFn = std::function<bool(ChunkType&)>;

    /**
     * @brief Construct and start the background prefetch thread.
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
        // Pre-load first chunk synchronously so next() can return immediately
        PP_DBG("[PP] Constructor: loading slot 0...\n");
        
        read_has_data_[0] = load_fn_(slots_[0]);
        read_ready_[0] = true;
        PP_DBG("[PP] Constructor: slot 0 done, has_data=%d\n", (int)read_has_data_[0]);
        

        if (read_has_data_[0]) {
            // Start background thread to prefetch slot 1
            write_idx_ = 1;
            PP_DBG("[PP] Constructor: starting worker thread (write_idx_=1)\n");
            
            worker_ = std::thread(&PingPongPrefetcher::worker_loop, this);
        } else {
            finished_ = true;
        }
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
        PP_DBG("[PP] next(): write_idx_=%d, idx=%d, ready[0]=%d, ready[1]=%d, data[0]=%d, data[1]=%d\n",
            write_idx_, idx, (int)read_ready_[0], (int)read_ready_[1],
            (int)read_has_data_[0], (int)read_has_data_[1]);
        

        // Wait for read slot to be ready
        {
            std::unique_lock<std::mutex> lk(mu_);
            PP_DBG("[PP] next(): acquired lock, checking read_ready_[%d]=%d\n", idx, (int)read_ready_[idx]);
            
            cv_ready_.wait(lk, [&]{ return read_ready_[idx]; });
            PP_DBG("[PP] next(): wait completed, read_ready_[%d]=%d\n", idx, (int)read_ready_[idx]);
            
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
        PP_DBG("[PP] worker_loop() started\n");
        
        while (true) {
            int target;
            {
                std::unique_lock<std::mutex> lk(mu_);
                PP_DBG("[PP] worker: checking condition, done_=%d, write_idx_=%d, ready[wi]=%d\n",
                    (int)done_, write_idx_, (int)read_ready_[write_idx_]);
                
                cv_work_.wait(lk, [&]{
                    return done_ || !read_ready_[write_idx_];
                });
                PP_DBG("[PP] worker: condition met, done_=%d\n", (int)done_);
                
                if (done_) return;
                target = write_idx_;
            }

            PP_DBG("[PP] worker: loading target=%d\n", target);
            
            bool has_data = load_fn_(slots_[target]);
            PP_DBG("[PP] worker: load done, has_data=%d\n", (int)has_data);
            

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
