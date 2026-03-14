/**
 * @file io/file_reader.hpp
 * @brief Platform-agnostic random-access file reader using positional reads.
 *
 * Uses pread() on POSIX (Linux/macOS/NFS) and ReadFile() with OVERLAPPED
 * on Windows. Thread-safe for concurrent reads on POSIX platforms.
 *
 * @author Zach DeBruine
 * @date 2026
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string>

#ifdef _WIN32
// Avoid <windows.h> entirely — MinGW C++17+ std::byte conflicts with
// rpcndr.h byte typedef.  Use POSIX-compat _open/_read/_lseeki64 instead.
#  include <io.h>
#  include <fcntl.h>
#  include <sys/stat.h>
#  include <mutex>
#else
#  include <fcntl.h>
#  include <unistd.h>
#  include <sys/stat.h>
#  include <errno.h>
#endif

namespace streampress {

/// Platform-agnostic random-access file reader.
/// Supports positional reads without moving any file cursor.
/// Thread-safe for concurrent pread() calls on POSIX platforms.
/// On Windows, uses a mutex to serialise _lseeki64 + _read.
class FileReader {
public:
    explicit FileReader(const std::string& path) {
#ifdef _WIN32
        fd_ = _open(path.c_str(), _O_RDONLY | _O_BINARY);
        if (fd_ < 0)
            throw std::runtime_error("Cannot open file: " + path);
        __int64 sz = _lseeki64(fd_, 0, SEEK_END);
        if (sz < 0) {
            _close(fd_);
            fd_ = -1;
            throw std::runtime_error("Cannot get file size: " + path);
        }
        file_size_ = static_cast<uint64_t>(sz);
        _lseeki64(fd_, 0, SEEK_SET);
#else
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0)
            throw std::runtime_error("Cannot open file: " + path);

        struct stat st;
        if (fstat(fd_, &st) != 0) {
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("Cannot stat file: " + path);
        }
        file_size_ = static_cast<uint64_t>(st.st_size);
#endif
    }

    ~FileReader() { close(); }

    // Non-copyable
    FileReader(const FileReader&) = delete;
    FileReader& operator=(const FileReader&) = delete;

    // Movable
    FileReader(FileReader&& other) noexcept
        : file_size_(other.file_size_)
        , fd_(other.fd_)
    {
        other.file_size_ = 0;
        other.fd_ = -1;
    }

    FileReader& operator=(FileReader&& other) noexcept {
        if (this != &other) {
            close();
            file_size_ = other.file_size_;
            fd_ = other.fd_;
            other.fd_ = -1;
            other.file_size_ = 0;
        }
        return *this;
    }

    /// Read `size` bytes from `offset` into `buf`.
    /// Returns number of bytes actually read.
    size_t pread(uint64_t offset, void* buf, size_t size) const {
        if (size == 0) return 0;
#ifdef _WIN32
        std::lock_guard<std::mutex> lock(mtx_);
        if (_lseeki64(fd_, static_cast<__int64>(offset), SEEK_SET) < 0)
            throw std::runtime_error("_lseeki64 failed at offset " + std::to_string(offset));
        int n = _read(fd_, buf, static_cast<unsigned int>(size));
        if (n < 0)
            throw std::runtime_error("_read failed at offset " + std::to_string(offset));
        return static_cast<size_t>(n);
#else
        ssize_t n = ::pread(fd_, buf, size, static_cast<off_t>(offset));
        if (n < 0)
            throw std::runtime_error("pread failed at offset " + std::to_string(offset)
                                     + ": errno=" + std::to_string(errno));
        return static_cast<size_t>(n);
#endif
    }

    uint64_t file_size() const { return file_size_; }

    bool is_open() const { return fd_ >= 0; }

    void close() {
#ifdef _WIN32
        if (fd_ >= 0) {
            _close(fd_);
            fd_ = -1;
        }
#else
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
#endif
    }

private:
    uint64_t file_size_ = 0;
    int fd_ = -1;
#ifdef _WIN32
    mutable std::mutex mtx_;
#endif
};

}  // namespace streampress
