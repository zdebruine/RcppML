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
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
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
class FileReader {
public:
    explicit FileReader(const std::string& path) {
#ifdef _WIN32
        // Convert UTF-8 path to wide string
        int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
        if (wlen <= 0)
            throw std::runtime_error("Cannot convert path to wide string: " + path);
        std::wstring wpath(wlen, L'\0');
        MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, &wpath[0], wlen);

        hFile_ = CreateFileW(
            wpath.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            nullptr,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS,
            nullptr);
        if (hFile_ == INVALID_HANDLE_VALUE)
            throw std::runtime_error("Cannot open file: " + path);

        LARGE_INTEGER li;
        if (!GetFileSizeEx(hFile_, &li)) {
            CloseHandle(hFile_);
            hFile_ = INVALID_HANDLE_VALUE;
            throw std::runtime_error("Cannot get file size: " + path);
        }
        file_size_ = static_cast<uint64_t>(li.QuadPart);
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
#ifdef _WIN32
        , hFile_(other.hFile_)
#else
        , fd_(other.fd_)
#endif
    {
        other.file_size_ = 0;
#ifdef _WIN32
        other.hFile_ = INVALID_HANDLE_VALUE;
#else
        other.fd_ = -1;
#endif
    }

    FileReader& operator=(FileReader&& other) noexcept {
        if (this != &other) {
            close();
            file_size_ = other.file_size_;
#ifdef _WIN32
            hFile_ = other.hFile_;
            other.hFile_ = INVALID_HANDLE_VALUE;
#else
            fd_ = other.fd_;
            other.fd_ = -1;
#endif
            other.file_size_ = 0;
        }
        return *this;
    }

    /// Read `size` bytes from `offset` into `buf`.
    /// Returns number of bytes actually read.
    size_t pread(uint64_t offset, void* buf, size_t size) const {
        if (size == 0) return 0;
#ifdef _WIN32
        OVERLAPPED ov = {};
        ov.Offset     = static_cast<DWORD>(offset & 0xFFFFFFFF);
        ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
        DWORD bytes_read = 0;
        if (!ReadFile(hFile_, buf, static_cast<DWORD>(size), &bytes_read, &ov))
            throw std::runtime_error("ReadFile failed at offset " + std::to_string(offset));
        return static_cast<size_t>(bytes_read);
#else
        ssize_t n = ::pread(fd_, buf, size, static_cast<off_t>(offset));
        if (n < 0)
            throw std::runtime_error("pread failed at offset " + std::to_string(offset)
                                     + ": errno=" + std::to_string(errno));
        return static_cast<size_t>(n);
#endif
    }

    uint64_t file_size() const { return file_size_; }

    bool is_open() const {
#ifdef _WIN32
        return hFile_ != INVALID_HANDLE_VALUE;
#else
        return fd_ >= 0;
#endif
    }

    void close() {
#ifdef _WIN32
        if (hFile_ != INVALID_HANDLE_VALUE) {
            CloseHandle(hFile_);
            hFile_ = INVALID_HANDLE_VALUE;
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
#ifdef _WIN32
    HANDLE hFile_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
#endif
};

}  // namespace streampress
