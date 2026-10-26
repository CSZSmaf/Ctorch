#include "io/output_file.h"

#include <atomic>
#include <chrono>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace dl::io::archive {

OutputFile::OutputFile(const std::filesystem::path& path) : path_(path), temporary_(path) {
    static std::atomic<unsigned long long> sequence{0};
    const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
    temporary_ += ".tmp." + std::to_string(tick) + "." + std::to_string(sequence++);
    stream_.exceptions(std::ios::failbit | std::ios::badbit);
    stream_.open(temporary_, std::ios::binary | std::ios::trunc);
}

OutputFile::~OutputFile() {
    stream_.exceptions(std::ios::goodbit);
    if (stream_.is_open()) stream_.close();
    if (!temporary_.empty()) {
        std::error_code error;
        std::filesystem::remove(temporary_, error);
    }
}

void OutputFile::commit() {
    stream_.close();
#ifdef _WIN32
    if (!MoveFileExW(temporary_.c_str(), path_.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw std::filesystem::filesystem_error("replace checkpoint", temporary_, path_,
            std::error_code(GetLastError(), std::system_category()));
#else
    std::filesystem::rename(temporary_, path_);
#endif
    temporary_.clear();
}

} // namespace dl::io::archive
