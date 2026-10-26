#pragma once

#include <filesystem>
#include <fstream>

namespace dl::io::archive {

// 同目录临时文件完成写入后替换目标；未提交的文件由对象释放。
class OutputFile {
public:
    explicit OutputFile(const std::filesystem::path& path);
    ~OutputFile();
    std::ostream& stream() { return stream_; }
    void commit();
private:
    std::filesystem::path path_, temporary_;
    std::ofstream stream_;
};

} // namespace dl::io::archive
