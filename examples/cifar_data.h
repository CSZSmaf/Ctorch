#pragma once

#include <dl/data.h>

#include <array>
#include <filesystem>
#include <fstream>

namespace examples {

// CIFAR-10 二进制记录依次保存标签与三个 32×32 通道，布局直接对应 NCHW。
inline dl::data::TensorDataset load_cifar10(const std::filesystem::path& directory, bool training) {
    constexpr dl::Index pixels = 3 * 32 * 32, per_file = 10000;
    const dl::Index files = training ? 5 : 1, count = files * per_file;
    std::vector<float> images(static_cast<std::size_t>(count * pixels));
    std::vector<dl::Index> labels(static_cast<std::size_t>(count));
    std::array<unsigned char, pixels + 1> record;
    for (dl::Index file = 0; file < files; ++file) {
        const auto name = training ? "data_batch_" + std::to_string(file + 1) + ".bin" : "test_batch.bin";
        std::ifstream source;
        source.exceptions(std::ios::failbit | std::ios::badbit);
        source.open(directory / name, std::ios::binary);
        for (dl::Index sample = 0; sample < per_file; ++sample) {
            source.read(reinterpret_cast<char*>(record.data()), record.size());
            const auto index = file * per_file + sample;
            labels[index] = record[0];
            for (dl::Index pixel = 0; pixel < pixels; ++pixel)
                images[index * pixels + pixel] = static_cast<float>(record[pixel + 1]) / 127.5f - 1.0f;
        }
    }
    return {dl::from_span<float>(images, {count, 3, 32, 32}), dl::from_span<dl::Index>(labels, {count})};
}

} // namespace examples
