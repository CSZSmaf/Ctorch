#include "dl/data.h"

#include "core/layout.h"
#include "dl/ops.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <numeric>

namespace dl::data {
namespace {

std::uint32_t read_big_endian_u32(std::istream& input) {
    std::array<unsigned char, 4> bytes{};
    input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    std::uint32_t value = 0;
    for (auto byte : bytes) {
        value = (value << 8) | byte;
    }
    return value;
}

struct IdxBytes {
    Shape shape;
    std::vector<unsigned char> values;
};

IdxBytes read_idx_bytes(const std::filesystem::path& path) {
    std::ifstream input;
    input.exceptions(std::ios::failbit | std::ios::badbit);
    input.open(path, std::ios::binary);
    const auto header = read_big_endian_u32(input);
    Shape shape(header & 0xff);
    for (auto& size : shape) {
        size = read_big_endian_u32(input);
    }
    std::vector<unsigned char> values(static_cast<std::size_t>(detail::numel(shape)));
    input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size()));
    return {std::move(shape), std::move(values)};
}

} // namespace

DataLoader::DataLoader(TensorDataset dataset, LoaderOptions options, Generator& generator)
    : DataLoader(dataset.size(), [dataset](std::span<const Index> selected) {
        auto indices = from_span<Index>(selected, {static_cast<Index>(selected.size())});
        auto input_indices = indices.to(dataset.input.device());
        // 两份数据位于同一设备时共用索引，不同设备分别取得自己的索引。
        auto target_indices = dataset.target.device() == dataset.input.device()
            ? input_indices : indices.to(dataset.target.device());
        return Batch{index_select(dataset.input, 0, input_indices),
                     index_select(dataset.target, 0, target_indices)};
      }, options, generator) {}

DataLoader::DataLoader(Index size, Collate collate, LoaderOptions options, Generator& generator)
    : size_(size), collate_(std::move(collate)), options_(options), generator_(generator),
      order_(static_cast<std::size_t>(size)) {
    reset();
}

void DataLoader::reset() {
    cursor_ = 0;
    std::iota(order_.begin(), order_.end(), Index{0});
    if (options_.shuffle) {
        generator_.shuffle(order_);
    }
}

LoaderState DataLoader::state_dict() const { return {options_, order_, cursor_, generator_.state()}; }

void DataLoader::load_state_dict(const LoaderState& state) {
    options_ = state.options;
    order_ = state.order;
    cursor_ = state.cursor;
    generator_.load_state(state.random_state);
}

std::optional<Batch> DataLoader::next() {
    const auto remaining = size_ - cursor_;
    if (remaining == 0 || (options_.drop_last && remaining < options_.batch_size)) {
        return std::nullopt;
    }
    const auto count = std::min(remaining, options_.batch_size);
    auto selected = std::span<const Index>(order_).subspan(static_cast<std::size_t>(cursor_),
                                                          static_cast<std::size_t>(count));
    auto batch = collate_(selected);
    cursor_ += count;
    return batch;
}

TensorDataset load_mnist(const std::filesystem::path& images, const std::filesystem::path& labels) {
    auto image_bytes = read_idx_bytes(images);
    auto label_bytes = read_idx_bytes(labels);
    std::vector<float> pixels(image_bytes.values.size());
    std::transform(image_bytes.values.begin(), image_bytes.values.end(), pixels.begin(),
                    [](unsigned char pixel) { return static_cast<float>(pixel) / 255.0f; });
    std::vector<Index> targets(label_bytes.values.begin(), label_bytes.values.end());
    auto shape = detail::insert_axis(image_bytes.shape, 1, 1);
    return {from_span<float>(pixels, std::move(shape)), from_span<Index>(targets, std::move(label_bytes.shape))};
}

} // namespace dl::data
