#include "dl/data.h"
#include "dl/ops.h"

#include <algorithm>
#include <numeric>

namespace dl::data {

PaddedSequence pad_sequence(std::span<const Tensor> sequences, double value) {
    std::vector<Index> lengths;
    for (const auto& tensor : sequences) lengths.push_back(tensor.size(0));
    const auto width = *std::max_element(lengths.begin(), lengths.end());
    std::vector<Tensor> padded;
    padded.reserve(sequences.size());
    for (const auto& tensor : sequences) {
        Shape padding(static_cast<std::size_t>(tensor.ndim() * 2), 0);
        padding.back() = width - tensor.size(0);
        padded.push_back(pad(tensor, std::move(padding), value));
    }
    auto sizes = from_span<Index>(lengths, {static_cast<Index>(lengths.size())}, false, sequences[0].device());
    auto positions = arange(static_cast<double>(width), {.dtype = DType::i64, .device = sequences[0].device()});
    return {stack(padded, 0), sizes, lt(positions, sizes.unsqueeze(1))};
}

std::vector<Index> weighted_order(std::span<const double> weights, Index count, Generator& generator) {
    std::discrete_distribution<Index> sample(weights.begin(), weights.end());
    std::vector<Index> result(static_cast<std::size_t>(count));
    for (auto& index : result) index = sample(generator.engine());
    return result;
}

std::vector<Index> bucket_order(std::span<const Index> lengths, Index batch_size,
                              Generator& generator, Index bucket_batches) {
    std::vector<Index> order(lengths.size());
    std::iota(order.begin(), order.end(), Index{0});
    generator.shuffle(order);
    const auto window = static_cast<std::size_t>(batch_size * bucket_batches);
    for (std::size_t start = 0; start < order.size(); start += window) {
        auto end = order.begin() + static_cast<std::ptrdiff_t>(std::min(start + window, order.size()));
        std::stable_sort(order.begin() + static_cast<std::ptrdiff_t>(start), end,
            [&](Index a, Index b) { return lengths[a] < lengths[b]; });
    }
    const auto batch = static_cast<std::size_t>(batch_size);
    std::vector<Index> groups(order.size() / batch);
    std::iota(groups.begin(), groups.end(), Index{0});
    generator.shuffle(groups);
    std::vector<Index> result;
    result.reserve(order.size());
    for (auto group : groups) {
        auto begin = order.begin() + group * batch_size;
        result.insert(result.end(), begin, begin + batch_size);
    }
    result.insert(result.end(), order.begin() + static_cast<std::ptrdiff_t>(groups.size() * batch), order.end());
    return result;
}

} // namespace dl::data
