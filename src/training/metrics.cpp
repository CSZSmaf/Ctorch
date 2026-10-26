#include "dl/training.h"

#include <algorithm>
#include <numeric>

namespace dl::training {

ConfusionMatrix::ConfusionMatrix(Index classes) : classes_(classes), counts_(static_cast<std::size_t>(classes * classes), 0) {}

void ConfusionMatrix::update(const Tensor& prediction, const Tensor& target, std::optional<Index> ignore_index) {
    auto predicted = prediction.to_vector<Index>();
    auto expected = target.to_vector<Index>();
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (ignore_index && expected[i] == *ignore_index) continue;
        ++counts_[expected[i] * classes_ + predicted[i]];
    }
}

void ConfusionMatrix::reset() { std::fill(counts_.begin(), counts_.end(), Index{0}); }

ClassificationScores ConfusionMatrix::scores() const {
    ClassificationScores result;
    auto ratio = [](Index numerator, Index denominator) { return denominator == 0 ? 0.0 : double(numerator) / double(denominator); };
    Index correct = 0, total = std::accumulate(counts_.begin(), counts_.end(), Index{0});
    for (Index c = 0; c < classes_; ++c) {
        const auto diagonal = counts_[c * classes_ + c];
        Index actual = 0, predicted = 0;
        for (Index k = 0; k < classes_; ++k) { actual += counts_[c * classes_ + k]; predicted += counts_[k * classes_ + c]; }
        correct += diagonal;
        result.precision.push_back(ratio(diagonal, predicted));
        result.recall.push_back(ratio(diagonal, actual));
        result.f1.push_back(ratio(2 * diagonal, actual + predicted));
        result.iou.push_back(ratio(diagonal, actual + predicted - diagonal));
    }
    result.accuracy = ratio(correct, total);
    return result;
}

Tensor ConfusionMatrix::state_dict() const { return from_span<Index>(counts_, {classes_, classes_}); }
void ConfusionMatrix::load_state_dict(const Tensor& counts) { classes_ = counts.size(0); counts_ = counts.to_vector<Index>(); }

} // namespace dl::training
