#include "dl/training.h"

#include "core/tensor_impl.h"
#include "ops/numeric.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace dl::training {
namespace {

Tensor scalar(double value) { return full({}, value, {.dtype = DType::f64}); }
Tensor integer(Index value) { return from_span<Index>(std::span(&value, 1), {}); }

std::unordered_map<std::string, Tensor> named(std::span<const NamedTensor> state) {
    return {state.begin(), state.end()};
}

bool observe(double metric, const MonitorOptions& options, MetricHistory& history) {
    const auto improvement = options.mode == MetricMode::minimize ? history.best - metric : metric - history.best;
    if (!history.initialized || improvement > options.min_delta) {
        history = {metric, 0, true};
        return true;
    }
    ++history.bad_steps;
    return false;
}

StateDict save_monitor(const MonitorOptions& options, const MetricHistory& history) {
    return {{"mode", integer(static_cast<Index>(options.mode))}, {"patience", integer(options.patience)},
        {"min_delta", scalar(options.min_delta)}, {"best", scalar(history.best)},
        {"bad_steps", integer(history.bad_steps)}, {"initialized", integer(history.initialized)}};
}

void load_monitor(std::span<const NamedTensor> state, MonitorOptions& options, MetricHistory& history) {
    auto values = named(state);
    options = {static_cast<MetricMode>(values.at("mode").item<Index>()), values.at("patience").item<Index>(),
        values.at("min_delta").item<double>()};
    history = {values.at("best").item<double>(), values.at("bad_steps").item<Index>(),
        values.at("initialized").item<Index>() != 0};
}

} // namespace

bool EarlyStopping::step(double metric) {
    observe(metric, options_, history_);
    return history_.bad_steps >= options_.patience;
}
StateDict EarlyStopping::state_dict() const { return save_monitor(options_, history_); }
void EarlyStopping::load_state_dict(std::span<const NamedTensor> state) { load_monitor(state, options_, history_); }

ReduceLROnPlateau::ReduceLROnPlateau(optim::Optimizer& optimizer, PlateauOptions options)
    : optimizer_(optimizer), options_(options) {}

bool ReduceLROnPlateau::step(double metric) {
    observe(metric, options_.monitor, history_);
    if (history_.bad_steps <= options_.monitor.patience) return false;
    history_.bad_steps = 0;
    bool changed = false;
    for (std::size_t i = 0; i < optimizer_.groups().size(); ++i) {
        const auto old = optimizer_.learning_rate(i);
        const auto rate = std::max(old * options_.factor, options_.minimum_rate);
        if (rate < old) { optimizer_.set_learning_rate(rate, i); changed = true; }
    }
    return changed;
}
StateDict ReduceLROnPlateau::state_dict() const {
    auto state = save_monitor(options_.monitor, history_);
    state.emplace_back("factor", scalar(options_.factor));
    state.emplace_back("minimum_rate", scalar(options_.minimum_rate));
    return state;
}
void ReduceLROnPlateau::load_state_dict(std::span<const NamedTensor> state) {
    load_monitor(state, options_.monitor, history_);
    auto values = named(state);
    options_.factor = values.at("factor").item<double>();
    options_.minimum_rate = values.at("minimum_rate").item<double>();
}

ExponentialMovingAverage::ExponentialMovingAverage(const nn::Module& model, double decay) : decay_(decay) {
    std::unordered_set<std::string> parameters;
    for (const auto& [name, tensor] : model.named_parameters()) parameters.insert(name);
    for (const auto& [name, tensor] : model.state_dict()) {
        const bool parameter = parameters.contains(name);
        averages_.emplace_back(name, tensor.detach().astype(parameter ? accumulation_dtype(tensor.dtype()) : tensor.dtype()).clone());
        parameters_.push_back(parameter);
    }
}

void ExponentialMovingAverage::update(const nn::Module& model) {
    auto values = named(model.state_dict());
    for (std::size_t i = 0; i < averages_.size(); ++i) {
        auto& [name, tensor] = averages_[i];
        auto source = values.at(name).detach().to(tensor.device()).astype(tensor.dtype());
        auto& target = detail::TensorAccess::mutable_data(tensor);
        if (parameters_[i]) detail::numeric::momentum(target, detail::data_of(source), decay_, 1.0 - decay_);
        else detail::numeric::copy_into(target, detail::data_of(source));
    }
    ++steps_;
}

void ExponentialMovingAverage::copy_to(nn::Module& model) const {
    auto destination = named(model.state_dict());
    StateDict state;
    for (const auto& [name, tensor] : averages_) state.emplace_back(name, tensor.astype(destination.at(name).dtype()));
    model.load_state_dict(state);
}

StateDict ExponentialMovingAverage::state_dict() const {
    StateDict state{{"decay", scalar(decay_)}, {"steps", integer(steps_)}};
    for (const auto& [name, tensor] : averages_) state.emplace_back("model." + name, tensor);
    return state;
}

void ExponentialMovingAverage::load_state_dict(std::span<const NamedTensor> state) {
    auto values = named(state);
    decay_ = values.at("decay").item<double>();
    steps_ = values.at("steps").item<Index>();
    for (auto& [name, tensor] : averages_) {
        auto source = values.at("model." + name).to(tensor.device());
        detail::numeric::copy_into(detail::TensorAccess::mutable_data(tensor), detail::data_of(source));
    }
}

} // namespace dl::training
