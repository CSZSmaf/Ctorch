#include "dl/optim.h"

#include "optim/update.h"

#include <numeric>
#include <unordered_set>

namespace dl::optim {

Optimizer::Optimizer(std::vector<Tensor> parameters, double learning_rate, double weight_decay)
    : Optimizer(std::vector<ParameterGroup>{{std::move(parameters), {}, {}}}, learning_rate, weight_decay) {}

Optimizer::Optimizer(std::vector<ParameterGroup> groups, double learning_rate, double weight_decay) {
    std::unordered_set<const void*> seen;
    for (auto& group : groups) {
        ParameterGroupState saved{{}, group.learning_rate.value_or(learning_rate), group.weight_decay.value_or(weight_decay)};
        for (auto& parameter : group.parameters) {
            if (seen.insert(dl::detail::TensorAccess::identity(parameter)).second) {
                saved.parameters.push_back(parameters_.size());
                parameters_.push_back(std::move(parameter));
            }
        }
        groups_.push_back(std::move(saved));
    }
}

void Optimizer::load_groups(const std::vector<ParameterGroupState>& groups, double learning_rate, double weight_decay) {
    groups_ = groups;
    if (groups_.empty()) {
        ParameterGroupState group{std::vector<std::size_t>(parameters_.size()), learning_rate, weight_decay};
        std::iota(group.parameters.begin(), group.parameters.end(), 0);
        groups_.push_back(std::move(group));
    }
}

void Optimizer::zero_grad() {
    for (const auto& parameter : parameters_) {
        parameter.zero_grad();
    }
}

SGD::SGD(std::vector<Tensor> parameters, SGDOptions options)
    : Optimizer(std::move(parameters), options.learning_rate, options.weight_decay), options_(options), velocity_(parameters_.size()) {}

SGD::SGD(std::vector<ParameterGroup> groups, SGDOptions options)
    : Optimizer(std::move(groups), options.learning_rate, options.weight_decay), options_(options), velocity_(parameters_.size()) {}

void SGD::step() {
    detail::batch_parameters(parameters_, groups_, velocity_, options_, detail::prepare_sgd_update,
        dl::detail::numeric::sgd);
}

AdamW::AdamW(std::vector<Tensor> parameters, AdamWOptions options)
    : Optimizer(std::move(parameters), options.learning_rate, options.weight_decay), options_(options), states_(parameters_.size()) {}

AdamW::AdamW(std::vector<ParameterGroup> groups, AdamWOptions options)
    : Optimizer(std::move(groups), options.learning_rate, options.weight_decay), options_(options), states_(parameters_.size()) {}

Adam::Adam(std::vector<Tensor> parameters, AdamOptions options)
    : Optimizer(std::move(parameters), options.learning_rate, options.weight_decay), options_(options), states_(parameters_.size()) {}

Adam::Adam(std::vector<ParameterGroup> groups, AdamOptions options)
    : Optimizer(std::move(groups), options.learning_rate, options.weight_decay), options_(options), states_(parameters_.size()) {}

void AdamW::step() {
    detail::step_adam_parameters(parameters_, groups_, states_, options_, true);
}

void Adam::step() {
    detail::step_adam_parameters(parameters_, groups_, states_,
        {options_.learning_rate, options_.beta1, options_.beta2, options_.epsilon, options_.weight_decay}, false);
}

RMSprop::RMSprop(std::vector<Tensor> parameters, RMSpropOptions options)
    : Optimizer(std::move(parameters), options.learning_rate, options.weight_decay), options_(options), states_(parameters_.size()) {}

RMSprop::RMSprop(std::vector<ParameterGroup> groups, RMSpropOptions options)
    : Optimizer(std::move(groups), options.learning_rate, options.weight_decay), options_(options), states_(parameters_.size()) {}

void RMSprop::step() {
    detail::batch_parameters(parameters_, groups_, states_, options_, detail::prepare_rmsprop_update,
        dl::detail::numeric::rmsprop);
}

Adagrad::Adagrad(std::vector<Tensor> parameters, AdagradOptions options)
    : Optimizer(std::move(parameters), options.learning_rate, options.weight_decay), options_(options), states_(parameters_.size()) {}

Adagrad::Adagrad(std::vector<ParameterGroup> groups, AdagradOptions options)
    : Optimizer(std::move(groups), options.learning_rate, options.weight_decay), options_(options), states_(parameters_.size()) {}

void Adagrad::step() {
    detail::batch_parameters(parameters_, groups_, states_, options_, detail::prepare_adagrad_update,
        dl::detail::numeric::adagrad);
}

} // namespace dl::optim
