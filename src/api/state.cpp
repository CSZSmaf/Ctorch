#include "api/common.h"

#include "core/tensor_impl.h"
#include "optim/state.h"

#include <unordered_map>

using namespace dl::capi;

namespace {

// C 调用传入有身份的优化器，将其完整状态转成外部分支的规范参数顺序。
dl::optim::OptimizerState functional_optimizer_state(const dl::nn::FunctionalState& state,
    const dl::optim::Optimizer& optimizer) {
    const auto& parameters = optimizer.parameters();
    std::unordered_map<const void*, std::size_t> source;
    source.reserve(parameters.size());
    for (std::size_t i = 0; i < parameters.size(); ++i) {
        source.emplace(dl::detail::TensorAccess::identity(parameters[i]), i);
    }
    std::vector<std::size_t> order;
    order.reserve(state.parameters.size());
    for (const auto& parameter : state.parameters) {
        order.push_back(source.at(dl::detail::TensorAccess::identity(parameter)));
    }
    return dl::optim::detail::reorder_state(optimizer.state_dict(), order);
}

std::vector<dl::io::NamedGenerator> named_generators(const dl_named_generator* generators, std::size_t count) {
    std::vector<dl::io::NamedGenerator> result;
    result.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        result.push_back({generators[index].name, *generators[index].generator->value});
    }
    return result;
}

template <class Restore>
dl_status restore_functional(Restore restore, dl_functional_state** state, dl_optimizer** optimizer,
    dl_random_key* key, dl_training_progress* progress) {
    *state = nullptr;
    *optimizer = nullptr;
    return invoke([&] {
        auto restored = restore();
        // 优化器从返回的叶集合构造，外部状态与更新对象共享同一组新身份。
        std::shared_ptr<dl::optim::Optimizer> update(dl::optim::from_state(restored.state.parameters, restored.optimizer));
        auto model = std::make_unique<dl_functional_state>(dl_functional_state{std::move(restored.state)});
        auto owner = std::make_unique<dl_optimizer>(dl_optimizer{std::move(update)});
        *key = convert(restored.random_key);
        *progress = {restored.progress.epoch, restored.progress.step};
        *state = model.release();
        *optimizer = owner.release();
    });
}

} // namespace

extern "C" {

void dl_state_release(dl_state* state) { delete state; }
size_t dl_state_size(const dl_state* state) { return state->value.size(); }
const char* dl_state_name(const dl_state* state, size_t index) { return state->value[index].first.c_str(); }
dl_status dl_state_tensor(const dl_state* state, size_t index, dl_tensor** output) {
    return make_handle(output, [&] { return state->value[index].second; });
}
dl_status dl_state_create(const char* const* names, const dl_tensor* const* tensors, size_t count, dl_state** output) {
    return make_handle(output, [&] {
        dl::StateDict result;
        result.reserve(count);
        for (size_t index = 0; index < count; ++index) {
            result.emplace_back(names[index], tensors[index]->value);
        }
        return result;
    });
}
dl_status dl_state_save(const char* path, const dl_state* state) {
    return invoke([&] { dl::io::save(utf8_path(path), state->value); });
}
dl_status dl_state_load(const char* path, dl_state** output) {
    return make_handle(output, [&] { return dl::io::load(utf8_path(path)); });
}

dl_status dl_loader_create(const dl_tensor* input, const dl_tensor* target, dl_loader_options options,
    dl_generator* generator, dl_loader** output) {
    *output = nullptr;
    return invoke([&] {
        auto owner = generator ? generator->value : nullptr;
        auto loader = std::make_shared<dl::data::DataLoader>(dl::data::TensorDataset{input->value, target->value},
            convert(options), random_generator(generator));
        *output = new dl_loader{std::move(owner), std::move(loader)};
    });
}
void dl_loader_release(dl_loader* loader) { delete loader; }
dl_status dl_loader_reset(dl_loader* loader) { return invoke([&] { loader->value->reset(); }); }
dl_status dl_loader_next(dl_loader* loader, dl_tensor** input, dl_tensor** target) {
    *input = nullptr;
    *target = nullptr;
    return invoke([&] {
        if (auto batch = loader->value->next()) {
            auto x = std::make_unique<dl_tensor>(dl_tensor{std::move(batch->input)});
            auto y = std::make_unique<dl_tensor>(dl_tensor{std::move(batch->target)});
            *input = x.release();
            *target = y.release();
        }
    });
}

void dl_checkpoint_release(dl_checkpoint* checkpoint) { delete checkpoint; }

dl_status dl_checkpoint_capture_functional(const dl_functional* functional, const dl_functional_state* state,
    const dl_optimizer* optimizer, dl_random_key key, dl_training_progress progress,
    const dl_loader* loader, const dl_named_generator* generators, size_t count, dl_checkpoint** output) {
    return make_handle(output, [&] {
        auto streams = named_generators(generators, count);
        return dl::io::capture_checkpoint(functional->value, state->value,
            functional_optimizer_state(state->value, *optimizer->value),
            convert(key), {progress.epoch, progress.step}, loader ? loader->value.get() : nullptr, streams);
    });
}

dl_status dl_checkpoint_restore_functional(const dl_checkpoint* checkpoint, const dl_functional* functional,
    dl_device device, dl_loader* loader, const dl_named_generator* generators, size_t count,
    dl_functional_state** state, dl_optimizer** optimizer, dl_random_key* key, dl_training_progress* progress) {
    return restore_functional([&] {
        auto streams = named_generators(generators, count);
        return dl::io::restore_checkpoint(checkpoint->value, functional->value, convert(device),
            loader ? loader->value.get() : nullptr, streams);
    }, state, optimizer, key, progress);
}

dl_status dl_checkpoint_capture_functional_prefetch(const dl_functional* functional, const dl_functional_state* state,
    const dl_optimizer* optimizer, dl_random_key key, dl_training_progress progress,
    const dl_prefetch_loader* loader, const dl_named_generator* generators, size_t count, dl_checkpoint** output) {
    return make_handle(output, [&] {
        auto streams = named_generators(generators, count);
        return dl::io::capture_checkpoint(functional->value, state->value,
            functional_optimizer_state(state->value, *optimizer->value),
            convert(key), {progress.epoch, progress.step}, *loader->value, streams);
    });
}

dl_status dl_checkpoint_restore_functional_prefetch(const dl_checkpoint* checkpoint, const dl_functional* functional,
    dl_device device, dl_prefetch_loader* loader, const dl_named_generator* generators, size_t count,
    dl_functional_state** state, dl_optimizer** optimizer, dl_random_key* key, dl_training_progress* progress) {
    return restore_functional([&] {
        auto streams = named_generators(generators, count);
        return dl::io::restore_checkpoint(checkpoint->value, functional->value, convert(device), *loader->value, streams);
    }, state, optimizer, key, progress);
}

dl_status dl_checkpoint_set_random_key(dl_checkpoint* checkpoint, dl_random_key key) {
    return invoke([&] { checkpoint->value.random_key = convert(key); });
}
dl_status dl_checkpoint_random_key(const dl_checkpoint* checkpoint, dl_random_key* output, int32_t* present) {
    return invoke([&] {
        *present = checkpoint->value.random_key.has_value();
        if (*present) *output = convert(*checkpoint->value.random_key);
    });
}
dl_status dl_checkpoint_capture(const dl_module* model, const dl_optimizer* optimizer, dl_training_progress progress,
    const dl_loader* loader, const dl_named_generator* generators, size_t count, dl_checkpoint** output) {
    return make_handle(output, [&] {
        auto streams = named_generators(generators, count);
        return dl::io::capture_checkpoint(*model->value, *optimizer->value, {progress.epoch, progress.step},
            loader ? loader->value.get() : nullptr, streams);
    });
}
dl_status dl_checkpoint_restore(const dl_checkpoint* checkpoint, dl_module* model, dl_optimizer* optimizer,
    dl_loader* loader, const dl_named_generator* generators, size_t count, dl_training_progress* progress) {
    return invoke([&] {
        auto streams = named_generators(generators, count);
        const auto restored = dl::io::restore_checkpoint(checkpoint->value, *model->value, *optimizer->value,
            loader ? loader->value.get() : nullptr, streams);
        *progress = {restored.epoch, restored.step};
    });
}
dl_status dl_checkpoint_save(const char* path, const dl_checkpoint* checkpoint) {
    return invoke([&] { dl::io::save_checkpoint(utf8_path(path), checkpoint->value); });
}
dl_status dl_checkpoint_capture_prefetch(const dl_module* model, const dl_optimizer* optimizer,
    dl_training_progress progress, const dl_prefetch_loader* loader, const dl_named_generator* generators,
    size_t count, dl_checkpoint** output) {
    return make_handle(output, [&] {
        auto streams = named_generators(generators, count);
        return dl::io::capture_checkpoint(*model->value, *optimizer->value, {progress.epoch, progress.step},
            *loader->value, streams);
    });
}
dl_status dl_checkpoint_restore_prefetch(const dl_checkpoint* checkpoint, dl_module* model,
    dl_optimizer* optimizer, dl_prefetch_loader* loader, const dl_named_generator* generators,
    size_t count, dl_training_progress* progress) {
    return invoke([&] {
        auto streams = named_generators(generators, count);
        const auto restored = dl::io::restore_checkpoint(checkpoint->value, *model->value, *optimizer->value,
            *loader->value, streams);
        *progress = {restored.epoch, restored.step};
    });
}
dl_status dl_checkpoint_load(const char* path, dl_checkpoint** output) {
    return make_handle(output, [&] { return dl::io::load_checkpoint(utf8_path(path)); });
}
dl_status dl_checkpoint_extra(const dl_checkpoint* checkpoint, dl_state** output) {
    return make_handle(output, [&] { return checkpoint->value.extra; });
}
dl_status dl_checkpoint_set_extra(dl_checkpoint* checkpoint, const dl_state* state) {
    return invoke([&] {
        auto values = state->value;
        for (auto& [name, tensor] : values) {
            auto value = tensor.detach();
            tensor = value.device() == dl::Device::cpu() ? value.clone() : value.to(dl::Device::cpu());
        }
        checkpoint->value.extra = std::move(values);
    });
}

} // extern "C"
