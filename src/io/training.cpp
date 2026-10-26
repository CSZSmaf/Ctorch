#include "dl/io.h"

#include "core/tensor_impl.h"
#include "io/archive.h"
#include "io/training_archive.h"
#include "io/output_file.h"
#include "nn/state_names.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace dl::io {
namespace {

constexpr std::array<char, 8> format{'C', 'T', 'R', 'A', 'I', 'N', 0, 7};

int read_training_header(std::istream& input) {
    std::array<char, 8> header;
    input.read(header.data(), header.size());
    if (!std::equal(format.begin(), format.end() - 1, header.begin()) || header[7] < 1 || header[7] > 7) {
        throw std::runtime_error("unsupported training checkpoint format");
    }
    return header[7];
}

std::vector<std::string> parameter_names(const nn::Module& model, const optim::Optimizer& optimizer) {
    const auto parameters = model.named_parameters();
    std::unordered_map<const void*, std::string_view> names;
    for (const auto& [name, tensor] : parameters) {
        names.emplace(detail::TensorAccess::identity(tensor), name);
    }
    std::vector<std::string> result;
    result.reserve(optimizer.parameters().size());
    for (const auto& tensor : optimizer.parameters()) {
        result.emplace_back(names.at(detail::TensorAccess::identity(tensor)));
    }
    return result;
}

void write_modes(std::ostream& output, std::span<const std::pair<std::string, bool>> modes) {
    archive::write_integer(output, static_cast<std::uint64_t>(modes.size()));
    for (const auto& [name, enabled] : modes) {
        archive::write_text(output, name);
        archive::write_integer(output, static_cast<std::uint8_t>(enabled));
    }
}

nn::TrainingModes read_modes(std::istream& input) {
    nn::TrainingModes result;
    const auto count = archive::read_integer<std::uint64_t>(input);
    result.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t index = 0; index < count; ++index) {
        auto name = archive::read_text(input);
        result.emplace_back(std::move(name), archive::read_integer<std::uint8_t>(input) != 0);
    }
    return result;
}

void write_loader(std::ostream& output, const data::LoaderState& state) {
    using namespace archive;
    write_integer(output, static_cast<std::uint64_t>(state.options.batch_size));
    write_integer(output, static_cast<std::uint8_t>(state.options.shuffle));
    write_integer(output, static_cast<std::uint8_t>(state.options.drop_last));
    write_integer(output, static_cast<std::uint64_t>(state.cursor));
    write_integer(output, static_cast<std::uint64_t>(state.order.size()));
    for (auto index : state.order) {
        write_integer(output, static_cast<std::uint64_t>(index));
    }
    write_text(output, state.random_state);
}

data::LoaderState read_loader(std::istream& input) {
    using namespace archive;
    data::LoaderState state;
    state.options = {static_cast<Index>(read_integer<std::uint64_t>(input)),
        read_integer<std::uint8_t>(input) != 0, read_integer<std::uint8_t>(input) != 0};
    state.cursor = static_cast<Index>(read_integer<std::uint64_t>(input));
    state.order.resize(static_cast<std::size_t>(read_integer<std::uint64_t>(input)));
    for (auto& index : state.order) {
        index = static_cast<Index>(read_integer<std::uint64_t>(input));
    }
    state.random_state = read_text(input);
    return state;
}

void write_scaler(std::ostream& output, const GradScalerState& state) {
    using namespace archive;
    write_double(output, state.options.initial_scale);
    write_double(output, state.options.growth_factor);
    write_double(output, state.options.backoff_factor);
    write_integer(output, static_cast<std::uint64_t>(state.options.growth_interval));
    write_integer(output, static_cast<std::uint8_t>(state.options.enabled));
    write_double(output, state.scale);
    write_integer(output, static_cast<std::uint64_t>(state.finite_steps));
}

GradScalerState read_scaler(std::istream& input) {
    using namespace archive;
    GradScalerState state;
    state.options.initial_scale = read_double(input);
    state.options.growth_factor = read_double(input);
    state.options.backoff_factor = read_double(input);
    state.options.growth_interval = static_cast<Index>(read_integer<std::uint64_t>(input));
    state.options.enabled = read_integer<std::uint8_t>(input) != 0;
    state.scale = read_double(input);
    state.finite_steps = static_cast<Index>(read_integer<std::uint64_t>(input));
    return state;
}

void capture_runtime(TrainingCheckpoint& result, TrainingProgress progress,
    const data::DataLoader* loader, std::span<const NamedGenerator> generators,
    std::span<const NamedScheduler> schedulers, const GradScaler* scaler) {
    result.progress = progress;
    result.random_state = default_generator().state();
    if (loader) {
        result.loader = loader->state_dict();
    }
    for (const auto& entry : generators) {
        result.generators.emplace_back(entry.name, entry.generator.get().state());
    }
    for (const auto& entry : schedulers) {
        result.schedulers.emplace_back(entry.name, entry.scheduler.get().state_dict());
    }
    if (scaler) result.scaler = scaler->state_dict();
}

void restore_runtime(const TrainingCheckpoint& checkpoint, data::DataLoader* loader,
    std::span<const NamedGenerator> generators,
    std::span<const NamedScheduler> schedulers, GradScaler* scaler) {
    if (checkpoint.loader) {
        loader->load_state_dict(*checkpoint.loader);
    }
    for (const auto& entry : generators) {
        const auto source = std::find_if(checkpoint.generators.begin(), checkpoint.generators.end(),
            [&](const auto& value) { return value.first == entry.name; });
        entry.generator.get().load_state(source->second);
    }
    for (const auto& entry : schedulers) {
        const auto source = std::find_if(checkpoint.schedulers.begin(), checkpoint.schedulers.end(),
            [&](const auto& value) { return value.first == entry.name; });
        entry.scheduler.get().load_state_dict(source->second);
    }
    if (scaler && checkpoint.scaler) scaler->load_state_dict(*checkpoint.scaler);
    default_generator().load_state(checkpoint.random_state);
}

void require_runtime(const TrainingCheckpoint& checkpoint, const data::DataLoader* loader,
    std::span<const NamedGenerator> generators, std::span<const NamedScheduler> schedulers, const GradScaler* scaler) {
    if (checkpoint.loader && !loader) throw std::invalid_argument("checkpoint restoration requires its data loader");
    if (checkpoint.scaler && !scaler) throw std::invalid_argument("checkpoint restoration requires its gradient scaler");
    auto same_names = [](const auto& saved, const auto& supplied, const char* kind) {
        std::vector<std::string_view> expected, actual;
        for (const auto& entry : saved) expected.push_back(entry.first);
        for (const auto& entry : supplied) actual.push_back(entry.name);
        std::sort(expected.begin(), expected.end());
        std::sort(actual.begin(), actual.end());
        if (expected != actual) throw std::invalid_argument(std::string("checkpoint ") + kind + " names differ");
    };
    same_names(checkpoint.generators, generators, "generator");
    same_names(checkpoint.schedulers, schedulers, "scheduler");
}

} // namespace

TrainingCheckpoint capture_checkpoint(const nn::Module& model, const optim::Optimizer& optimizer,
    TrainingProgress progress, const data::DataLoader* loader, std::span<const NamedGenerator> generators,
    std::span<const NamedScheduler> schedulers, const GradScaler* scaler) {
    TrainingCheckpoint result;
    result.model = clone_state(model.state_dict());
    result.modes = model.training_modes();
    result.gradient_modes = model.gradient_modes();
    result.optimizer_parameters = parameter_names(model, optimizer);
    result.optimizer = archive::copy_optimizer(optimizer.state_dict(), Device::cpu());
    capture_runtime(result, progress, loader, generators, schedulers, scaler);
    return result;
}

TrainingProgress restore_checkpoint(const TrainingCheckpoint& checkpoint, nn::Module& model,
    optim::Optimizer& optimizer, data::DataLoader* loader, std::span<const NamedGenerator> generators,
    std::span<const NamedScheduler> schedulers, GradScaler* scaler) {
    require_runtime(checkpoint, loader, generators, schedulers, scaler);
    if (checkpoint.optimizer.index() != optimizer.state_dict().index())
        throw std::invalid_argument("checkpoint optimizer type differs");
    const auto names = parameter_names(model, optimizer);
    const auto source_names = nn::state_detail::canonical_names(checkpoint.optimizer_parameters, model.state_aliases());
    if (source_names.size() != names.size()) throw std::invalid_argument("checkpoint optimizer parameter counts differ");
    auto optimizer_state = archive::reorder_optimizer(checkpoint.optimizer, source_names, names);
    model.load_state_dict(checkpoint.model);
    model.load_training_modes(checkpoint.modes);
    if (checkpoint.gradient_modes) model.load_gradient_modes(*checkpoint.gradient_modes);
    optimizer.load_state_dict(optimizer_state);
    for (const auto& parameter : model.parameters()) parameter.zero_grad();
    restore_runtime(checkpoint, loader, generators, schedulers, scaler);
    return checkpoint.progress;
}

TrainingCheckpoint capture_checkpoint(const nn::FunctionalModule& functional, const nn::FunctionalState& state,
    const optim::OptimizerState& optimizer, RandomKey random_key, TrainingProgress progress,
    const data::DataLoader* loader, std::span<const NamedGenerator> generators, const GradScaler* scaler) {
    TrainingCheckpoint result;
    auto append = [&](std::span<const std::string> names, std::span<const Tensor> values) {
        for (std::size_t i = 0; i < names.size(); ++i) {
            result.model.emplace_back(names[i], archive::copy_tensor(values[i], Device::cpu()));
        }
    };
    append(functional.parameter_names(), state.parameters);
    append(functional.buffer_names(), state.buffers);
    result.gradient_modes.emplace();
    for (std::size_t i = 0; i < state.parameters.size(); ++i) {
        result.gradient_modes->emplace_back(functional.parameter_names()[i], state.parameters[i].requires_grad());
    }
    for (std::size_t i = 0; i < state.training.size(); ++i) {
        result.modes.emplace_back(functional.module_names()[i], state.training[i] != 0);
    }
    result.optimizer_parameters.assign(functional.parameter_names().begin(), functional.parameter_names().end());
    result.optimizer = archive::copy_optimizer(optimizer, Device::cpu());
    result.random_key = random_key;
    capture_runtime(result, progress, loader, generators, {}, scaler);
    return result;
}

FunctionalTrainingState restore_checkpoint(const TrainingCheckpoint& checkpoint,
    const nn::FunctionalModule& functional, Device device, data::DataLoader* loader,
    std::span<const NamedGenerator> generators, GradScaler* scaler) {
    require_runtime(checkpoint, loader, generators, {}, scaler);
    FunctionalTrainingState result{{}, {}, checkpoint.random_key.value(), checkpoint.progress};
    std::unordered_map<std::string_view, const Tensor*> values;
    for (const auto& [name, tensor] : checkpoint.model) values.emplace(name, &tensor);
    std::unordered_map<std::string_view, bool> gradients, modes;
    for (const auto& [name, enabled] : checkpoint.gradient_modes.value()) gradients.emplace(name, enabled);
    for (const auto& [name, enabled] : checkpoint.modes) modes.emplace(name, enabled);
    const auto aliases = functional.state_aliases();
    nn::state_detail::add_aliases(values, aliases);
    nn::state_detail::add_aliases(gradients, aliases);

    // 名称决定状态对应关系，新叶的身份独立于原 Module，也独立于 CPU 快照。
    for (const auto& name : functional.parameter_names()) {
        auto parameter = archive::copy_tensor(*values.at(name), device);
        parameter.requires_grad_(gradients.at(name));
        result.state.parameters.push_back(std::move(parameter));
    }
    for (const auto& name : functional.buffer_names()) {
        result.state.buffers.push_back(archive::copy_tensor(*values.at(name), device));
    }
    for (const auto& name : functional.module_names()) result.state.training.push_back(modes.at(name));
    const auto source_names = nn::state_detail::canonical_names(checkpoint.optimizer_parameters, aliases);
    result.optimizer = archive::copy_optimizer(archive::reorder_optimizer(checkpoint.optimizer,
        source_names, functional.parameter_names()), device);
    restore_runtime(checkpoint, loader, generators, {}, scaler);
    return result;
}

TrainingCheckpoint capture_checkpoint(const nn::Module& model, const optim::Optimizer& optimizer,
    TrainingProgress progress, const data::PrefetchLoader& loader, std::span<const NamedGenerator> generators,
    std::span<const NamedScheduler> schedulers, const GradScaler* scaler) {
    auto pending = loader.state_dict();
    auto result = capture_checkpoint(model, optimizer, progress, &loader.source(), generators, schedulers, scaler);
    result.prefetch = std::move(pending);
    return result;
}

TrainingProgress restore_checkpoint(const TrainingCheckpoint& checkpoint, nn::Module& model,
    optim::Optimizer& optimizer, data::PrefetchLoader& loader, std::span<const NamedGenerator> generators,
    std::span<const NamedScheduler> schedulers, GradScaler* scaler) {
    require_runtime(checkpoint, &loader.source(), generators, schedulers, scaler);
    loader.load_state_dict(checkpoint.prefetch.value_or(data::PrefetchState{}));
    auto progress = restore_checkpoint(checkpoint, model, optimizer, &loader.source(), generators, schedulers, scaler);
    return progress;
}

TrainingCheckpoint capture_checkpoint(const nn::FunctionalModule& functional, const nn::FunctionalState& state,
    const optim::OptimizerState& optimizer, RandomKey random_key, TrainingProgress progress,
    const data::PrefetchLoader& loader, std::span<const NamedGenerator> generators, const GradScaler* scaler) {
    auto pending = loader.state_dict();
    auto result = capture_checkpoint(functional, state, optimizer, random_key, progress, &loader.source(), generators, scaler);
    result.prefetch = std::move(pending);
    return result;
}

FunctionalTrainingState restore_checkpoint(const TrainingCheckpoint& checkpoint,
    const nn::FunctionalModule& functional, Device device, data::PrefetchLoader& loader,
    std::span<const NamedGenerator> generators, GradScaler* scaler) {
    require_runtime(checkpoint, &loader.source(), generators, {}, scaler);
    loader.load_state_dict(checkpoint.prefetch.value_or(data::PrefetchState{}));
    auto result = restore_checkpoint(checkpoint, functional, device, &loader.source(), generators, scaler);
    return result;
}

void save_checkpoint(const std::filesystem::path& path, const TrainingCheckpoint& checkpoint) {
    using namespace archive;
    OutputFile file(path);
    auto& output = file.stream();
    output.write(format.data(), format.size());
    write_state(output, checkpoint.model);
    write_modes(output, checkpoint.modes);
    write_integer(output, static_cast<std::uint64_t>(checkpoint.optimizer_parameters.size()));
    for (const auto& name : checkpoint.optimizer_parameters) {
        write_text(output, name);
    }
    write_optimizer(output, checkpoint.optimizer);
    write_integer(output, static_cast<std::uint64_t>(checkpoint.progress.epoch));
    write_integer(output, static_cast<std::uint64_t>(checkpoint.progress.step));
    write_text(output, checkpoint.random_state);
    write_integer(output, static_cast<std::uint8_t>(checkpoint.loader.has_value()));
    if (checkpoint.loader) {
        write_loader(output, *checkpoint.loader);
    }
    write_integer(output, static_cast<std::uint64_t>(checkpoint.generators.size()));
    for (const auto& [name, state] : checkpoint.generators) {
        write_text(output, name);
        write_text(output, state);
    }
    write_integer(output, static_cast<std::uint64_t>(checkpoint.schedulers.size()));
    for (const auto& [name, state] : checkpoint.schedulers) {
        write_text(output, name);
        write_scheduler(output, state);
    }
    write_state(output, checkpoint.extra);
    write_integer(output, static_cast<std::uint8_t>(checkpoint.prefetch.has_value()));
    if (checkpoint.prefetch) {
        write_integer(output, static_cast<std::uint64_t>(checkpoint.prefetch->depth));
        write_integer(output, static_cast<std::uint8_t>(checkpoint.prefetch->background));
        write_integer(output, static_cast<std::uint64_t>(checkpoint.prefetch->pending.size()));
        for (const auto& batch : checkpoint.prefetch->pending) {
            StateDict values{{"input", batch.input}, {"target", batch.target}};
            for (std::size_t i = 0; i < batch.extra.size(); ++i) values.emplace_back("extra." + std::to_string(i), batch.extra[i]);
            write_state(output, values);
        }
    }
    write_integer(output, static_cast<std::uint8_t>(checkpoint.scaler.has_value()));
    if (checkpoint.scaler) write_scaler(output, *checkpoint.scaler);
    write_integer(output, static_cast<std::uint8_t>(checkpoint.gradient_modes.has_value()));
    if (checkpoint.gradient_modes) write_modes(output, *checkpoint.gradient_modes);
    write_integer(output, static_cast<std::uint8_t>(checkpoint.random_key.has_value()));
    if (checkpoint.random_key) {
        write_integer(output, checkpoint.random_key->seed);
        write_integer(output, checkpoint.random_key->stream);
    }
    file.commit();
}

TrainingCheckpoint load_checkpoint(const std::filesystem::path& path) {
    using namespace archive;
    std::ifstream input;
    input.exceptions(std::ios::failbit | std::ios::badbit);
    input.open(path, std::ios::binary);
    const auto version = read_training_header(input);
    const bool extended = version >= 2;
    TrainingCheckpoint result;
    result.model = read_state(input);
    result.modes = read_modes(input);
    const auto parameter_count = read_integer<std::uint64_t>(input);
    for (std::uint64_t index = 0; index < parameter_count; ++index) {
        result.optimizer_parameters.push_back(read_text(input));
    }
    result.optimizer = read_optimizer(input, extended);
    result.progress = {static_cast<Index>(read_integer<std::uint64_t>(input)),
        static_cast<Index>(read_integer<std::uint64_t>(input))};
    result.random_state = read_text(input);
    if (read_integer<std::uint8_t>(input) != 0) {
        result.loader = read_loader(input);
    }
    const auto generator_count = read_integer<std::uint64_t>(input);
    for (std::uint64_t index = 0; index < generator_count; ++index) {
        auto name = read_text(input);
        result.generators.emplace_back(std::move(name), read_text(input));
    }
    if (extended) {
        const auto scheduler_count = read_integer<std::uint64_t>(input);
        for (std::uint64_t index = 0; index < scheduler_count; ++index) {
            auto name = read_text(input);
            result.schedulers.emplace_back(std::move(name), read_scheduler(input));
        }
    }
    result.extra = read_state(input);
    if (version >= 3 && read_integer<std::uint8_t>(input) != 0) {
        data::PrefetchState state;
        state.depth = static_cast<std::size_t>(read_integer<std::uint64_t>(input));
        if (version >= 7) state.background = read_integer<std::uint8_t>(input) != 0;
        const auto count = read_integer<std::uint64_t>(input);
        for (std::uint64_t index = 0; index < count; ++index) {
            auto values = read_state(input);
            data::Batch batch{std::move(values[0].second), std::move(values[1].second), {}};
            for (std::size_t i = 2; i < values.size(); ++i) batch.extra.push_back(std::move(values[i].second));
            state.pending.push_back(std::move(batch));
        }
        result.prefetch = std::move(state);
    }
    if (version >= 4 && read_integer<std::uint8_t>(input) != 0) result.scaler = read_scaler(input);
    if (version >= 5 && read_integer<std::uint8_t>(input) != 0) result.gradient_modes = read_modes(input);
    if (version >= 6 && read_integer<std::uint8_t>(input) != 0) {
        result.random_key = RandomKey{read_integer<std::uint64_t>(input), read_integer<std::uint64_t>(input)};
    }
    return result;
}

void set_extra(TrainingCheckpoint& checkpoint, std::string_view name, std::span<const NamedTensor> state) {
    const auto prefix = std::string(name) + ".";
    auto copy = clone_state(state);
    std::erase_if(checkpoint.extra, [&](const auto& entry) { return entry.first.starts_with(prefix); });
    for (auto& [key, tensor] : copy) checkpoint.extra.emplace_back(prefix + key, std::move(tensor));
}

StateDict get_extra(const TrainingCheckpoint& checkpoint, std::string_view name) {
    const auto prefix = std::string(name) + ".";
    StateDict result;
    for (const auto& [key, tensor] : checkpoint.extra)
        if (key.starts_with(prefix)) result.emplace_back(key.substr(prefix.size()), tensor);
    return result;
}

} // namespace dl::io
