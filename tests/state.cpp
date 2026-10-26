#include "test_support.h"

#include <filesystem>

namespace {

struct Tied : dl::nn::Module {
    std::shared_ptr<dl::nn::Linear> a, b;
    explicit Tied(bool reverse) {
        auto weight = test::tensor({.3}, {1, 1}, true);
        a = std::make_shared<dl::nn::Linear>(weight, test::tensor({.1}, {1}, true));
        b = std::make_shared<dl::nn::Linear>(weight, test::tensor({-.2}, {1}, true));
        if (reverse) { register_module("b", b); register_module("a", a); }
        else { register_module("a", a); register_module("b", b); }
    }
    dl::Tensor operator()(const dl::Tensor& x) { return (*a)(x) + (*b)(x); }
};

struct Registry : dl::nn::Module {
    using Module::register_parameter;
    using Module::register_buffer;
    using Module::register_module;
};

struct Scale : dl::nn::UnaryModule {
    dl::Tensor weight = register_parameter("weight", test::tensor({2}, {}, true));
    dl::Tensor forward(const dl::Tensor& x) override { return x * bound(weight); }
};

template <class Function> void rejects(Function function) {
    bool rejected = false;
    try { function(); } catch (const std::invalid_argument&) { rejected = true; }
    test::expect(rejected, "registration rejects ambiguous names");
}

void update(Tied& model, dl::optim::Optimizer& optimizer) {
    optimizer.zero_grad();
    auto x = test::tensor({-1, 0, 1}, {3, 1});
    auto target = test::tensor({-1, 1, 3}, {3, 1});
    dl::mse_loss(model(x), target).backward();
    optimizer.step();
}

void legacy_names(dl::io::TrainingCheckpoint& state) {
    for (auto& [name, value] : state.model) if (name == "a.weight") name = "b.weight";
    for (auto& name : state.optimizer_parameters) if (name == "a.weight") name = "b.weight";
    for (auto& [name, enabled] : *state.gradient_modes) if (name == "a.weight") name = "b.weight";
}

} // namespace

int main() {
    return test::run([] {
        dl::manual_seed(17);
        dl::nn::TransformerDecoder decoder(2, 8, 4, 2, 12, 16, {.dtype = dl::DType::f64});
        auto input = dl::arange(48, {.dtype = dl::DType::f64}).reshape({1, 6, 8}) / 48.0;
        auto full = decoder(input);
        auto cache = decoder.make_cache();
        auto prefix = decoder.decode(input.narrow(1, 0, 2), cache);
        auto suffix = decoder.decode(input.narrow(1, 2, 4), cache);
        auto cached = dl::concat({prefix, suffix}, 1).to_vector<double>();
        auto expected_full = full.to_vector<double>();
        for (std::size_t i = 0; i < cached.size(); ++i) test::close(cached[i], expected_full[i]);
        for (auto& layer : cache) layer.truncate(2);
        test::expect(cache.front().length() == 2, "cache keeps an existing prefix");
        dl::nn::Linear base(3, 2, false, {.dtype = dl::DType::f64});
        dl::nn::LoRALinear adapter(base, 2, 4);
        auto adapted = adapter(test::tensor({1, 2, 3}, {1, 3}));
        test::close(dl::sum(adapted).item<double>(),
            dl::sum((*adapter.merged())(test::tensor({1, 2, 3}, {1, 3}))).item<double>());
        dl::Generator data_random(19);
        dl::data::DataLoader data({dl::arange(7), dl::arange(7)}, {.batch_size = 2, .shuffle = false}, data_random);
        dl::data::PrefetchLoader pipeline(data, dl::Device::cpu(), 2, true);
        test::values(pipeline.next()->input, {0, 1});
        auto pending = pipeline.state_dict();
        auto position = data.state_dict();
        auto expected = pipeline.next()->input;
        pipeline.load_state_dict(pending);
        data.load_state_dict(position);
        auto actual = pipeline.next()->input;
        test::expect(expected.to_vector<float>() == actual.to_vector<float>(), "background prefetch restore");
        dl::training::EarlyStopping stop({.patience = 2});
        test::expect(!stop.step(2) && !stop.step(1) && !stop.step(1), "metric improvement");
        dl::training::EarlyStopping resumed;
        resumed.load_state_dict(stop.state_dict());
        test::expect(resumed.step(1), "early stopping restore");
        Registry registry;
        auto parameter = test::tensor({1}, {}, true);
        registry.register_parameter("weight", parameter);
        rejects([&] { registry.register_parameter("weight", test::tensor({2}, {}, true)); });
        rejects([&] { registry.register_buffer("weight", dl::zeros({})); });
        rejects([&] { registry.register_module("weight", std::make_shared<Registry>()); });
        rejects([&] { registry.register_buffer("a.b", dl::zeros({})); });
        rejects([&] { registry.register_parameter("", parameter); });
        test::expect(registry.parameters().size() == 1, "failed registration preserves state");

        Tied source(false), target(true);
        test::expect(source.parameters().size() == 3 && target.parameters().size() == 3, "shared parameter deduplication");
        test::expect(source.state_aliases() == target.state_aliases(), "stable aliases after reordered registration");
        test::expect(target.state_aliases() == std::vector<dl::NameAlias>{{"b.weight", "a.weight"}}, "canonical path");
        dl::optim::AdamW left(source.parameters(), {.learning_rate = .02});
        dl::optim::AdamW right(target.parameters(), {.learning_rate = .02});
        update(source, left);
        auto view = source.state_dict();
        auto snapshot = dl::clone_state(view);
        auto checkpoint = dl::io::capture_checkpoint(source, left, {.step = 1});
        legacy_names(checkpoint);
        const auto path = std::filesystem::path("shared-state.train.ct");
        dl::io::save_checkpoint(path, checkpoint);
        auto loaded = dl::io::load_checkpoint(path);
        auto progress = dl::io::restore_checkpoint(loaded, target, right);
        test::expect(progress.step == 1, "checkpoint progress");
        for (int step = 0; step < 3; ++step) { update(source, left); update(target, right); }
        test::values(source.a->weight() - target.a->weight(), {0});
        test::values(*source.a->bias() - *target.a->bias(), {0});
        test::expect(view[0].second.data<void>() == source.a->weight().data<void>(), "state view identity");
        test::expect(snapshot[0].second.data<void>() != view[0].second.data<void>(), "snapshot owns storage");
        test::expect(snapshot[0].second.to_vector<double>() != view[0].second.to_vector<double>(), "snapshot remains unchanged");
        test::expect(!snapshot[0].second.requires_grad(), "snapshot detaches gradients");
        std::filesystem::remove(path);

        dl::nn::FunctionalModule functional(target);
        loaded.random_key = dl::RandomKey{7};
        auto restored = dl::io::restore_checkpoint(loaded, functional);
        auto x = test::tensor({3}, {1, 1});
        auto value = functional.call(restored.state, restored.random_key, [&] { return target(x); }).output;
        source.load_state_dict(snapshot);
        test::values(value - source(x), {0});
        test::expect(dl::summary(target).find("b.weight -> a.weight") != std::string::npos, "shared summary");

        Scale scale;
        dl::nn::FunctionalModule bound(scale);
        auto state = bound.state();
        state.parameters[0] = test::tensor({5}, {}, true);
        test::values(scale(test::tensor({3}, {})), {6});
        test::values(bound.call(state, dl::RandomKey{0}, [&] { return scale(test::tensor({3}, {})); }).output, {15});
        dl::optim::SGD optimizer(scale.parameters(), {.learning_rate = .1});
        dl::GradScaler scaler({.initial_scale = 8});
        scaler.scale(scale(test::tensor({3}, {}))).backward();
        test::expect(scaler.unscale_(optimizer), "optimizer unscale alias");
        test::gradient(scale.weight, {3});
    });
}
