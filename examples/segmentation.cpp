#include <dl/dl.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <random>

namespace {

// 两次卷积之间降采样，再把低分辨率特征与浅层特征合并。
class Segmenter : public dl::nn::UnaryModule {
public:
    Segmenter() {
        encoder = add_module<dl::nn::Conv2d>("encoder", 1, 4, 3, dl::Conv2dOptions{.padding = 1});
        center = add_module<dl::nn::Conv2d>("center", 4, 8, 3, dl::Conv2dOptions{.padding = 1});
        decoder = add_module<dl::nn::Conv2d>("decoder", 12, 4, 3, dl::Conv2dOptions{.padding = 1});
        head = add_module<dl::nn::Conv2d>("head", 4, 1, 1);
    }
    dl::Tensor forward(const dl::Tensor& input) override {
        auto skip = dl::relu((*encoder)(input));
        auto down = dl::relu((*center)(dl::max_pool2d(skip)));
        auto up = dl::interpolate(down, {input.size(2), input.size(3)}, dl::InterpolationMode::bilinear);
        return (*head)(dl::relu((*decoder)(dl::concat({up, skip}, 1))));
    }
private:
    std::shared_ptr<dl::nn::Conv2d> encoder, center, decoder, head;
};

// 输入是带噪声的圆形图像，目标是圆形区域的二值掩码。
dl::data::TensorDataset circles(dl::Index count, std::uint64_t seed) {
    std::mt19937_64 random(seed);
    std::uniform_real_distribution<float> coordinate(4, 11), radius(2, 4);
    std::normal_distribution<float> noise(0, .25f);
    std::vector<float> images, masks;
    for (dl::Index n = 0; n < count; ++n) {
        const auto cx = coordinate(random), cy = coordinate(random), r = radius(random);
        for (int y = 0; y < 16; ++y) for (int x = 0; x < 16; ++x) {
            const float mask = (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r ? 1.f : 0.f;
            masks.push_back(mask);
            images.push_back(mask + noise(random));
        }
    }
    return {dl::from_span<float>(images, {count, 1, 16, 16}), dl::from_span<float>(masks, {count, 1, 16, 16})};
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::cout << "usage: ctorch_segmentation <output-directory> [cpu|cuda]\n"; return 0; }
    const auto output = std::filesystem::path(argv[1]);
    std::filesystem::create_directories(output);
    const auto device = argc > 2 && std::string_view(argv[2]) == "cuda" ? dl::Device::cuda() : dl::Device::cpu();
    dl::manual_seed(12);
    dl::Generator sampling(13);
    dl::data::DataLoader train(circles(48, 21), {.batch_size = 8}, sampling);
    dl::data::DataLoader validation(circles(16, 22), {.batch_size = 8, .shuffle = false}, sampling);
    dl::data::PrefetchLoader batches(train, device, 2, true), validation_batches(validation, device);
    Segmenter model, evaluation;
    model.to(device); evaluation.to(device); evaluation.eval();
    dl::optim::AdamW optimizer(model.parameters(), {.learning_rate = .01});
    dl::training::ExponentialMovingAverage ema(model, .9);
    dl::training::EarlyStopping stopping({.patience = 3, .min_delta = 1e-4});
    dl::training::ReduceLROnPlateau schedule(optimizer, {{.patience = 1}, .5, 1e-5});
    dl::BinaryLossOptions binary;
    binary.positive_weight = dl::full({}, 3.0, {.device = device});
    auto objective = [&](const dl::Tensor& logits, const dl::Tensor& target) {
        return dl::binary_cross_entropy_with_logits(logits, target, binary)
            + dl::dice_loss(dl::sigmoid(logits), target, {2, 3});
    };
    dl::Index steps = 0;
    double best_loss = std::numeric_limits<double>::infinity();
    for (dl::Index epoch = 0; epoch < 6; ++epoch) {
        batches.reset();
        double train_loss = 0;
        while (auto batch = batches.next()) {
            optimizer.zero_grad();
            auto loss = objective(model(batch->input), batch->target);
            loss.backward(); optimizer.step(); ema.update(model);
            train_loss += loss.item<float>() * batch->input.size(0);
            ++steps;
        }
        ema.copy_to(evaluation);
        validation_batches.reset();
        dl::training::ConfusionMatrix metrics(2);
        double valid_loss = 0;
        {
            dl::NoGradGuard guard;
            while (auto batch = validation_batches.next()) {
                auto logits = evaluation(batch->input);
                valid_loss += objective(logits, batch->target).item<float>() * batch->input.size(0);
                metrics.update(dl::gt(logits, dl::zeros_like(logits)).astype(dl::DType::i64), batch->target.astype(dl::DType::i64));
            }
        }
        valid_loss /= 16;
        if (valid_loss < best_loss) {
            best_loss = valid_loss;
            dl::io::save(output / "best.ct", evaluation.state_dict());
        }
        const bool finished = stopping.step(valid_loss);
        schedule.step(valid_loss);
        auto state = dl::io::capture_checkpoint(model, optimizer, {.epoch = epoch + 1, .step = steps}, batches);
        dl::io::set_extra(state, "ema", ema.state_dict());
        dl::io::set_extra(state, "stopping", stopping.state_dict());
        dl::io::set_extra(state, "schedule", schedule.state_dict());
        state.extra.emplace_back("best_validation_loss", dl::full({}, best_loss, {.dtype = dl::DType::f64}));
        dl::io::save_checkpoint(output / "latest.train.ct", state);
        std::cout << "epoch=" << epoch + 1 << " train_loss=" << train_loss / 48
                  << " validation_loss=" << valid_loss << " foreground_iou=" << metrics.scores().iou[1] << '\n';
        if (finished) break;
    }
}
