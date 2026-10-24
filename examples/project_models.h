#pragma once

#include <dl/dl.h>

#include <memory>
#include <numeric>

namespace examples {

inline std::unique_ptr<dl::nn::Sequential> mnist_mlp() {
    auto model = std::make_unique<dl::nn::Sequential>();
    model->append<dl::nn::Flatten>("flatten");
    model->append<dl::nn::Linear>("hidden", 784, 128);
    model->append<dl::nn::ReLU>("activation");
    model->append<dl::nn::Linear>("output", 128, 10);
    return model;
}

// 28×28 输入经过两次无填充卷积和池化，得到 16×4×4 个特征。
inline std::unique_ptr<dl::nn::Sequential> lenet() {
    auto model = std::make_unique<dl::nn::Sequential>();
    model->append<dl::nn::Conv2d>("conv1", 1, 6, dl::Size2{5});
    model->append<dl::nn::ReLU>("relu1");
    model->append<dl::nn::MaxPool2d>("pool1");
    model->append<dl::nn::Conv2d>("conv2", 6, 16, dl::Size2{5});
    model->append<dl::nn::ReLU>("relu2");
    model->append<dl::nn::MaxPool2d>("pool2");
    model->append<dl::nn::Flatten>("flatten");
    model->append<dl::nn::Linear>("hidden1", 16 * 4 * 4, 120);
    model->append<dl::nn::ReLU>("relu3");
    model->append<dl::nn::Linear>("hidden2", 120, 84);
    model->append<dl::nn::ReLU>("relu4");
    model->append<dl::nn::Linear>("output", 84, 10);
    return model;
}

class ResidualBlock : public dl::nn::UnaryModule {
public:
    ResidualBlock(dl::Index input, dl::Index output, dl::Index stride = 1) {
        first_ = add_module<dl::nn::Conv2d>("conv1", input, output, dl::Size2{3},
            dl::Conv2dOptions{.stride = stride, .padding = 1}, false);
        first_norm_ = add_module<dl::nn::BatchNorm2d>("norm1", output);
        second_ = add_module<dl::nn::Conv2d>("conv2", output, output, dl::Size2{3},
            dl::Conv2dOptions{.padding = 1}, false);
        second_norm_ = add_module<dl::nn::BatchNorm2d>("norm2", output);
        // 降采样时用 1×1 投影同时调整捷径的通道数和空间尺寸。
        if (input != output || stride != 1) {
            shortcut_ = add_module<dl::nn::Sequential>("shortcut");
            shortcut_->append<dl::nn::Conv2d>("conv", input, output, dl::Size2{1},
                dl::Conv2dOptions{.stride = stride}, false);
            shortcut_->append<dl::nn::BatchNorm2d>("norm", output);
        }
    }

    dl::Tensor forward(const dl::Tensor& input) override {
        auto residual = shortcut_ ? (*shortcut_)(input) : input;
        auto hidden = dl::relu((*first_norm_)((*first_)(input)));
        return dl::relu((*second_norm_)((*second_)(hidden)) + residual);
    }

private:
    std::shared_ptr<dl::nn::Conv2d> first_, second_;
    std::shared_ptr<dl::nn::BatchNorm2d> first_norm_, second_norm_;
    std::shared_ptr<dl::nn::Sequential> shortcut_;
};

inline std::unique_ptr<dl::nn::Sequential> resnet20() {
    auto model = std::make_unique<dl::nn::Sequential>();
    model->append<dl::nn::Conv2d>("stem", 3, 16, dl::Size2{3}, dl::Conv2dOptions{.padding = 1}, false);
    model->append<dl::nn::BatchNorm2d>("stem_norm", 16);
    model->append<dl::nn::ReLU>("stem_relu");
    dl::Index input = 16;
    for (dl::Index block = 0; block < 9; ++block) {
        const dl::Index output = 16 << (block / 3);
        const dl::Index stride = block == 3 || block == 6 ? 2 : 1;
        model->append<ResidualBlock>("block" + std::to_string(block), input, output, stride);
        input = output;
    }
    model->append<dl::nn::AdaptiveAvgPool2d>("pool", dl::Size2{1});
    model->append<dl::nn::Flatten>("flatten");
    model->append<dl::nn::Linear>("output", 64, 10);
    return model;
}

struct TextModelOptions {
    dl::Index length = 32, dimensions = 32, layers = 2, heads = 4;
    double embedding_scale = 1.0;
};

inline constexpr TextModelOptions medium_text{128, 128, 4, 4, 0.02};

class ByteTransformer : public dl::nn::UnaryModule {
public:
    explicit ByteTransformer(TextModelOptions options = {})
        : positions_(register_buffer("positions", make_positions(options.length))),
          mask_(register_buffer("causal_mask", dl::causal_mask(options.length, options.length))) {
        auto embedding = [&](dl::Index count) {
            auto weight = dl::randn({count, options.dimensions}) * options.embedding_scale;
            weight.requires_grad_(true);
            return weight;
        };
        tokens_ = add_module<dl::nn::Embedding>("tokens", embedding(256));
        positions_embedding_ = add_module<dl::nn::Embedding>("positions_embedding", embedding(options.length));
        encoder_ = add_module<dl::nn::TransformerEncoder>("encoder", options.layers, options.dimensions,
            options.heads, 4 * options.dimensions, 0.1);
        // 输入和输出词表共享参数；生成时只计算当前前缀的有效位置。
        output_ = add_module<dl::nn::Linear>("output", tokens_->weight());
    }

    void set_attention_backend(dl::AttentionBackend backend) {
        attention_backend_ = backend;
        encoder_->set_attention_backend(backend, backend == dl::AttentionBackend::cudnn);
    }

    dl::Tensor forward(const dl::Tensor& input) override {
        const auto length = input.size(1);
        auto positions = bound_buffer(positions_).narrow(0, 0, length);
        auto hidden = (*tokens_)(input) + (*positions_embedding_)(positions);
        const auto mask = attention_backend_ == dl::AttentionBackend::math
            ? std::optional{bound_buffer(mask_).narrow(0, 0, length).narrow(1, 0, length)} : std::nullopt;
        return (*output_)(encoder_->forward(hidden, mask));
    }

private:
    static dl::Tensor make_positions(dl::Index length) {
        std::vector<dl::Index> values(static_cast<std::size_t>(length));
        std::iota(values.begin(), values.end(), dl::Index{0});
        return dl::from_span<dl::Index>(values, {length});
    }

    dl::Tensor positions_, mask_;
    dl::AttentionBackend attention_backend_ = dl::AttentionBackend::math;
    std::shared_ptr<dl::nn::Embedding> tokens_, positions_embedding_;
    std::shared_ptr<dl::nn::TransformerEncoder> encoder_;
    std::shared_ptr<dl::nn::Linear> output_;
};

} // namespace examples
