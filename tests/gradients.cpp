#include "gradcheck.h"

using test::directional_check;

int main() {
    return test::run([] {
        directional_check({-.7, .2, 1.2, -1.1, .5, .9}, {2, 3}, [](const auto& x) {
            return dl::sum(dl::pad(x.transpose(0, 1), {1, 2, 2, 0}, .5) * .7);
        });
        directional_check({-.7, .2, 1.2, -1.1, .5, .9}, {2, 3}, [](const auto& x) {
            return dl::cross_entropy(x, test::indices({2, 0}), dl::CrossEntropyOptions{
                .weight = test::tensor({.5, 2, 1}, {3}), .label_smoothing = .1});
        });
        directional_check({-.7, .2, 1.2, -1.1, .5, .9}, {2, 3}, [](const auto& x) {
            auto target = test::tensor({0, 1, 1, 0, 1, 0}, {2, 3});
            return dl::focal_loss(x, target) + dl::dice_loss(dl::sigmoid(x), target, {1});
        });
        // 常量长向量的均值保持常量，归一化输出及其和的梯度都为零。
        for (dl::Index width : {1001, 4096}) {
            for (float value : {0.1f, 1.1f, 10.1f, 1e6f}) {
                auto source = dl::full({width, 2}, value, {.requires_grad = true});
                auto input = source.transpose(0, 1);
                for (auto actual : dl::mean(input, {1}).to_vector<float>()) test::close(actual, value, 0, 0);
                for (auto actual : dl::var(input, {1}, 0).to_vector<float>()) test::close(actual, 0, 0, 0);
                auto layer = dl::layer_norm(input, {width});
                for (auto actual : layer.to_vector<float>()) test::close(actual, 0, 0, 0);
                dl::sum(layer).backward();
                for (auto actual : source.grad()->to_vector<float>()) test::close(actual, 0, 1e-10, 0);
                source.zero_grad();
                auto batch = dl::batch_norm(source, dl::BatchNormState{dl::zeros({2}), dl::ones({2})},
                    std::nullopt, std::nullopt, true, 1.0);
                for (auto actual : batch.output.to_vector<float>()) test::close(actual, 0, 0, 0);
                for (auto actual : batch.state->mean.to_vector<float>()) test::close(actual, value, 0, 0);
                for (auto actual : batch.state->variance.to_vector<float>()) test::close(actual, 0, 0, 0);
                dl::sum(batch.output).backward();
                for (auto actual : source.grad()->to_vector<float>()) test::close(actual, 0, 1e-10, 0);
            }
        }
        // 大偏移下的近常量数据保留小方差，非连续轴沿原布局归约。
        auto close_values = test::tensor({1e6, 1e6 + .125, 1e6 + .25, 1e6 + .375}, {2, 2})
            .astype(dl::DType::f32).detach();
        close_values.requires_grad_();
        auto variance = dl::var(close_values.transpose(0, 1), {0}, 0);
        test::values(variance, {.00390625, .00390625}, 1e-9, 1e-7);
        dl::sum(variance).backward();
        test::gradient(close_values, {-.0625, .0625, -.0625, .0625});

        directional_check({-0.7, 0.2, 1.2, -1.1, 0.5, 0.9}, {2, 3}, [](const auto& x) {
            return dl::sum(dl::gelu(x) + dl::sigmoid(x) + dl::tanh(x));
        });
        directional_check({-0.7, 0.2, 1.2, -1.1, 0.5, 0.9}, {2, 3}, [](const auto& x) {
            auto coefficients = test::tensor({0.5, -1, 2, -0.25, 1.5, 0.75}, {3, 2});
            return dl::sum(dl::relu(x.transpose(0, 1)) * coefficients);
        });
        directional_check({0.5, 0.9, 1.6, 2.1, 0.7, 1.3}, {2, 3}, [](const auto& x) {
            return dl::mean(dl::log(x) + dl::sqrt(x) + dl::exp(x * 0.1));
        });
        directional_check({0.5, 0.9, 1.6, 2.1, 0.7, 1.3}, {2, 3}, [](const auto& x) {
            auto numerator = test::tensor({0.2, 0.5, 0.7}, {3});
            return dl::sum(numerator / x);
        });
        directional_check({0.5, 0.9, 1.6, 2.1, 0.7, 1.3}, {2, 3}, [](const auto& x) {
            auto numerator = x.transpose(0, 1);
            auto denominator = x.slice(0, 0, 1).transpose(0, 1);
            return dl::sum(numerator / denominator);
        });
        directional_check({-0.7, 0.2, 1.2, -1.1, 0.5, 0.9}, {2, 3}, [](const auto& x) {
            auto flattened = x.transpose(0, 1).reshape({6});
            return dl::sum(flattened * flattened);
        });
        directional_check({-0.7, 0.2, 1.2, -1.1, 0.5, 0.9}, {2, 3}, [](const auto& x) {
            auto coefficients = test::tensor({0.2, 0.5, -0.3, 0.7, -0.4, 0.1}, {2, 3});
            return dl::sum(dl::softmax(x, 1) * coefficients);
        });
        directional_check({-0.7, 0.2, 1.2, -1.1, 0.5, 0.9}, {2, 3}, [](const auto& x) {
            return dl::sum(dl::amax(x.transpose(0, 1), {0}) * test::tensor({2, -3}, {2}));
        });
        directional_check({1000.1, 999.7, 1000.4, -1000.2, -999.5, -1000.8}, {2, 3}, [](const auto& x) {
            return dl::cross_entropy(x, test::indices({0, 2}));
        });
        directional_check({-12, -0.4, 0, 0.7, 3, 12}, {2, 3}, [](const auto& x) {
            auto logits = x.transpose(0, 1);
            auto targets = dl::sigmoid(logits * 0.1);
            return dl::binary_cross_entropy_with_logits(logits, targets);
        });
        directional_check({0.1, 0.3, -0.7, 0.4, 0.2, -0.1}, {2, 3}, [](const auto& x) {
            return dl::mean(dl::matmul(x, x.transpose(0, 1)));
        });
        directional_check({0.1, 0.3, -0.7, 0.4, 0.2, -0.1}, {3, 2}, [](const auto& x) {
            auto batches = test::tensor({1, 2, 3, 4, 5, 6, 2, -1, 1, 3, 0, 2}, {2, 2, 3});
            return dl::sum(dl::matmul(batches, x));
        });
        dl::nn::LayerNorm normalization({3}, 1e-5, false);
        directional_check({0.1, 0.3, -0.7}, {3}, [](const auto& x) {
            return dl::matmul(x, x);
        });
        directional_check({0.1, 0.3, -0.7, 0.4, 0.2, -0.1}, {2, 3}, [](const auto& x) {
            auto vector = x.reshape({6}).slice(0, 0, 6, 2);
            auto matrices = x.reshape({1, 2, 3}).expand({3, 2, 3});
            return dl::sum(dl::matmul(matrices, vector) * dl::matmul(vector, matrices.transpose(1, 2)));
        });
        directional_check({0.1, 0.3, -0.7, 0.4, 0.2, -0.1}, {2, 3}, [&](const auto& x) {
            return dl::sum(normalization(x) * test::tensor({0.2, -0.5, 0.9}, {3}));
        });
        directional_check({0.1, 0.3, -0.7, 0.4, 0.2, -0.1, 0.8, -0.5,
            1.1, 0.7, -0.3, 0.5, 0.1, -0.2, 0.4, 0.6}, {4, 2, 2}, [](const auto& x) {
            auto input = x.slice(0, 0, 2).transpose(1, 2);
            auto weight = x.slice(0, 2, 3).reshape({2, 2});
            auto bias = x.slice(0, 3, 4).reshape({2, 2});
            auto seed = test::tensor({0.2, -0.5, 0.9, 0.4, -0.3, 0.7, 0.1, -0.6}, {2, 2, 2});
            return dl::sum(dl::layer_norm(input, {2, 2}, weight, bias, 0.25) * seed);
        });

        // 打包输入、滤波器和偏置，使一次方向检查同时经过三条卷积梯度路径。
        std::vector<double> convolution(28);
        for (std::size_t i = 0; i < convolution.size(); ++i) convolution[i] = (static_cast<double>(i) - 11) * .1;
        directional_check(convolution, {28}, [](const auto& x) {
            auto input = x.slice(0, 0, 18).reshape({1, 2, 3, 3}).transpose(2, 3);
            auto weight = x.slice(0, 18, 26).reshape({2, 1, 2, 2});
            auto output = dl::conv2d(input, weight, x.slice(0, 26, 28), {.groups = 2});
            return dl::mean(output * output);
        });
        directional_check({.1, .7, .2, -.4, .5, .8, -.2, .3, .6}, {1, 1, 3, 3}, [](const auto& x) {
            auto average = dl::avg_pool2d(x.transpose(2, 3), {.kernel = 2, .stride = 1, .padding = 1}, false);
            auto maximum = dl::max_pool2d(x, {.kernel = 2, .stride = 1});
            return dl::sum(average * average) + dl::sum(maximum * maximum);
        });
        directional_check({.1, .3, .2, -.5, .8, -.4, -.1, .9}, {2, 2, 1, 2}, [](const auto& x) {
            auto output = dl::batch_norm(x).output;
            return dl::sum(output * test::tensor({.2, -.1, .3, .5, -.4, .8, -.2, .1}, x.shape()));
        });
        directional_check({.1, -.3, .5, .2, -.7, .4, .6, -.1, .9, .3, -.2, .8}, {3, 2, 2}, [](const auto& x) {
            auto output = dl::scaled_dot_product_attention(x.slice(0, 0, 1), x.slice(0, 1, 2),
                x.slice(0, 2, 3), dl::causal_mask(2, 2));
            return dl::sum(output * test::tensor({.1, .8, -.2, .4}, {1, 2, 2}));
        });
        directional_check({.1, .3, -.7, .4, .2, -.1}, {2, 3}, [](const auto& x) {
            auto output = dl::embedding(test::indices({0, 1, 0}), x);
            return dl::sum(output * output);
        });
        directional_check({.1, .3, -.7, .4, .2, -.1}, {2, 3}, [](const auto& x) {
            return dl::checkpoint({x, x}, [](std::span<const dl::Tensor> inputs) {
                return dl::sum(inputs[0] * inputs[1] + inputs[0]);
            });
        });
    });
}
