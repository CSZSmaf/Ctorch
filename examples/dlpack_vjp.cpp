#include <dl/dl.h>
#include <dl/dlpack.h>

#include <array>
#include <iostream>

namespace {

/// 一个只使用 DLTensor 和标准容器的外部 CPU 算子，输入为二维 float32。
struct ExternalResult {
    std::array<std::int64_t, 2> shape, strides;
    std::vector<float> values;
    DLManagedTensorVersioned tensor{};
};

dl::DLPackTensor external_square(const DLTensor& input, float factor) {
    auto result = std::make_unique<ExternalResult>();
    const auto rows = input.shape[0], columns = input.shape[1];
    result->shape = {rows, columns};
    result->strides = {columns, 1};
    result->values.resize(static_cast<std::size_t>(rows * columns));
    const auto* values = reinterpret_cast<const float*>(
        static_cast<const std::byte*>(input.data) + input.byte_offset);
    for (std::int64_t row = 0; row < rows; ++row) {
        for (std::int64_t column = 0; column < columns; ++column) {
            const float value = values[row * input.strides[0] + column * input.strides[1]];
            result->values[row * columns + column] = factor * value * value;
        }
    }

    auto& tensor = result->tensor;
    tensor.version = {1, 0};
    tensor.manager_ctx = result.get();
    tensor.deleter = [](DLManagedTensorVersioned* value) {
        delete static_cast<ExternalResult*>(value->manager_ctx);
    };
    tensor.dl_tensor = {result->values.data(), {kDLCPU, 0}, 2, {kDLFloat, 32, 1},
        result->shape.data(), result->strides.data(), 0};
    dl::DLPackTensor owner(&tensor);
    result.release();
    return owner;
}

dl::Tensor scaled_square(const dl::Tensor& input, float factor) {
    return dl::custom_vjp({input},
        [factor](dl::VjpContext& context, std::span<const dl::Tensor> inputs) {
            context.save_for_backward({inputs[0]});
            auto descriptor = dl::to_dlpack(inputs[0]);
            // 前向结果保留外部容器，由最后一个张量视图触发其释放函数。
            return dl::from_dlpack(external_square(descriptor->dl_tensor, factor));
        },
        [factor](const dl::VjpContext& context, const dl::Tensor& gradient) {
            dl::VjpGradients contributions(context.input_count());
            if (context.needs_input_grad(0))
                contributions[0] = gradient * context.saved_tensors()[0] * (2.0 * factor);
            return contributions;
        });
}

} // namespace

int main() {
    const std::array<float, 6> values{-2, -1, 0, 1, 2, 3};
    auto input = dl::from_span<float>(values, {2, 3}, true);
    auto output = scaled_square(input.transpose(0, 1), 0.5f);
    dl::sum(output).backward();
    std::cout << "output";
    for (auto value : output.to_vector<float>()) std::cout << ' ' << value;
    std::cout << "\ngradient";
    for (auto value : input.grad()->to_vector<float>()) std::cout << ' ' << value;
    std::cout << '\n';
}
